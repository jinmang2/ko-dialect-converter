"""Cross-run evaluation leaderboard for the dialect-conversion models.

``eval_grpo_checkpoints.py`` sweeps the *checkpoints inside one* GRPO run; this module
ranks the *final adapters across different* runs (SFT base, grpo_500, grpo_v2,
grpo_arm1, grpo_arm2, ...) on the same held-out slice so the question "did GRPO —
and which arm — actually beat SFT?" has one answer table instead of N separate logs.

Selection metric (plan §3 A4; MO-GRPO arXiv:2509.22047):
  Ranking defaults to ``reconstruction_bleu`` — a proxy-independent signal (the
  classifier/eojeol_map that GRPO trains against never enters it), so the leaderboard
  cannot be gamed by reward over-optimisation the way a TDR/J-score ranking would be.

The pure helpers here (discovery, ranking, table/JSON formatting) carry no GPU
dependency and are unit-tested; the generation harness lives in ``evaluate_run``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metric_registry import glossary_markdown, header_label, orient
from .metrics import _remap_unit, evaluate_all

logger = logging.getLogger(__name__)

# Metrics shown in the leaderboard table, in column order. ``higher = better`` for all.
LEADERBOARD_METRICS = (
    "reconstruction_bleu",
    "copy_margin",
    "tdr",
    "dfs",
    "chrf",
    "bleu",
    "eojeol_accuracy",
    "jscore",
)

# Proxy-independent default — see module docstring / plan §3 A4.
DEFAULT_SELECT_BY = "reconstruction_bleu"

# The two orthogonal axes of dialect conversion. Style strength and content preservation
# are in empirical tension in the text-style-transfer literature (Hu et al. 2022 survey,
# arXiv:2010.12742, benchmarks the trade-off across 19 systems; the explicit negative-
# correlation finding is Mukherjee, Kasner & Dušek 2022, arXiv:2312.14708; direct-reward
# TST: Liu/Neubig/Wieting, arXiv:2010.12771). No single number wins, so we report the
# Pareto frontier over them instead of forcing a rank. (See docs/REFERENCES.md A4/A5.)
DIALECTNESS_AXIS = "copy_margin"  # copy-debiased conversion strength (gold vs source)
FIDELITY_AXIS = "reconstruction_bleu"  # proxy-independent content preservation


@dataclass(frozen=True)
class RunSpec:
    """One model to evaluate: a base checkpoint plus an optional LoRA adapter."""

    tag: str
    base: str
    adapter: str | None


def discover_runs(
    outputs_dir: str | Path,
    base: str,
    *,
    include_sft: bool = True,
    glob_pattern: str = "grpo*",
    sft_tag: str = "SFT",
) -> list[RunSpec]:
    """Find every run directory holding a final ``adapter_config.json``.

    The SFT baseline (``base`` with no adapter) is prepended when ``include_sft`` so
    every GRPO arm is ranked against the model it was fine-tuned from.
    """
    root = Path(outputs_dir)
    specs: list[RunSpec] = []
    if include_sft:
        specs.append(RunSpec(sft_tag, base, None))

    base_resolved = str(Path(base))
    for path in sorted(root.glob(glob_pattern)):
        if not path.is_dir():
            continue
        if not (path / "adapter_config.json").exists():
            continue
        if str(path) == base_resolved:
            continue
        specs.append(RunSpec(path.name, base, str(path)))
    return specs


def rank_rows(
    rows: list[tuple[str, dict[str, Any]]],
    metric: str,
    *,
    descending: bool = True,
) -> list[tuple[str, dict[str, Any]]]:
    """Stable-sort ``(tag, metrics)`` rows by ``metric`` (missing/NaN sink to the bottom).

    Ties keep input order, so a run that merely matches the leader is never promoted
    above it by sort noise.
    """

    def key(item: tuple[str, dict[str, Any]]) -> float:
        value = item[1].get(metric)
        if not isinstance(value, (int, float)):
            return float("-inf") if descending else float("inf")
        as_float = float(value)
        if as_float != as_float:  # NaN
            return float("-inf") if descending else float("inf")
        return as_float

    return sorted(rows, key=key, reverse=descending)


def deltas_vs_baseline(
    rows: list[tuple[str, dict[str, Any]]],
    baseline_tag: str,
    metrics: tuple[str, ...] = LEADERBOARD_METRICS,
) -> dict[str, dict[str, float]]:
    """Per-run ``metric - baseline_metric`` for each leaderboard metric.

    Lets a reader see at a glance whether an arm actually moved the proxy-independent
    signal relative to SFT, not just its absolute score.
    """
    baseline = next((m for tag, m in rows if tag == baseline_tag), None)
    if baseline is None:
        return {}
    out: dict[str, dict[str, float]] = {}
    for tag, m in rows:
        if tag == baseline_tag:
            continue
        out[tag] = {
            k: float(m[k]) - float(baseline[k])
            for k in metrics
            if isinstance(m.get(k), (int, float)) and isinstance(baseline.get(k), (int, float))
        }
    return out


def _axis_value(metrics: dict[str, Any], axis: str) -> float:
    """Direction-oriented axis value (higher = better) for Pareto/ranking; missing → -inf."""
    v = metrics.get(axis)
    if not isinstance(v, (int, float)) or float(v) != float(v):
        return float("-inf")
    return orient(axis, float(v))


def pareto_frontier(
    rows: list[tuple[str, dict[str, Any]]],
    axes: tuple[str, ...] = (DIALECTNESS_AXIS, FIDELITY_AXIS),
) -> list[str]:
    """Tags of runs not dominated on ``axes`` (all higher = better).

    A run is dominated when another run is ≥ on every axis and > on at least one. The
    frontier is the set of defensible choices — e.g. SFT (max fidelity) and arm2 (max
    conversion) can both be Pareto-optimal, which is exactly why "pick one" is the wrong
    question. Reference: TST trade-off curves (arXiv:2010.12742 §eval).
    """

    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        ge_all = all(_axis_value(a, k) >= _axis_value(b, k) for k in axes)
        gt_any = any(_axis_value(a, k) > _axis_value(b, k) for k in axes)
        return ge_all and gt_any

    frontier = []
    for i, (tag, m) in enumerate(rows):
        if not any(dominates(other, m) for j, (_, other) in enumerate(rows) if j != i):
            frontier.append(tag)
    return frontier


def harmonic_joint(values: list[float]) -> float:
    """Harmonic mean of (0,1] axis scores — 0 if any axis is ≤ 0.

    The harmonic mean punishes imbalance harder than the geometric mean (a model that
    aces style but tanks content scores near 0). This is **our own** summary statistic:
    verified against arXiv:2010.12771 (§2.3, Eq.10) — that paper combines its rewards by a
    weighted *sum* for training and selects checkpoints by the *arithmetic mean* of style
    accuracy + BLEU; it uses no harmonic mean. (We deliberately avoid their style-accuracy-in-
    the-loop selection — that is proxy-dependent; see the metrics.py ADR.) Unlike ``jscore`` this is meant
    to *summarise* a run, but like jscore it must not drive checkpoint selection
    (circularity, plan §3 A4) — selection stays on the proxy-independent fidelity axis.
    """
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def joint_score(
    metrics: dict[str, Any],
    *,
    dialectness_axis: str = DIALECTNESS_AXIS,
    fidelity_axis: str = FIDELITY_AXIS,
    dialectness_range: tuple[float, float] = (-100.0, 100.0),
    fidelity_range: tuple[float, float] = (0.0, 100.0),
) -> float:
    """Harmonic-mean joint of a run's dialectness + fidelity, each remapped to (0,1]."""
    d = _remap_unit(_axis_value(metrics, dialectness_axis), *dialectness_range)
    f = _remap_unit(_axis_value(metrics, fidelity_axis), *fidelity_range)
    return harmonic_joint([d, f])


def aggregate_rows(
    per_region: dict[str, list[tuple[str, dict[str, Any]]]],
    *,
    weights: dict[str, float] | None = None,
    metrics: tuple[str, ...] = LEADERBOARD_METRICS,
) -> list[tuple[str, dict[str, Any]]]:
    """Combine per-region rows into one OVERALL row-set via a (weighted) mean per metric.

    ``per_region`` maps region → ``[(tag, metrics), ...]``. A run is aggregated only over
    the regions where it was evaluated (so adding jeju/jeolla/chungcheong later just adds
    keys). ``weights`` (e.g. per-region sample counts) defaults to equal weighting. The
    aggregate keeps a run only if it appears in at least one region.
    """
    weights = weights or {}
    # tag -> metric -> [(value, weight)]
    acc: dict[str, dict[str, list[tuple[float, float]]]] = {}
    order: list[str] = []
    for region, rows in per_region.items():
        w = float(weights.get(region, 1.0))
        for tag, m in rows:
            if tag not in acc:
                acc[tag] = {}
                order.append(tag)
            for k in metrics:
                v = m.get(k)
                if isinstance(v, (int, float)) and float(v) == float(v):
                    acc[tag].setdefault(k, []).append((float(v), w))

    out: list[tuple[str, dict[str, Any]]] = []
    for tag in order:
        agg: dict[str, Any] = {"n_regions": 0}
        region_count = 0
        for k in metrics:
            pairs = acc[tag].get(k)
            if not pairs:
                continue
            tot_w = sum(w for _, w in pairs) or 1.0
            agg[k] = sum(v * w for v, w in pairs) / tot_w
            region_count = max(region_count, len(pairs))
        agg["n_regions"] = region_count
        out.append((tag, agg))
    return out


def format_leaderboard_table(
    ranked: list[tuple[str, dict[str, Any]]],
    metrics: tuple[str, ...] = LEADERBOARD_METRICS,
    *,
    frontier: list[str] | None = None,
) -> str:
    """Render ranked rows as a GitHub-flavoured markdown table.

    When ``frontier`` is given, Pareto-optimal runs are flagged with ``★`` in the run
    column so the defensible-choice set is visible at a glance.
    """
    on_frontier = set(frontier or [])
    header = "| rank | run | " + " | ".join(header_label(m) for m in metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 2)
    lines = [header, sep]
    for rank, (tag, m) in enumerate(ranked, 1):
        cells = []
        for k in metrics:
            v = m.get(k)
            cells.append(f"{float(v):.3f}" if isinstance(v, (int, float)) else "n/a")
        label = f"{tag} ★" if tag in on_frontier else tag
        lines.append(f"| {rank} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_leaderboard_payload(
    ranked: list[tuple[str, dict[str, Any]]],
    *,
    select_by: str,
    target_do: str,
    n_samples: int,
    baseline_tag: str = "SFT",
) -> dict[str, Any]:
    """Assemble a machine-readable leaderboard record (JSON-serialisable)."""
    best_tag = ranked[0][0] if ranked else None
    frontier = pareto_frontier(ranked)
    return {
        "schema": "ko_dialect.leaderboard/v2",
        "target_do": target_do,
        "n_samples": n_samples,
        "select_by": select_by,
        "best_run": best_tag,
        "ranking": [tag for tag, _ in ranked],
        "pareto_frontier": frontier,
        "pareto_axes": [DIALECTNESS_AXIS, FIDELITY_AXIS],
        "joint_score": {tag: round(joint_score(m), 4) for tag, m in ranked},
        "metric_directions": {m: header_label(m) for m in LEADERBOARD_METRICS},
        "glossary_markdown": glossary_markdown(LEADERBOARD_METRICS),
        "rows": {tag: metrics for tag, metrics in ranked},
        "deltas_vs_baseline": deltas_vs_baseline(ranked, baseline_tag),
    }


# ---------------------------------------------------------------------------
# Generation harness (GPU) — kept thin; pure logic above stays import-light.
# ---------------------------------------------------------------------------

REVERSE_PROMPT = "다음 방언 문장을 표준어로 바꿔줘.\n방언: {dialect}\n표준어: "


def build_dia2std_prompt(dialect_text: str) -> str:
    """Reverse prompt (dialect → standard) for the reconstruction-BLEU pass."""
    return REVERSE_PROMPT.format(dialect=dialect_text)


def make_classifier_embed_fn(classifier, cls_tokenizer, device, *, max_length: int = 128):
    """Sentence embedder = the dialect classifier's penultimate features.

    DIA-REFINE (arXiv:2511.06680) computes DFS with "our fine-tuned classifier, final
    classification layer removed". Our TextCNN already exposes ``hidden_states`` (the
    concatenated max-pooled conv features feeding the classifier head), so this is the
    faithful analogue. Returns a callable ``list[str] -> (N, D)`` tensor for ``compute_dfs``.
    """
    import torch

    @torch.no_grad()
    def embed_fn(texts: list[str]):
        enc = cls_tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        out = classifier(enc["input_ids"].to(device))
        return out.hidden_states

    return embed_fn


def evaluate_run(
    spec: RunSpec,
    *,
    prompts: list[str],
    gold: list[str],
    src: list[str],
    emaps: list[list[dict[str, Any]]],
    target_do: str,
    classifier,
    cls_tokenizer,
    model_tokenizer,
    generate_fn,
    with_dfs: bool = True,
) -> tuple[dict[str, float], list[str], list[str]]:
    """Load ``spec``, generate forward + reverse, and score with ``evaluate_all``.

    ``generate_fn(model, tokenizer, prompts) -> list[str]`` is injected so the GPU
    path is swappable (and mockable in tests). When ``with_dfs`` the DIA-REFINE DFS axis
    is computed from the classifier's penultimate embedding. Returns
    ``(metrics, forward_outputs, reverse_outputs)`` so callers can run paired
    significance on per-sentence scores.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        spec.base, torch_dtype=torch.float16, device_map="auto"
    )
    if spec.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, spec.adapter).merge_and_unload()
    model.eval()
    device = next(model.parameters()).device
    classifier.to(device).eval()

    outs = generate_fn(model, model_tokenizer, prompts)
    rev_prompts = [build_dia2std_prompt(o) for o in outs]
    rev = generate_fn(model, model_tokenizer, rev_prompts)

    embed_fn = make_classifier_embed_fn(classifier, cls_tokenizer, device) if with_dfs else None
    metrics = evaluate_all(
        outputs=outs,
        dialect_refs=gold,
        standard_refs=src,
        dialect_eojeol_maps=emaps,
        target_do=target_do,
        classifier=classifier,
        cls_tokenizer=cls_tokenizer,
        embed_fn=embed_fn,
        reverse_outputs=rev,
    )
    del model
    torch.cuda.empty_cache()
    return metrics, outs, rev
