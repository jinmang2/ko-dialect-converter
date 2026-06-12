"""Build an evidence-rich Hugging Face model card from a leaderboard record.

The whole reason for the leaderboard is decision support — "which model do we ship?" —
so when a model is pushed to the Hub its card should carry that evidence: the per-region
metrics (with direction arrows), which regions it is Pareto-optimal in, and whether its
difference from the SFT baseline is statistically significant. This module is pure
(string in, string out) so the card can be unit-tested without network or a model.
"""

from __future__ import annotations

from typing import Any

from .metric_registry import header_label

_LANG_TAGS = ["ko"]


def _yaml_frontmatter(
    *, base_model: str, license: str, tags: list[str], pipeline_tag: str
) -> str:
    tag_lines = "\n".join(f"  - {t}" for t in tags)
    lang_lines = "\n".join(f"  - {ln}" for ln in _LANG_TAGS)
    return (
        "---\n"
        f"language:\n{lang_lines}\n"
        f"license: {license}\n"
        f"base_model: {base_model}\n"
        f"pipeline_tag: {pipeline_tag}\n"
        f"tags:\n{tag_lines}\n"
        "---\n"
    )


def _run_eval_table(run_tag: str, record: dict[str, Any], metrics: tuple[str, ...]) -> str:
    """One row per region for *this* run, columns = metrics with direction arrows."""
    per_region = record.get("per_region", {})
    header = "| region | " + " | ".join(header_label(m) for m in metrics) + " | Pareto |"
    sep = "|" + "---|" * (len(metrics) + 2)
    lines = [header, sep]
    for region, payload in per_region.items():
        row = payload.get("rows", {}).get(run_tag)
        if row is None:
            continue
        cells = []
        for m in metrics:
            v = row.get(m)
            cells.append(f"{float(v):.3f}" if isinstance(v, (int, float)) else "n/a")
        star = "★" if run_tag in payload.get("pareto_frontier", []) else ""
        lines.append(f"| {region} | " + " | ".join(cells) + f" | {star} |")
    overall = record.get("overall")
    if overall and run_tag in overall.get("rows", {}):
        row = overall["rows"][run_tag]
        cells = [
            f"{float(row.get(m)):.3f}" if isinstance(row.get(m), (int, float)) else "n/a"
            for m in metrics
        ]
        star = "★" if run_tag in overall.get("pareto_frontier", []) else ""
        lines.append("| **OVERALL** | " + " | ".join(cells) + f" | {star} |")
    return "\n".join(lines)


def _significance_note(run_tag: str, record: dict[str, Any]) -> str:
    notes = []
    scopes = dict(record.get("per_region", {}))
    if record.get("overall"):
        scopes["OVERALL"] = record["overall"]
    for scope, payload in scopes.items():
        sig = (payload.get("significance_vs_baseline") or {}).get(run_tag)
        if not sig:
            continue
        verdict = "significant" if sig.get("significant_05") else "n.s."
        notes.append(
            f"- **{scope}**: Δrecon_bleu vs SFT = {sig['delta']:+.2f} "
            f"(p={sig['p_value']:.3f}, {verdict})"
        )
    return "\n".join(notes)


def build_model_card(
    *,
    repo_id: str,
    run_tag: str,
    base_model: str,
    record: dict[str, Any] | None = None,
    license: str = "apache-2.0",
    metrics: tuple[str, ...] = (
        "reconstruction_bleu",
        "copy_margin",
        "tdr",
        "dfs",
        "eojeol_accuracy",
    ),
    extra_tags: tuple[str, ...] = (),
) -> str:
    """Assemble the full model-card markdown string for ``run_tag``."""
    tags = ["korean", "dialect", "translation", "qwen2", "lora", *extra_tags]
    parts = [
        _yaml_frontmatter(
            base_model=base_model,
            license=license,
            tags=tags,
            pipeline_tag="text-generation",
        ),
        f"\n# {repo_id}\n",
        "Korean **standard → dialect** translation (방언 어투 변환) with a small causal LM "
        f"(`{base_model}` + LoRA). Run: **{run_tag}**.\n",
        "## Intended use\n",
        "Convert standard Korean into a target regional dialect (강원도/경상도, more to "
        "come). Trained for `std2dia`; the reverse direction is used only for evaluation.\n",
        "## How to load\n",
        "```python\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        f'tok = AutoTokenizer.from_pretrained("{repo_id}")\n'
        f'model = AutoModelForCausalLM.from_pretrained("{repo_id}", torch_dtype="float16")\n'
        "```\n",
    ]

    if record is not None:
        select_by = record.get("select_by", "reconstruction_bleu")
        parts += [
            "## Evaluation\n",
            f"Held-out std2dia slice, ranked by `{select_by}` (proxy-independent; "
            "MO-GRPO arXiv:2509.22047). ★ = Pareto-optimal over "
            "(copy_margin↑ × reconstruction_bleu↑) — a defensible choice no other run "
            "dominates. Significance vs the SFT baseline is a paired bootstrap "
            "(Koehn 2004).\n",
            _run_eval_table(run_tag, record, metrics),
            "\n### Significance vs SFT baseline\n",
            _significance_note(run_tag, record) or "_(baseline run — no delta)_",
            "\n### Metric glossary\n",
            record.get("glossary_markdown", ""),
        ]

    parts += [
        "\n## Training\n",
        "- Base: QLoRA SFT on AI-Hub Korean dialect speech, then (for GRPO runs) "
        "Stage-3 GRPO with fidelity-anchored rewards.\n"
        "- Hardware: RTX 2060 (Turing, fp16 only — no bf16).\n",
        "## Limitations\n",
        "- Dialect conversion has many valid outputs; exact-match is low by nature.\n"
        "- TDR/DFS are classifier proxies (monitoring only). Select on reconstruction_bleu "
        "+ qualitative judgement, never on the reward proxy (circularity).\n",
    ]
    return "\n".join(parts)
