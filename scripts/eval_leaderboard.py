#!/usr/bin/env python3
"""Rank every dialect-conversion run (SFT + all GRPO arms) per region AND overall.

Where ``eval_grpo_checkpoints.py`` sweeps the checkpoints *inside one* run, this sweeps
the *final adapters across* runs, on each requested region and on a weighted aggregate,
so "did GRPO beat SFT, which arm, and where?" gets one decision-ready report.

Scenarios:
  * one region:        --target_do gangwondo
  * several regions:   --target_do gangwondo,gyeongsangdo   (per-region + OVERALL tables)
  * all available:     --all_regions                        (every classifier-supported
                                                             region present in the data)

Every metric is printed with its direction arrow (↑/↓ — see the glossary) so e.g.
copy_margin is unambiguously higher-is-better. Ranking is by ``reconstruction_bleu``
(proxy-independent; plan §3 A4 / MO-GRPO arXiv:2509.22047). Each table reports the
Pareto frontier over (copy_margin × reconstruction_bleu) and paired-bootstrap
significance vs SFT (Koehn 2004). When data for jeju/jeolla/chungcheong is added and the
classifier learns them, they appear automatically.

Writes outputs/eval_logs/leaderboard_<scope>_<ts>.{json,md} (per-region + overall).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import fire
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from ko_dialect.evaluation import grpo_runs
from ko_dialect.evaluation.generation import generate_batched
from ko_dialect.evaluation.leaderboard import (
    DIALECTNESS_AXIS,
    FIDELITY_AXIS,
    LEADERBOARD_METRICS,
    RunSpec,
    aggregate_rows,
    build_leaderboard_payload,
    discover_runs,
    evaluate_run,
    format_leaderboard_table,
    pareto_frontier,
    rank_rows,
)
from ko_dialect.evaluation.metric_registry import glossary_markdown
from ko_dialect.evaluation.metrics import DO_TO_LABEL
from ko_dialect.evaluation.significance import PairedResult, paired_bootstrap, significance_marker
from ko_dialect.models import TextCNNForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("leaderboard")

BASE = "outputs/sft_merged"
CLS = "outputs/classifier_clean"


def _make_generate_fn(device, batch_size: int = 16, max_new_tokens: int = 64):
    def generate_fn(model, tok, prompts):
        return generate_batched(
            model,
            tok,
            prompts,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            max_length=448,
        )

    return generate_fn


def _resolve_specs(runs: str | None, base: str, include_sft: bool) -> list[RunSpec]:
    if not runs:
        return discover_runs("outputs", base, include_sft=include_sft)
    specs: list[RunSpec] = []
    if include_sft:
        specs.append(RunSpec("SFT", base, None))
    for run in (r.strip() for r in runs.split(",") if r.strip()):
        specs.append(RunSpec(Path(run).name, base, run))
    return specs


def _resolve_regions(target_do: str, all_regions: bool, ds) -> list[str]:
    """Regions to evaluate: explicit list, or every classifier-supported region in data."""
    present = set(ds.unique("do"))
    supported = [do for do in DO_TO_LABEL if do in present]
    if all_regions:
        if not supported:
            raise SystemExit("No classifier-supported regions found in the dataset.")
        return supported
    wanted = [r.strip() for r in target_do.split(",") if r.strip()]
    for r in wanted:
        if r not in DO_TO_LABEL:
            logger.warning("Region %s is not in the classifier (TDR/DFS will be 0).", r)
    return wanted


def _region_slice(ds_valid, target_do: str, n: int):
    ds = ds_valid.filter(
        lambda x: (
            x["do"] == target_do and x["direction"] == "std2dia" and x["standard"] != x["dialect"]
        )
    )
    ds = ds.select(range(grpo_runs.clamp_select_count(len(ds), n)))
    return (
        list(ds["prompt"]),
        list(ds["dialect"]),
        list(ds["standard"]),
        [m or [] for m in ds["dialect_eojeol_map"]],
    )


def _significance_block(recon_scores: dict[str, list[float]], baseline_tag: str) -> dict:
    sig: dict[str, dict] = {}
    if baseline_tag not in recon_scores:
        return sig
    base = recon_scores[baseline_tag]
    for tag, scores in recon_scores.items():
        if tag == baseline_tag or len(scores) != len(base):
            continue
        sig[tag] = paired_bootstrap(scores, base).as_dict()
    return sig


def _print_scope(title: str, ranked, frontier, significance, baseline_tag, select_by):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    print(format_leaderboard_table(ranked, frontier=frontier))
    print(
        f"\nPareto frontier ({DIALECTNESS_AXIS}↑ × {FIDELITY_AXIS}↑): "
        f"{', '.join(frontier)}  (★ = defensible, no run dominates it)"
    )
    if significance:
        print(f"Paired bootstrap vs {baseline_tag} (per-sentence recon BLEU, Koehn 2004):")
        for tag, s in significance.items():
            marker = significance_marker(
                PairedResult(s["delta"], s["p_value"], s["win_rate"], s["n_boot"])
            )
            verdict = "significant" if s["significant_05"] else "n.s."
            print(f"  {tag:12s} Δ={s['delta']:+.2f}  p={s['p_value']:.3f}  {marker} ({verdict})")


def main(
    target_do: str = "gangwondo",
    all_regions: bool = False,
    n: int = 150,
    select_by: str = "reconstruction_bleu",
    runs: str | None = None,
    include_sft: bool = True,
    base: str = BASE,
    classifier_path: str = CLS,
    cls_tokenizer_path: str = CLS,
    dataset_path: str = "outputs/datasets/grpo",
    max_new_tokens: int = 64,
    out_dir: str = "outputs/eval_logs",
    wandb_project: str | None = None,
    mlflow_experiment: str | None = None,
):
    """Evaluate every run per region + overall and emit a ranked, annotated leaderboard.

    Pass ``--wandb_project ko-dialect`` and/or ``--mlflow_experiment ko-dialect`` to also
    push per-region, per-run metric panels (eval/<region>/<run>/<metric>) to that tracker;
    omit both for the plain local run.
    """
    specs = _resolve_specs(runs, base, include_sft)
    if not specs:
        raise SystemExit("No runs to evaluate (no outputs/grpo* adapters found).")
    baseline_tag = next((s.tag for s in specs if s.adapter is None), "SFT")

    ds_valid = load_from_disk(dataset_path)["valid"]
    regions = _resolve_regions(target_do, all_regions, ds_valid)

    track = bool(wandb_project or mlflow_experiment)
    run_name = f"leaderboard_{'_'.join(regions)}"
    started_mlflow = False
    if wandb_project:
        import wandb

        wandb.init(
            project=wandb_project,
            name=run_name,
            job_type="eval",
            config={"n": n, "select_by": select_by, "runs": [s.tag for s in specs]},
        )
    if mlflow_experiment:
        import mlflow

        # Reuse an already-active run (an external caller's, or one left by a prior
        # in-process call) rather than clobbering it — only start, and later end, a run
        # we own. Avoids both "Run already active" on re-entry and ending someone else's
        # run. A stale run we don't own is closed by MLflow's atexit hook on process exit.
        if mlflow.active_run() is None:
            mlflow.set_experiment(mlflow_experiment)
            mlflow.start_run(run_name=run_name)
            started_mlflow = True
        mlflow.log_params({"n": n, "select_by": select_by, "n_runs": len(specs)})
    logger.info("Runs: %s", ", ".join(s.tag for s in specs))
    logger.info("Regions: %s", ", ".join(regions))

    cls_tok = AutoTokenizer.from_pretrained(cls_tokenizer_path, local_files_only=True)
    if cls_tok.pad_token is None:
        cls_tok.pad_token = cls_tok.eos_token
    classifier = TextCNNForSequenceClassification.from_pretrained(classifier_path)
    model_tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if model_tok.pad_token is None:
        model_tok.pad_token = model_tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_fn = _make_generate_fn(device, max_new_tokens=max_new_tokens)

    from sacrebleu.metrics import BLEU

    sent_bleu = BLEU(effective_order=True)

    per_region_rows: dict[str, list[tuple[str, dict]]] = {}
    per_region_payload: dict[str, dict] = {}
    region_weights: dict[str, float] = {}
    pooled_recon: dict[str, list[float]] = {tag: [] for tag in (s.tag for s in specs)}

    for region in regions:
        prompts, gold, src, emaps = _region_slice(ds_valid, region, n)
        if not prompts:
            logger.warning("No std2dia samples for %s; skipping.", region)
            continue
        region_weights[region] = float(len(prompts))
        logger.info("=== region %s: %d samples ===", region, len(prompts))

        rows: list[tuple[str, dict]] = []
        recon_scores: dict[str, list[float]] = {}
        for spec in specs:
            logger.info("  [%s] %s", region, spec.tag)
            metrics, _outs, rev = evaluate_run(
                spec,
                prompts=prompts,
                gold=gold,
                src=src,
                emaps=emaps,
                target_do=region,
                classifier=classifier,
                cls_tokenizer=cls_tok,
                model_tokenizer=model_tok,
                generate_fn=generate_fn,
            )
            rows.append((spec.tag, metrics))
            scores = [sent_bleu.sentence_score(r, [s]).score for r, s in zip(rev, src)]
            recon_scores[spec.tag] = scores
            pooled_recon[spec.tag].extend(scores)

        ranked = rank_rows(rows, select_by)
        frontier = pareto_frontier(ranked)
        significance = _significance_block(recon_scores, baseline_tag)
        payload = build_leaderboard_payload(
            ranked,
            select_by=select_by,
            target_do=region,
            n_samples=len(prompts),
            baseline_tag=baseline_tag,
        )
        payload["significance_vs_baseline"] = significance
        per_region_rows[region] = ranked
        per_region_payload[region] = payload
        if track:
            from ko_dialect.monitoring import log_panels, metrics_panel

            for tag, metrics in ranked:
                log_panels(metrics_panel(metrics, f"eval/{region}/{tag}"))
        _print_scope(
            f"LEADERBOARD — {region} std2dia, n={len(prompts)}, by {select_by}↑",
            ranked,
            frontier,
            significance,
            baseline_tag,
            select_by,
        )

    if not per_region_rows:
        raise SystemExit("No region produced samples; nothing to rank.")

    # ---- OVERALL aggregate (only meaningful with ≥2 regions) ----
    overall_payload = None
    if len(per_region_rows) >= 2:
        agg = aggregate_rows(per_region_rows, weights=region_weights)
        agg_ranked = rank_rows(agg, select_by)
        agg_frontier = pareto_frontier(agg_ranked)
        agg_sig = _significance_block(pooled_recon, baseline_tag)
        overall_payload = build_leaderboard_payload(
            agg_ranked,
            select_by=select_by,
            target_do="OVERALL",
            n_samples=int(sum(region_weights.values())),
            baseline_tag=baseline_tag,
        )
        overall_payload["significance_vs_baseline"] = agg_sig
        overall_payload["regions"] = list(per_region_rows)
        overall_payload["region_weights"] = region_weights
        _print_scope(
            f"LEADERBOARD — OVERALL (sample-weighted over {', '.join(per_region_rows)})",
            agg_ranked,
            agg_frontier,
            agg_sig,
            baseline_tag,
            select_by,
        )

    print("\n" + glossary_markdown(LEADERBOARD_METRICS))
    print(
        "\nNOTE: selection metric is proxy-independent; tdr/dfs/jscore are monitoring-only.\n"
        "NOTE: no single winner is forced — inspect the Pareto frontier (★) per region "
        "and overall; ≈ means the recon gap vs baseline is within sampling noise."
    )
    scope_runs = " ".join((overall_payload or per_region_payload[regions[0]])["pareto_frontier"])
    print(
        "\nTo compare the Pareto-optimal runs at inference time (no commit needed):\n"
        f"  python scripts/serve_compare.py --runs '{scope_runs}' "
        f"--target_do {regions[0]}"
    )

    # ---- persist ----
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    scope = "overall" if overall_payload else regions[0]
    record = {
        "schema": "ko_dialect.leaderboard_multi/v1",
        "select_by": select_by,
        "regions": list(per_region_rows),
        "per_region": per_region_payload,
        "overall": overall_payload,
        "glossary_markdown": glossary_markdown(LEADERBOARD_METRICS),
    }
    json_path = out_root / f"leaderboard_{scope}_{ts}.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf-8")

    md_sections = [f"# Leaderboard — {', '.join(per_region_rows)} (by {select_by}↑)\n"]
    scopes = list(per_region_rows.items())
    if overall_payload:
        md_sections.append("## OVERALL (sample-weighted)\n")
        md_sections.append(
            format_leaderboard_table(
                rank_rows(
                    [
                        (t, m)
                        for t, m in [
                            (tag, overall_payload["rows"][tag])
                            for tag in overall_payload["ranking"]
                        ]
                    ],
                    select_by,
                ),
                frontier=overall_payload["pareto_frontier"],
            )
        )
    for region, ranked in scopes:
        md_sections.append(f"\n## {region}\n")
        md_sections.append(
            format_leaderboard_table(ranked, frontier=per_region_payload[region]["pareto_frontier"])
        )
    md_sections.append("\n## Metric glossary\n")
    md_sections.append(glossary_markdown(LEADERBOARD_METRICS))
    md_path = out_root / f"leaderboard_{scope}_{ts}.md"
    md_path.write_text("\n".join(md_sections) + "\n", "utf-8")
    print(f"\nWrote {json_path}\n      {md_path}")

    if wandb_project:
        import wandb

        wandb.finish()
    if started_mlflow:
        import mlflow

        mlflow.end_run()


if __name__ == "__main__":
    fire.Fire(main)
