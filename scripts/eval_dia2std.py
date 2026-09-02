#!/usr/bin/env python3
"""Evaluate the **deployed** direction: dialect → standard, per region, against the copy floor.

Why this exists next to ``eval_leaderboard.py``: that script filters to
``direction == "std2dia"`` (standard → dialect), and so does ``compare_sft_grpo.py``, and
so does the GRPO prompt set (``stage0_build_datasets.py``). But the shipped task is the
other way round — ``ondevice/eval/make_eval_prompts.py`` sets ``DIRECTION = "dia2std"``
("방언 → 표준어", the offline voice-command normaliser). Every checkpoint this project has
selected was therefore scored on the direction it does not deploy.

The headline column is ``gain_over_copy``, not chrF. Emitting the input unchanged already
scores 56.7 chrF on Gyeongsang, 76.8 on Gangwon, 79.9 on Chungcheong and 18.9 on Jeju
(held-out, measured 2026-08-05), so raw chrF cannot be compared across regions: a Gangwon
76.9 is +0.07 over doing nothing while a Gyeongsang 83.9 is +27.

Usage:
    python scripts/eval_dia2std.py --model_path outputs/sft_5region_merged
    python scripts/eval_dia2std.py --model_path outputs/sft_merged --n 200 --short_only
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fire  # noqa: E402
import torch  # noqa: E402

from ko_dialect.data.labels import SUPPORTED_DO  # noqa: E402
from ko_dialect.data.template import ChatTemplate  # noqa: E402
from ko_dialect.evaluation.metrics import (  # noqa: E402
    compute_chrf,
    compute_copy_baseline,
    compute_gain_over_copy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_dia2std")

SHORT_EOJEOL_MAX = 6  # the bucket the on-device demo collapsed on


def _score(outputs: list[str], sources: list[str], golds: list[str]) -> dict[str, float]:
    """chrF of the model, of the do-nothing baseline, and the gap between them.

    Delegates to ``evaluation.metrics`` so this script and the leaderboard cannot end up
    with two different definitions of the same number.
    """
    copied = sum(1 for o, s in zip(outputs, sources) if o.strip() == s.strip())
    return {
        "chrf": round(compute_chrf(outputs, golds), 2),
        "copy_baseline": round(compute_copy_baseline(sources, golds), 2),
        "gain_over_copy": round(compute_gain_over_copy(outputs, sources, golds), 2),
        "copy_rate": round(copied / len(outputs), 4),
        "n": len(outputs),
    }


def main(
    model_path: str,
    raw_dataset_path: str = "outputs/dialect_raw_new_spk",
    split: str = "valid",
    regions: str | None = None,
    n: int = 150,
    max_new_tokens: int = 96,
    batch_size: int = 16,
    short_only: bool = False,
    json_out: str | None = None,
) -> None:
    """Generate dia2std for each region and report gain over the copy baseline.

    Args:
        model_path: A merged model directory (not a bare LoRA adapter).
        raw_dataset_path: Defaults to the **speaker-disjoint** view; the shipped AI-Hub
            split shares 12.3% of its valid speakers with train, which inflates held-out
            numbers (docs/CORPUS_ANALYSIS_5REGION.md §4).
        regions: Comma-separated subset; default is every supported region present.
        n: Rows per region.
        short_only: Restrict to <= 6 eojeol inputs — the bucket real usage failed on.
    """
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ds_all = load_from_disk(raw_dataset_path)[split]
    wanted = (
        [r.strip() for r in regions.split(",")]
        if regions
        else sorted(set(ds_all["do"]) & SUPPORTED_DO)
    )

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    template = ChatTemplate()

    results: dict[str, dict[str, float]] = {}
    for region in wanted:
        rows = ds_all.filter(lambda x, r=region: x["do"] == r and not x["is_identical"])
        if short_only:
            rows = rows.filter(lambda x: len(x["dialect"].split()) <= SHORT_EOJEOL_MAX)
        if not len(rows):
            logger.warning("no rows for %s; skipping", region)
            continue
        rows = rows.select(range(min(n, len(rows))))

        sources = [r["dialect"] for r in rows]
        golds = [r["standard"] for r in rows]
        outputs: list[str] = []
        for start in range(0, len(sources), batch_size):
            chunk = sources[start : start + batch_size]
            prompts = [template.build_prompt(tok, s, region, "dia2std") for s in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            outputs += [
                t.strip()
                for t in tok.batch_decode(
                    out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
                )
            ]

        results[region] = _score(outputs, sources, golds)
        logger.info("%s: %s", region, results[region])

    bucket = f"short<={SHORT_EOJEOL_MAX}" if short_only else "all"
    print(f"\n=== dia2std (deployed direction) — {model_path} [{split}, {bucket}] ===")
    print(
        f"{'region':16s} {'chrF':>8s} {'copy_baseline':>14s} {'gain_over_copy↑':>16s} {'copy_rate↓':>11s} {'n':>5s}"
    )
    for region, m in results.items():
        print(
            f"{region:16s} {m['chrf']:>8.2f} {m['copy_baseline']:>14.2f} "
            f"{m['gain_over_copy']:>16.2f} {m['copy_rate']:>10.1%} {m['n']:>5d}"
        )
    if results:
        mean_gain = sum(m["gain_over_copy"] for m in results.values()) / len(results)
        print(f"\n{'MACRO mean gain_over_copy':16s} {mean_gain:>8.2f}")
        print(
            "\nReading: gain_over_copy is chrF(gen,gold) − chrF(source,gold). Negative means "
            "the model is worse than emitting its input unchanged. Raw chrF is NOT comparable "
            "across regions — each has a different copy floor."
        )

    if json_out:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model_path": model_path,
                    "raw_dataset_path": raw_dataset_path,
                    "split": split,
                    "bucket": bucket,
                    "per_region": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("wrote %s", path)


if __name__ == "__main__":
    fire.Fire(main)
