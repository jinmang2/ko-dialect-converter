#!/usr/bin/env python3
"""Measure how much *learnable signal* a raw dialect corpus actually carries.

Two questions this answers, both of which decided real project choices:

1. **Is a corpus worth training on?** ``dialect_raw_old`` (the 2020 v1 five-region text)
   was shelved on a "2–4% signal" reading. That number needs a reproducible definition
   and a side-by-side against ``dialect_raw_new`` before the call is re-affirmed or
   reversed — this script is that definition.
2. **Why does the model fail on short sentences?** Measured on the live corpus: after
   ``filter_identical`` only ~4% of trainable rows are ≤5 어절, so the SFT model
   essentially never sees short input. That is a *sampling* defect, not a data shortage.

Signal is reported at three levels, because they disagree in an informative way:
    pair-level   — share of pairs where dialect != standard  (what filter_identical keeps)
    eojeol-level — share of words that change, over ALL pairs (corpus-wide edit density)
    edited-only  — share of words that change, among non-identical pairs only

Usage:
    python scripts/audit_raw_signal.py --raw_dataset_path outputs/dialect_raw_new
    python scripts/audit_raw_signal.py --raw_dataset_path outputs/dialect_raw_old --n 30000
    python scripts/audit_raw_signal.py --raw_dataset_path outputs/dialect_raw_new \
        --json_out outputs/eval_logs/signal_raw_new.json
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fire  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("audit_raw_signal")

SHORT_EOJEOL_MAX = 5  # "short utterance" threshold — the regime the demo failed on
PERCENTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * q), len(sorted_values) - 1)
    return sorted_values[idx]


def changed_eojeol_ratio(standard: str, dialect: str) -> tuple[int, int]:
    """(changed words, total words) between the two sides, via longest-common-subsequence.

    Format-independent: it does not rely on ``dialect_eojeol_map``, whose key names differ
    between the old (``idx``) and new (``standard_idx``) schemas.
    """
    s_words, d_words = standard.split(), dialect.split()
    total = max(len(s_words), len(d_words))
    if total == 0:
        return 0, 0
    matched = sum(
        block.size for block in SequenceMatcher(None, s_words, d_words).get_matching_blocks()
    )
    return total - matched, total


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"rows": 0}

    lengths = sorted(float(r["n_eojeol"]) for r in rows)
    non_identical = [r for r in rows if not r["is_identical"]]
    changed_all = sum(r["changed"] for r in rows)
    total_all = sum(r["total"] for r in rows) or 1
    changed_edit = sum(r["changed"] for r in non_identical)
    total_edit = sum(r["total"] for r in non_identical) or 1

    short_all = sum(1 for r in rows if r["n_eojeol"] <= SHORT_EOJEOL_MAX)
    short_edit = sum(1 for r in non_identical if r["n_eojeol"] <= SHORT_EOJEOL_MAX)

    by_kind: dict[str, int] = defaultdict(int)
    for r in rows:
        by_kind[r.get("speech_kind") or "unknown"] += 1

    return {
        "rows": n,
        "trainable_rows": len(non_identical),
        "pair_signal": len(non_identical) / n,
        "eojeol_signal_all": changed_all / total_all,
        "eojeol_signal_edited_only": changed_edit / total_edit,
        "eojeol_len": {f"p{int(q * 100)}": _percentile(lengths, q) for q in PERCENTILES},
        "short_share_all": short_all / n,
        "short_share_trainable": (short_edit / len(non_identical)) if non_identical else 0.0,
        "prosody_coverage": sum(1 for r in rows if r["has_prosody"]) / n,
        "speech_kind": dict(sorted(by_kind.items())),
    }


def _print_block(title: str, s: dict[str, Any]) -> None:
    if not s.get("rows"):
        print(f"\n### {title}: (empty)")
        return
    lens = s["eojeol_len"]
    print(f"\n### {title}")
    print(f"  rows                     : {s['rows']:,}  (trainable {s['trainable_rows']:,})")
    print(f"  pair signal (dia!=std)   : {s['pair_signal']:.1%}")
    print(f"  eojeol signal (all pairs): {s['eojeol_signal_all']:.2%}   <- the '% signal' number")
    print(f"  eojeol signal (edited)   : {s['eojeol_signal_edited_only']:.2%}")
    print("  eojeol length            : " + " ".join(f"{k}={v:.0f}" for k, v in lens.items()))
    print(
        f"  <= {SHORT_EOJEOL_MAX} eojeol             : "
        f"{s['short_share_all']:.1%} of all, {s['short_share_trainable']:.1%} of trainable"
    )
    print(f"  prosody (F0) coverage    : {s['prosody_coverage']:.1%}")
    print(f"  speech_kind              : {s['speech_kind']}")


def main(
    raw_dataset_path: str = "outputs/dialect_raw_new",
    split: str = "train",
    n: int = 30_000,
    json_out: str | None = None,
) -> None:
    """Audit one raw dataset.

    Args:
        raw_dataset_path: A ``save_to_disk`` DatasetDict (dialect_raw_new / _old / _prosody_v2).
        split: Which split to audit ("train" or "valid").
        n: Rows sampled **per region** (deterministic). Keeps peak RAM small on a 6 GB box;
            set 0 to use everything.
        json_out: Optional path to also dump the summary as JSON.
    """
    from datasets import load_from_disk

    ds_dict = load_from_disk(raw_dataset_path)
    if split not in ds_dict:
        raise SystemExit(f"split {split!r} not in {sorted(ds_dict)}")
    ds = ds_dict[split]
    logger.info("%s[%s]: %d rows, columns=%s", raw_dataset_path, split, len(ds), ds.column_names)

    dos = ds["do"]
    regions = sorted(set(dos))
    by_region: dict[str, list[int]] = defaultdict(list)
    for i, do in enumerate(dos):
        by_region[do].append(i)

    per_region: dict[str, dict[str, Any]] = {}
    overall: list[dict[str, Any]] = []

    for region in regions:
        idx = by_region[region]
        if n and len(idx) > n:
            # Deterministic stride sample — no RNG state, reproducible across machines.
            stride = len(idx) / n
            idx = [idx[int(k * stride)] for k in range(n)]
        subset = ds.select(idx)
        std_col, dia_col = subset["standard"], subset["dialect"]
        kinds = subset["speech_kind"] if "speech_kind" in subset.column_names else [None] * len(idx)
        prosody = subset["prosody"] if "prosody" in subset.column_names else [None] * len(idx)

        rows: list[dict[str, Any]] = []
        for std, dia, kind, pros in zip(std_col, dia_col, kinds, prosody, strict=True):
            changed, total = changed_eojeol_ratio(std, dia)
            rows.append(
                {
                    "n_eojeol": len(dia.split()),
                    "is_identical": std == dia,
                    "changed": changed,
                    "total": total,
                    "speech_kind": kind,
                    "has_prosody": bool(pros),
                }
            )
        per_region[region] = _summarise(rows)
        overall.extend(rows)
        logger.info("audited %s (%d sampled of %d)", region, len(rows), len(by_region[region]))

    print("\n" + "=" * 78)
    print(f"SIGNAL AUDIT  {raw_dataset_path} [{split}]   (sample<= {n or 'all'} per region)")
    print("=" * 78)
    for region in regions:
        _print_block(region, per_region[region])
    _print_block("OVERALL", _summarise(overall))

    print(
        "\nReading: 'eojeol signal (all pairs)' is the corpus-wide share of words that "
        f"differ.\nA corpus below ~5% is mostly copy-through; '<= {SHORT_EOJEOL_MAX} eojeol "
        "of trainable' below ~10%\nmeans short-input failures are expected at inference."
    )

    if json_out:
        payload = {
            "raw_dataset_path": raw_dataset_path,
            "split": split,
            "sample_per_region": n,
            "per_region": per_region,
            "overall": _summarise(overall),
        }
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", out_path)


if __name__ == "__main__":
    fire.Fire(main)
