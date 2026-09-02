#!/usr/bin/env python3
"""How concentrated is each region's dialect vocabulary — i.e. would a lexicon work?

Motivation: measured dia2std, Jeju scores *below* the copy-the-input baseline while
Gyeongsang beats it by +27 chrF (docs/CORPUS_ANALYSIS_5REGION.md §6). One cheap
hypothesis is that Jeju is a closed set of unusual words that a lookup table could fix.
This measures that directly: if the top-N most frequent dialect→standard word mappings
cover most of the divergence, a lexicon is viable; if the divergence has a long tail,
only training on more of the region will help.

The comparison must use an **equal number of changed eojeols per region**, not equal
rows. Jeju is 5.6% of the corpus, so a per-row sample gives it a third of Gyeongsang's
tokens — and type/token ratio rises as a sample shrinks, which would manufacture exactly
the "Jeju is more diverse" conclusion this script is testing.

Usage:
    python scripts/audit_lexical_gap.py
    python scripts/audit_lexical_gap.py --budget 12000 --show 15
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fire  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("audit_lexical_gap")


def main(
    raw_dataset_path: str = "outputs/dialect_raw_new",
    split: str = "train",
    budget: int = 6000,
    show: int = 12,
    json_out: str | None = None,
) -> None:
    """Tally dialect→standard word mappings per region under an equal token budget.

    Args:
        budget: Changed eojeols collected per region. Regions that cannot reach it are
            reported with what they had, since a short region would otherwise look
            artificially diverse.
    """
    from datasets import load_from_disk

    ds = load_from_disk(raw_dataset_path)[split]
    if "dialect_eojeol_map" not in ds.column_names:
        raise SystemExit(f"{raw_dataset_path} has no 'dialect_eojeol_map' column.")

    pairs: dict[str, Counter] = defaultdict(Counter)
    seen: Counter = Counter()
    # Stride the whole split so the sample is not the head of one region's files.
    step = max(1, len(ds) // (budget * 10))
    for row in ds.select(range(0, len(ds), step)):
        region = row["do"]
        if seen[region] >= budget:
            continue
        for eojeol in row.get("dialect_eojeol_map") or []:
            dialect = (eojeol.get("dialect") or "").strip()
            standard = (eojeol.get("standard") or "").strip()
            if not dialect or not standard or dialect == standard:
                continue
            if seen[region] >= budget:
                break
            seen[region] += 1
            pairs[region][(dialect, standard)] += 1

    print(
        f"\n=== lexical concentration — {raw_dataset_path}[{split}], budget {budget:,}/region ==="
    )
    print(
        f"{'region':15s} {'tokens':>8s} {'types':>8s} {'TTR':>6s} "
        f"{'top100':>8s} {'top500':>8s} {'top1000':>8s}"
    )
    summary: dict[str, dict[str, float]] = {}
    for region in sorted(pairs):
        counter = pairs[region]
        tokens = sum(counter.values())
        cover = lambda k: sum(v for _, v in counter.most_common(k)) / tokens  # noqa: E731
        summary[region] = {
            "tokens": tokens,
            "types": len(counter),
            "ttr": round(len(counter) / tokens, 4),
            "top100": round(cover(100), 4),
            "top500": round(cover(500), 4),
            "top1000": round(cover(1000), 4),
            "reached_budget": tokens >= budget,
        }
        s = summary[region]
        flag = "" if s["reached_budget"] else "  (budget not reached)"
        print(
            f"{region:15s} {tokens:>8,} {len(counter):>8,} {s['ttr']:>6.2f} "
            f"{s['top100']:>7.1%} {s['top500']:>7.1%} {s['top1000']:>7.1%}{flag}"
        )

    print(
        "\nReading: low top100 coverage + high TTR = the divergence is spread thin, so a "
        "lookup table buys little and the region needs training exposure instead."
    )
    for region in sorted(pairs):
        top = ", ".join(f"{d}→{s}" for (d, s), _ in pairs[region].most_common(show))
        print(f"\n[{region}] {top}")

    if json_out:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"raw_dataset_path": raw_dataset_path, "budget": budget, "per_region": summary},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("wrote %s", path)


if __name__ == "__main__":
    fire.Fire(main)
