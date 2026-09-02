#!/usr/bin/env python3
"""Classify what the ASR actually gets wrong, per region — the GER task's real content.

``build_ger_dataset`` says the task exists (86–96% of rows need an edit). This says what
kind of edit, which decides whether a 0.5B LM is the right tool: rewriting digits as
Korean numerals and fixing particles is squarely in an LM's competence, while recovering
a misheard content word is closer to ASR rescoring and may not be learnable from text.

Usage:
    python scripts/audit_ger_errors.py --n 4000
    python scripts/audit_ger_errors.py --raw_dataset_path outputs/dialect_raw_new --show 6
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fire  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("audit_ger_errors")

_DIGIT = re.compile(r"\d")
# Particles and endings that carry no lexical content — a swap here is a grammar fix,
# not a misheard word.
_PARTICLE = re.compile(r"^(은|는|이|가|을|를|에|에서|으로|로|와|과|도|만|의|께|처럼|보다)$")


def _classify(src_tokens: list[str], tgt_tokens: list[str]) -> str:
    """Label one aligned edit. Order matters: the first matching rule wins."""
    src, tgt = " ".join(src_tokens), " ".join(tgt_tokens)
    if not src_tokens:
        return "insertion"
    if not tgt_tokens:
        return "deletion"
    if _DIGIT.search(src) and not _DIGIT.search(tgt):
        return "number_to_hangul"
    if src.replace(" ", "") == tgt.replace(" ", ""):
        return "spacing"
    if len(src_tokens) == 1 and len(tgt_tokens) == 1:
        a, b = src_tokens[0], tgt_tokens[0]
        if _PARTICLE.match(a) or _PARTICLE.match(b):
            return "particle"
        # Same stem, different tail → an ending/inflection fix rather than a new word.
        common = os.path.commonprefix([a, b])
        if len(common) >= max(1, min(len(a), len(b)) // 2):
            return "ending_or_inflection"
        return "lexical_substitution"
    return "phrase_rewrite"


def main(
    raw_dataset_path: str = "outputs/dialect_raw_new",
    split: str = "train",
    n: int = 4000,
    show: int = 4,
    json_out: str | None = None,
) -> None:
    """Sample rows, align ``stt_hypothesis`` against ``standard``, and tally edit types.

    Args:
        n: Rows sampled **per region** (evenly strided, so it is not the head of the file).
        show: Example edits to print per category.
    """
    from difflib import SequenceMatcher

    from datasets import load_from_disk

    ds = load_from_disk(raw_dataset_path)[split]
    if "stt_hypothesis" not in ds.column_names:
        raise SystemExit(
            f"{raw_dataset_path} has no 'stt_hypothesis' column — GER needs the v2 corpora."
        )

    by_region: dict[str, Counter] = defaultdict(Counter)
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "edits": 0, "clean": 0})
    examples: dict[str, list[tuple[str, str]]] = defaultdict(list)

    per_region_seen: Counter = Counter()
    step = max(1, len(ds) // (n * 6))
    for row in ds.select(range(0, len(ds), step)):
        region = row["do"]
        if per_region_seen[region] >= n:
            continue
        stt = (row.get("stt_hypothesis") or "").strip()
        std = (row["standard"] or "").strip()
        if not stt or not std:
            continue
        per_region_seen[region] += 1
        totals[region]["rows"] += 1
        if stt == std:
            totals[region]["clean"] += 1
            continue

        a, b = stt.split(), std.split()
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes():
            if tag == "equal":
                continue
            kind = _classify(a[i1:i2], b[j1:j2])
            by_region[region][kind] += 1
            totals[region]["edits"] += 1
            if len(examples[kind]) < show * 3:
                examples[kind].append((" ".join(a[i1:i2]), " ".join(b[j1:j2])))

    kinds = sorted({k for c in by_region.values() for k in c})
    print(f"\n=== GER edit types — {raw_dataset_path}[{split}] ===")
    header = f"{'region':15s} {'rows':>6s} {'clean':>7s} {'edits/row':>10s}  " + " ".join(
        f"{k[:14]:>15s}" for k in kinds
    )
    print(header)
    for region in sorted(by_region):
        c, t = by_region[region], totals[region]
        total = sum(c.values()) or 1
        row = (
            f"{region:15s} {t['rows']:>6d} {t['clean'] / max(1, t['rows']):>6.1%} "
            f"{t['edits'] / max(1, t['rows']):>10.2f}  "
        )
        row += " ".join(f"{c[k] / total:>14.1%}" for k in kinds)
        print(row)

    overall = Counter()
    for c in by_region.values():
        overall.update(c)
    grand = sum(overall.values()) or 1
    print(
        f"\n{'OVERALL':15s} {'':>6s} {'':>7s} {'':>10s}  "
        + " ".join(f"{overall[k] / grand:>14.1%}" for k in kinds)
    )

    print("\n=== examples (stt → standard) ===")
    for kind in kinds:
        print(f"\n[{kind}]  {overall[kind] / grand:.1%} of all edits")
        for src, tgt in examples[kind][:show]:
            print(f"    {src!r:40s} → {tgt!r}")

    if json_out:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "raw_dataset_path": raw_dataset_path,
                    "split": split,
                    "per_region": {r: dict(c) for r, c in by_region.items()},
                    "totals": {r: dict(t) for r, t in totals.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("wrote %s", path)


if __name__ == "__main__":
    fire.Fire(main)
