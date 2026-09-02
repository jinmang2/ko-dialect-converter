"""Dataset EDA + eval-set representativeness audit.

Answers two questions the repo had no tooling for:

1. What is actually in the corpus — which regions, how much of it carries any transfer
   signal at all, and what the annotation columns hold (several are populated by
   ``prepare_data.py`` and then never read by anything).
2. Whether the eval slice a score was computed over represents the population. Every eval
   path selects with ``select(range(n))``; this measures the distortion that introduces,
   per region, and what ``strategy="stratified"`` recovers.

    uv run python scripts/analyze_data.py columns
    uv run python scripts/analyze_data.py regions
    uv run python scripts/analyze_data.py edits --do gangwondo
    uv run python scripts/analyze_data.py evalset --n 150
    uv run python scripts/analyze_data.py all
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import fire

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from ko_dialect.evaluation.sampling import (  # noqa: E402
    bucket_shares,
    changed_eojeol_count,
    difficulty_bucket,
    stratified_indices,
)

RAW_DEFAULT = "outputs/dialect_raw_new"
GRPO_DEFAULT = "outputs/datasets/grpo"
BUCKETS = ("1", "2-3", "4-6", "7+")


def _pct(n: int, d: int) -> str:
    return f"{n / d * 100:5.1f}%" if d else "  n/a"


def _quantile(values: list[int], p: float) -> int:
    return sorted(values)[min(int(len(values) * p), len(values) - 1)]


def _load(path: str, split: str):
    from datasets import load_from_disk

    return load_from_disk(path)[split]


def columns(dataset: str = RAW_DEFAULT, split: str = "valid") -> None:
    """Per-column fill rate, plus the distribution of the annotation columns.

    ``speech_kind`` / ``intent`` / ``emotion`` are fully populated and read by nothing —
    this is what is on the table if a future experiment wants to use them.
    """
    ds = _load(dataset, split)
    print(f"\n[{dataset}:{split}]  n = {len(ds):,}\n")
    print("fill rate")
    for col in ds.column_names:
        vals = ds[col]
        empty = sum(1 for x in vals if x is None or x == "" or x == [] or x == {})
        print(f"  {col:<22} null/empty = {empty:>8,} ({_pct(empty, len(vals))})")

    for col in ("speech_kind", "intent", "emotion"):
        if col not in ds.column_names:
            continue
        counts = Counter(str(x) for x in ds[col])
        print(f"\n{col}  (distinct = {len(counts)})")
        for key, n in counts.most_common(8):
            label = key if len(key) <= 40 else key[:39] + "…"
            print(f"  {label:<42} {n:>8,} ({_pct(n, len(ds))})")


def regions(dataset: str = RAW_DEFAULT) -> None:
    """Region counts and how much of each split survives the ``is_identical`` filter.

    A row whose dialect text equals its standard text teaches nothing about the transfer,
    so every training and eval path drops it. That filter is not a rounding error here.
    """
    from datasets import load_from_disk

    ds = load_from_disk(dataset)
    for split in ds:
        s = ds[split]
        print(f"\n[{split}]  n = {len(s):,}")
        counts = Counter(s["do"])
        ident = Counter(do for do, i in zip(s["do"], s["is_identical"]) if i)
        for do, n in counts.most_common():
            usable = n - ident[do]
            print(
                f"  {do:<14} {n:>8,}   identical = {ident[do]:>7,} ({_pct(ident[do], n)})"
                f"   usable = {usable:>7,}"
            )
        total_ident = sum(ident.values())
        print(
            f"  {'TOTAL usable':<14} {len(s) - total_ident:>8,} ({_pct(len(s) - total_ident, len(s))})"
        )

        if "speech_kind" in s.column_names:
            print("  by speech_kind:")
            tot = Counter(s["speech_kind"])
            ki = Counter(k for k, i in zip(s["speech_kind"], s["is_identical"]) if i)
            for k, n in tot.most_common():
                print(f"    {k:<8} n = {n:>7,}   identical = {ki[k]:>7,} ({_pct(ki[k], n)})")


def edits(dataset: str = RAW_DEFAULT, split: str = "valid", do: str = "gangwondo") -> None:
    """How much actually changes in a usable pair — the copy-bias picture.

    If most pairs differ by one eojeol out of a dozen, n-gram metrics are largely scoring
    the copied remainder (DIA-REFINE arXiv:2511.06680), which is what copy_margin exists
    to separate out.
    """
    ds = _load(dataset, split)
    sub = ds.filter(lambda x: x["do"] == do and not x["is_identical"])
    if len(sub) == 0:
        print(f"No usable rows for {do} in {dataset}:{split}.")
        return

    changed = [changed_eojeol_count(d, s) for d, s in zip(sub["dialect"], sub["standard"])]
    lengths = [len(s.split()) for s in sub["standard"]]
    ratios = [c / max(n, 1) for c, n in zip(changed, lengths)]
    dia_chars = [len(s) for s in sub["dialect"]]
    std_chars = [len(s) for s in sub["standard"]]

    print(f"\n[{do}]  usable n = {len(sub):,}   standard eojeol p50 = {_quantile(lengths, 0.5)}")
    print(
        f"  changed eojeol   mean = {mean(changed):5.2f}"
        f"  p50 = {_quantile(changed, 0.5)}  p90 = {_quantile(changed, 0.9)}"
        f"  p99 = {_quantile(changed, 0.99)}"
    )
    print(
        f"  changed fraction mean = {mean(ratios) * 100:5.1f}%  p50 = {_quantile([int(r * 100) for r in ratios], 0.5)}%"
    )
    print(
        f"  chars: dialect mean = {mean(dia_chars):6.1f}   standard mean = {mean(std_chars):6.1f}"
        f"   Δ = {mean(dia_chars) - mean(std_chars):+.2f}"
    )
    shares = bucket_shares(
        [difficulty_bucket(d, s) for d, s in zip(sub["dialect"], sub["standard"])]
    )
    print("  difficulty mix: " + "  ".join(f"{b}={shares[b] * 100:.1f}%" for b in BUCKETS))


def evalset(
    dataset: str = GRPO_DEFAULT,
    split: str = "valid",
    dos: str = "gangwondo,gyeongsangdo",
    n: int = 150,
    direction: str = "std2dia",
) -> None:
    """Compare the population's difficulty mix against head-n and stratified-n.

    Total variation distance summarises the gap in one number: 0% means the eval slice has
    the population's mix. A large value means the reported score is measured on a slice
    that is easier (or harder) than the data — and because the gap differs per region, it
    also means per-region scores are not on the same footing.
    """
    ds = _load(dataset, split)
    has_direction = "direction" in ds.column_names

    print(f"\n[{dataset}:{split}]  n = {n} per region\n")
    header = f"{'region':<14}{'bucket':<8}{'population':>12}{'head-n':>10}{'stratified':>12}"
    print(header)
    print("-" * len(header))
    for do in (d.strip() for d in dos.split(",") if d.strip()):
        pop = ds.filter(
            lambda x, d=do: (
                x["do"] == d
                and x["standard"] != x["dialect"]
                and (not has_direction or x["direction"] == direction)
            )
        )
        if len(pop) == 0:
            print(f"{do:<14}(no rows)")
            continue

        buckets = [difficulty_bucket(d, s) for d, s in zip(pop["dialect"], pop["standard"])]
        p = bucket_shares(buckets)
        h = bucket_shares(buckets[:n])
        s = bucket_shares([buckets[i] for i in stratified_indices(buckets, n)])
        for b in BUCKETS:
            print(f"{do:<14}{b:<8}{p[b] * 100:11.1f}%{h[b] * 100:9.1f}%{s[b] * 100:11.1f}%")
        tv_h = sum(abs(p[b] - h[b]) for b in BUCKETS) / 2
        tv_s = sum(abs(p[b] - s[b]) for b in BUCKETS) / 2
        print(
            f"{do:<14}{'TVD':<8}{'—':>12}{tv_h * 100:9.1f}%{tv_s * 100:11.1f}%   (lower = more representative)"
        )
        print("-" * len(header))


def all(raw: str = RAW_DEFAULT, grpo: str = GRPO_DEFAULT, n: int = 150) -> None:  # noqa: A001
    """Run every section."""
    regions(raw)
    columns(raw)
    for do in ("gangwondo", "gyeongsangdo"):
        edits(raw, "valid", do)
    evalset(grpo, "valid", n=n)


if __name__ == "__main__":
    fire.Fire(
        {
            "columns": columns,
            "regions": regions,
            "edits": edits,
            "evalset": evalset,
            "all": all,
        }
    )
