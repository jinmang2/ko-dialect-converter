"""Deterministic eval-set selection.

Every eval path in this repo selects its sample with ``select(range(n))`` — a plain
first-n slice. That was a deliberate choice: it is reproducible without a seed, which is
what lets the on-device harness fingerprint an eval set and prove a phone score and a
desktop score were computed over the same rows.

The cost of that choice had never been measured. It is large. On
``outputs/datasets/grpo`` valid, the first 150 gangwondo rows are 60.7% "only one eojeol
differs" pairs while the population is 23.8% — a 2.5x enrichment of the easiest cases.
For gyeongsangdo the same slice is 40.0% vs 28.5%, a 1.4x enrichment. Because the
distortion differs *per region*, per-region scores are skewed by different amounts, which
is exactly what a weighted OVERALL aggregate assumes away. n-gram metrics reward copying
the source (the copy_margin motivation, DIA-REFINE arXiv:2511.06680), and a first-n slice
happens to over-sample the rows where copying is nearly correct.

``stratified_indices`` keeps determinism — no RNG, no seed — while matching the
population's difficulty mix: allocate n across difficulty buckets in proportion to the
population, then take the first rows within each bucket. Same inputs always give the same
index list, so the reproducibility hash the validity gate depends on still holds.
"""

from __future__ import annotations

from collections.abc import Sequence

# Bucket edges in changed-eojeol counts. The 1-eojeol bucket is deliberately its own
# stratum: it is the copy-friendly case that inflates chrF/BLEU, and it is where the
# first-n slice over-samples the most.
_BUCKET_EDGES: tuple[tuple[str, int, int], ...] = (
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-6", 4, 6),
    ("7+", 7, 1 << 30),
)


def changed_eojeol_count(dialect: str, standard: str) -> int:
    """How many eojeol differ between a dialect/standard pair.

    Positional comparison plus the eojeol-count difference. This is a difficulty proxy for
    bucketing, not an alignment: a true edit distance would be more accurate but would also
    make the bucket assignment (and therefore the eval set) sensitive to the alignment
    implementation, which is the kind of hidden coupling this module exists to avoid.
    """
    a, b = dialect.split(), standard.split()
    common = min(len(a), len(b))
    return sum(1 for i in range(common) if a[i] != b[i]) + abs(len(a) - len(b))


def edit_bucket(dialect: str, standard: str) -> str:
    """Difficulty stratum for one pair — see ``_BUCKET_EDGES``."""
    changed = changed_eojeol_count(dialect, standard)
    for name, low, high in _BUCKET_EDGES:
        if low <= changed <= high:
            return name
    return "1"  # changed == 0 cannot occur (identical pairs are filtered upstream)


def bucket_shares(buckets: Sequence[str]) -> dict[str, float]:
    """Population share per bucket, for reporting the distortion a selection introduces."""
    if not buckets:
        return {}
    total = len(buckets)
    counts: dict[str, int] = {}
    for b in buckets:
        counts[b] = counts.get(b, 0) + 1
    return {name: counts.get(name, 0) / total for name, _, _ in _BUCKET_EDGES}


def stratified_indices(buckets: Sequence[str], n: int) -> list[int]:
    """Pick ``n`` indices whose bucket mix matches ``buckets``, deterministically.

    Allocation uses the largest-remainder method so the per-bucket counts sum to exactly
    ``n`` rather than drifting with rounding. Within a bucket the earliest rows win, and
    the returned indices are sorted ascending, so the selection is a pure function of
    ``(buckets, n)`` — no seed, stable across runs and machines.

    Returns every index when ``n`` >= the population, and ``[]`` for an empty population.
    """
    total = len(buckets)
    if total == 0:
        return []
    if n >= total:
        return list(range(total))
    if n <= 0:
        return []

    by_bucket: dict[str, list[int]] = {}
    for i, b in enumerate(buckets):
        by_bucket.setdefault(b, []).append(i)

    # Largest remainder: floor each exact share, then hand out what is left to the buckets
    # with the biggest fractional parts (ties broken by the fixed bucket order).
    order = [name for name, _, _ in _BUCKET_EDGES if name in by_bucket]
    exact = {name: len(by_bucket[name]) * n / total for name in order}
    alloc = {name: min(int(exact[name]), len(by_bucket[name])) for name in order}
    remaining = n - sum(alloc.values())
    if remaining > 0:
        ranked = sorted(
            order, key=lambda name: (-(exact[name] - int(exact[name])), order.index(name))
        )
        for name in ranked:
            if remaining == 0:
                break
            room = len(by_bucket[name]) - alloc[name]
            if room > 0:
                alloc[name] += 1
                remaining -= 1
        # A bucket smaller than its share can leave slack; top up wherever room is left.
        while remaining > 0:
            progressed = False
            for name in order:
                if remaining == 0:
                    break
                if len(by_bucket[name]) - alloc[name] > 0:
                    alloc[name] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break

    picked: list[int] = []
    for name in order:
        picked.extend(by_bucket[name][: alloc[name]])
    return sorted(picked)
