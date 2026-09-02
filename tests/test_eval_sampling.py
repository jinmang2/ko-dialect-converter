"""Tests for deterministic eval-set selection (``ko_dialect.evaluation.sampling``).

The point of this module is that a representative eval set does not have to cost
reproducibility. These tests pin both halves of that claim: the bucket mix tracks the
population, and the selection is a pure function of its inputs.
"""

from __future__ import annotations

from ko_dialect.evaluation.sampling import (
    bucket_shares,
    changed_eojeol_count,
    edit_bucket,
    stratified_indices,
)


def test_changed_eojeol_count_counts_positional_and_length_differences():
    assert changed_eojeol_count("가 나 다", "가 나 다") == 0
    assert changed_eojeol_count("가 나 라", "가 나 다") == 1
    assert changed_eojeol_count("가 나", "가 나 다") == 1  # missing eojeol counts as changed
    assert changed_eojeol_count("바 사 아", "가 나 다") == 3


def test_edit_bucket_puts_the_copy_friendly_case_in_its_own_stratum():
    assert edit_bucket("가 나 라", "가 나 다") == "1"
    assert edit_bucket("바 사 다", "가 나 다") == "2-3"
    assert edit_bucket(" ".join("바사아자마가"), " ".join("가나다라마바")) == "4-6"
    assert edit_bucket(" ".join("abcdefgh"), " ".join("12345678")) == "7+"


def _pairs(spec: list[tuple[str, int]]) -> list[str]:
    """Expand ``[(bucket, count), ...]`` into a bucket list."""
    out: list[str] = []
    for bucket, count in spec:
        out.extend([bucket] * count)
    return out


def test_stratified_selection_matches_population_mix():
    # 50% easy / 30% mid / 20% hard, and a request for a tenth of it.
    buckets = _pairs([("1", 500), ("2-3", 300), ("4-6", 200)])
    picked = stratified_indices(buckets, 100)

    assert len(picked) == 100
    got = bucket_shares([buckets[i] for i in picked])
    assert got["1"] == 0.50
    assert got["2-3"] == 0.30
    assert got["4-6"] == 0.20


def test_stratified_selection_beats_a_head_slice_on_a_sorted_population():
    """The real failure mode: rows arrive grouped, so first-n reads one stratum only."""
    buckets = _pairs([("1", 300), ("2-3", 400), ("4-6", 300)])

    head = bucket_shares(buckets[:150])
    assert head["1"] == 1.0  # first-n sees nothing but the easiest bucket

    strat = bucket_shares([buckets[i] for i in stratified_indices(buckets, 150)])
    assert strat["1"] == 0.30
    assert strat["2-3"] == 0.40
    assert strat["4-6"] == 0.30


def test_allocation_sums_to_n_even_when_shares_do_not_divide_evenly():
    buckets = _pairs([("1", 7), ("2-3", 11), ("4-6", 13), ("7+", 5)])
    for n in range(1, len(buckets) + 1):
        assert len(stratified_indices(buckets, n)) == n


def test_selection_is_deterministic_and_ordered():
    buckets = _pairs([("1", 40), ("2-3", 35), ("4-6", 25)])
    first = stratified_indices(buckets, 30)

    assert first == stratified_indices(buckets, 30)  # no RNG, no seed
    assert first == sorted(first)  # original row order preserved for stable hashing
    assert len(set(first)) == len(first)


def test_small_bucket_does_not_lose_rows_to_rounding():
    """A stratum too small for its share must not silently shrink the eval set."""
    buckets = _pairs([("1", 99), ("7+", 1)])
    picked = stratified_indices(buckets, 50)

    assert len(picked) == 50
    assert all(0 <= i < len(buckets) for i in picked)


def test_degenerate_inputs():
    assert stratified_indices([], 10) == []
    assert stratified_indices(["1", "2-3"], 0) == []
    assert stratified_indices(["1", "2-3"], 5) == [0, 1]  # n over population returns all
    assert bucket_shares([]) == {}
