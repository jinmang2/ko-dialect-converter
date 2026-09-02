"""Tests for region-balanced subsampling of the SFT budget.

Measured motivation (docs/CORPUS_ANALYSIS_5REGION.md): a proportional draw gives Jeju
8.4% of the budget and Gyeongsang 31%, while a dia2std eval puts Gyeongsang at +27 chrF
over the copy-the-input baseline and Jeju at −1. The budget is being spent where it is
least needed. These tests pin the two rules that make the fix meaningful.
"""

from __future__ import annotations

from collections import Counter

import pytest
from datasets import Dataset

from ko_dialect.data import subsample_balanced_by_region


def _ds(counts: dict[str, int]) -> Dataset:
    rows = [{"text": f"{r}-{i}", "do": r} for r, n in counts.items() for i in range(n)]
    return Dataset.from_list(rows)


SKEWED = {
    "gyeongsangdo": 500,
    "jeollado": 400,
    "chungcheongdo": 300,
    "gangwondo": 200,
    "jejudo": 90,
}


def test_equal_quota_per_region():
    out = subsample_balanced_by_region(_ds(SKEWED), total=250, seed=0)
    counts = Counter(out["do"])
    assert set(counts) == set(SKEWED)
    assert set(counts.values()) == {50}  # 250 // 5


def test_scarce_region_contributes_all_it_has_without_topping_up():
    # Jeju holds 90 < the 100 quota. The shortfall must NOT be refilled from a
    # majority region — that would silently restore the imbalance.
    out = subsample_balanced_by_region(_ds(SKEWED), total=500, seed=0)
    counts = Counter(out["do"])
    assert counts["jejudo"] == 90
    assert counts["gyeongsangdo"] == 100
    assert len(out) == 490  # smaller than `total`, on purpose


def test_it_actually_rebalances_the_skew():
    out = subsample_balanced_by_region(_ds(SKEWED), total=250, seed=0)
    share = Counter(out["do"])["jejudo"] / len(out)
    proportional = SKEWED["jejudo"] / sum(SKEWED.values())
    assert share > proportional * 2


def test_rows_are_shuffled_not_grouped_by_region():
    out = subsample_balanced_by_region(_ds(SKEWED), total=250, seed=0)
    regions = out["do"]
    # A region-ordered result would train in blocks; assert the head is mixed.
    assert len(set(regions[:25])) > 1


def test_seed_is_deterministic_and_different_seeds_differ():
    a = subsample_balanced_by_region(_ds(SKEWED), total=250, seed=1)["text"]
    b = subsample_balanced_by_region(_ds(SKEWED), total=250, seed=1)["text"]
    c = subsample_balanced_by_region(_ds(SKEWED), total=250, seed=2)["text"]
    assert a == b
    assert a != c


def test_falsy_total_returns_the_dataset_untouched_but_says_so(caplog):
    ds = _ds(SKEWED)
    with caplog.at_level("WARNING"):
        assert subsample_balanced_by_region(ds, total=None) is ds
    # A silent no-op here means training proportionally while the config says balanced.
    assert "no budget" in caplog.text
    assert subsample_balanced_by_region(ds, total=0) is ds


def test_missing_region_column_raises():
    ds = Dataset.from_list([{"text": "x"}])
    with pytest.raises(ValueError, match="balance_regions needs"):
        subsample_balanced_by_region(ds, total=10)
