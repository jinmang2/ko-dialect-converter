from __future__ import annotations

import pytest

from ko_dialect.evaluation.significance import (
    BootstrapCI,
    bootstrap_ci,
    paired_bootstrap,
    significance_marker,
)


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [10.0, 12.0, 11.0, 9.0, 13.0, 8.0, 14.0, 10.5]
    a = bootstrap_ci(vals, n_boot=500, seed=7)
    b = bootstrap_ci(vals, n_boot=500, seed=7)
    assert a == b  # deterministic under seed
    assert a.lo <= a.mean <= a.hi
    assert a.mean == pytest.approx(sum(vals) / len(vals))


def test_bootstrap_ci_validates():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], confidence=1.5)


def test_bootstrap_ci_as_dict_keys():
    d = bootstrap_ci([1.0, 2.0, 3.0], n_boot=100, seed=1).as_dict()
    assert set(d) == {"mean", "ci_lo", "ci_hi", "confidence"}


def test_paired_bootstrap_detects_clear_winner():
    # system strictly dominates baseline on every sentence -> win_rate ~1, significant
    baseline = [10.0] * 30
    system = [20.0] * 30
    res = paired_bootstrap(system, baseline, n_boot=500, seed=3)
    assert res.delta == pytest.approx(10.0)
    assert res.win_rate == 1.0
    assert res.p_value == 0.0
    assert res.significant_05 is True


def test_paired_bootstrap_no_difference_is_not_significant():
    scores = [5.0, 7.0, 6.0, 8.0, 4.0, 9.0, 5.5, 6.5]
    res = paired_bootstrap(scores, list(scores), n_boot=500, seed=3)
    assert res.delta == pytest.approx(0.0)
    assert res.significant_05 is False


def test_paired_bootstrap_alignment_required():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0], n_boot=10)
    with pytest.raises(ValueError):
        paired_bootstrap([], [], n_boot=10)


def test_paired_bootstrap_is_deterministic():
    a = paired_bootstrap([1.0, 5.0, 2.0, 9.0], [1.0, 2.0, 2.0, 3.0], n_boot=200, seed=11)
    b = paired_bootstrap([1.0, 5.0, 2.0, 9.0], [1.0, 2.0, 2.0, 3.0], n_boot=200, seed=11)
    assert a == b


def test_significance_marker_symbols():
    up = paired_bootstrap([20.0] * 20, [10.0] * 20, n_boot=200, seed=1)
    down = paired_bootstrap([10.0] * 20, [20.0] * 20, n_boot=200, seed=1)
    ns = paired_bootstrap([5.0] * 20, [5.0] * 20, n_boot=200, seed=1)
    assert significance_marker(up) == "▲"
    assert significance_marker(down) == "▼"
    assert significance_marker(ns) == "≈"


def test_bootstrap_ci_dataclass_is_frozen():
    ci = BootstrapCI(mean=1.0, lo=0.5, hi=1.5, confidence=0.95)
    with pytest.raises(Exception):
        ci.mean = 2.0  # type: ignore[misc]
