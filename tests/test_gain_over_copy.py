"""Tests for ``copy_baseline`` / ``gain_over_copy``.

Measured on held-out dia2std data, the copy floor — what emitting the input unchanged
already scores — ranges from 18.9 (Jeju) to 79.9 (Chungcheong). Raw chrF therefore is
not comparable across regions: Gangwon's 76.9 is a +0.07 gain over doing nothing while
Gyeongsang's 83.9 is +27. These tests pin that distinction, including the part that is
easy to conflate — ``gain_over_copy`` is not ``copy_margin``.
"""

from __future__ import annotations

from ko_dialect.evaluation.metric_registry import get_spec
from ko_dialect.evaluation.metrics import (
    compute_chrf,
    compute_copy_baseline,
    compute_gain_over_copy,
)


def _gain(outputs, sources, golds):
    """Exercise the shipped functions — not a re-implementation of their formula."""
    return (
        compute_gain_over_copy(outputs, sources, golds),
        compute_copy_baseline(sources, golds),
    )


def test_a_model_that_copies_its_input_gains_nothing():
    sources = ["밥 뭇나", "어데 가노"]
    golds = ["밥 먹었니", "어디 가니"]
    gain, baseline = _gain(sources, sources, golds)
    assert baseline > 0  # the floor is not zero — that is the whole point
    assert abs(gain) < 1e-6


def test_a_model_worse_than_copying_scores_negative():
    sources = ["밥 뭇나"]
    golds = ["밥 먹었니"]
    gain, _ = _gain(["전혀 다른 문장입니다"], sources, golds)
    assert gain < 0


def test_a_perfect_model_gains_the_full_headroom():
    sources = ["밥 뭇나"]
    golds = ["밥 먹었니"]
    gain, baseline = _gain(golds, sources, golds)
    assert gain == 100.0 - baseline


def test_high_floor_region_yields_small_gain_for_the_same_raw_score():
    # Same generation quality, two regions whose sources sit at different distances
    # from gold. Raw chrF would call them equal; gain_over_copy does not.
    golds = ["오늘 날씨가 참 좋습니다"]
    near_source = ["오늘 날씨가 참 좋습니더"]  # Gangwon-like: gold ≈ source
    far_source = ["오널 날씨가 하영 좋수다"]  # Jeju-like: gold far from source

    gen = golds  # a perfect generation in both cases
    gain_near, floor_near = _gain(gen, near_source, golds)
    gain_far, floor_far = _gain(gen, far_source, golds)

    assert floor_near > floor_far
    assert gain_far > gain_near


def test_gain_over_copy_is_not_copy_margin():
    # copy_margin compares the GENERATION to the source; gain_over_copy compares the
    # SOURCE to the gold. A generation that ignores the input entirely can score well
    # on copy_margin while being far below the copy floor.
    sources = ["밥 뭇나"]
    golds = ["밥 먹었니"]
    junk = ["완전히 무관한 출력"]

    copy_margin = compute_chrf(junk, golds) - compute_chrf(junk, sources)
    gain, _ = _gain(junk, sources, golds)
    assert gain < 0
    assert copy_margin > gain  # the two axes disagree, by construction


def test_registry_marks_the_two_new_metrics_correctly():
    gain = get_spec("gain_over_copy")
    assert gain.higher_is_better is True
    assert gain.group == "selection"

    floor = get_spec("copy_baseline")
    # A floor is not a score to maximise; it must not carry an "up is good" arrow.
    assert floor.higher_is_better is False
