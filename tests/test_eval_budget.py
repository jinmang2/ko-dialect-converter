"""Tests for the eval-budget warning.

The trap this guards is invisible in config: ``num_eval_samples: null`` means "score the
whole valid split", so the cost is set by the corpus rather than the run. Going from 2
regions to 5 grew valid 45,061 → 183,406 rows and one eval 60 → 172 min; at
``eval_steps=500`` a 3.2 h training run became 28.9 h, 89% of it evaluation, with no
error and no obvious symptom other than "it feels slow".
"""

from __future__ import annotations

from ko_dialect.training import eval_budget_warning


def test_unbounded_eval_on_the_five_region_split_is_flagged():
    # The configuration that actually happened.
    msg = eval_budget_warning(None, eval_rows=183_406, max_steps=4554, eval_steps=500)
    assert msg is not None
    assert "183,406" in msg
    assert "num_eval_samples" in msg


def test_setting_num_eval_samples_silences_it():
    assert eval_budget_warning(2000, eval_rows=183_406, max_steps=4554, eval_steps=500) is None


def test_a_small_valid_split_is_not_flagged():
    # Unbounded eval is fine when the split is small — don't cry wolf.
    assert eval_budget_warning(None, eval_rows=2_000, max_steps=4554, eval_steps=500) is None


def test_raising_eval_steps_can_bring_it_under_the_threshold():
    # 25k rows ≈ 0.8 h per eval: nine of them is worth a warning, one is not.
    often = eval_budget_warning(None, eval_rows=25_000, max_steps=4554, eval_steps=500)
    rarely = eval_budget_warning(None, eval_rows=25_000, max_steps=4554, eval_steps=4554)
    assert often is not None
    assert rarely is None


def test_no_eval_rows_or_disabled_eval_is_not_flagged():
    assert eval_budget_warning(None, eval_rows=0, max_steps=4554, eval_steps=500) is None
    assert eval_budget_warning(None, eval_rows=183_406, max_steps=4554, eval_steps=0) is None


def test_reported_hours_scale_with_the_split_size():
    small = eval_budget_warning(None, eval_rows=50_000, max_steps=4554, eval_steps=500)
    large = eval_budget_warning(None, eval_rows=200_000, max_steps=4554, eval_steps=500)
    # Both flagged, and the bigger split must report the bigger number.
    hours = lambda m: float(m.split("≈ ")[1].split(" h")[0])  # noqa: E731
    assert hours(large) > hours(small) * 3
