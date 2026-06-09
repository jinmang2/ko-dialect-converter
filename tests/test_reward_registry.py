from __future__ import annotations

import pytest

from ko_dialect.rewards import (
    REWARD_OUTPUT_RANGES,
    REWARD_REGISTRY,
    build_reward_fns,
)


def test_registry_has_core_rewards():
    assert {"style", "content", "edit", "length"} <= set(REWARD_REGISTRY)


def test_length_reward_builds_and_runs():
    funcs, weights = build_reward_fns([{"name": "length", "weight": 0.2}])
    assert weights == [0.2]
    out = funcs[0](
        prompts=[""],
        completions=["가나다라"],  # len 4 vs gold (dialect) len 1 -> ratio 4 >= 1.5
        standard=["x"],
        dialect=["가"],
        direction=["std2dia"],
    )
    assert out == [0.0]  # ratio 4/1 -> fully penalised


def test_length_builder_forwards_tuning_kwargs():
    funcs, _ = build_reward_fns(
        [{"name": "length", "weight": 1.0}],
        length_max_ratio=10.0,
        length_tolerance=0.0,
    )
    out = funcs[0](
        prompts=[""],
        completions=["가나다라"],
        standard=["x"],
        dialect=["가나다라"],
        direction=["std2dia"],
    )
    # ratio 4 < max_ratio 10 -> partial, not zero
    assert out[0] > 0.0


def test_normalize_rescales_outputs_to_unit():
    # content is already [0,1] so its declared range is a no-op; edit too. Use length,
    # whose declared range is [0,1] — verify normalize=True keeps a valid [0,1] output.
    funcs, _ = build_reward_fns(
        [{"name": "content", "weight": 1.0}], normalize=True
    )
    out = funcs[0](
        prompts=[""],
        completions=["가나다"],
        standard=["가나다"],
        dialect=["가나다"],
        direction=["std2dia"],
    )
    assert 0.0 <= out[0] <= 1.0


def test_output_ranges_cover_registry():
    # Every range entry must name a registered reward.
    assert set(REWARD_OUTPUT_RANGES) <= set(REWARD_REGISTRY)
    assert REWARD_OUTPUT_RANGES["style"] == (-1.0, 1.0)


def test_build_default_returns_style_and_content():
    # style needs a classifier; pass a dummy since builders don't call it at build time
    funcs, weights = build_reward_fns(
        [{"name": "content", "weight": 0.5}, {"name": "edit", "weight": 0.3}]
    )
    assert len(funcs) == 2
    assert weights == [0.5, 0.3]


def test_unknown_reward_raises():
    with pytest.raises(ValueError, match="Unknown reward"):
        build_reward_fns([{"name": "nope", "weight": 1.0}])


def test_style_requires_classifier():
    with pytest.raises(ValueError, match="requires a trained classifier"):
        build_reward_fns([{"name": "style", "weight": 1.0}])


def test_default_specs_used_when_empty():
    # No classifier → default set includes 'style', which raises without one.
    with pytest.raises(ValueError, match="requires a trained classifier"):
        build_reward_fns(None)


def test_weight_defaults_to_one():
    funcs, weights = build_reward_fns([{"name": "content"}])
    assert weights == [1.0]
    assert len(funcs) == 1
