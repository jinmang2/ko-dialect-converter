from __future__ import annotations

import pytest

from ko_dialect.rewards import REWARD_REGISTRY, build_reward_fns


def test_registry_has_core_rewards():
    assert {"style", "content", "edit"} <= set(REWARD_REGISTRY)


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
