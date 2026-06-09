"""Config-driven reward registry.

Adding or swapping a GRPO reward becomes a one-line config change instead of a hand
edit. Each reward is registered under a name with a *builder* that binds
any required context (the classifier + its tokenizer) and returns a TRL-compatible
reward function ``fn(prompts, completions, **columns) -> list[float]``.

Config shape (YAML)::

    rewards:
      - {name: style,   weight: 1.0}
      - {name: content, weight: 0.5}
      - {name: edit,    weight: 0.3}   # optional

``build_reward_fns`` returns the parallel ``(funcs, weights)`` lists TRL needs, so the
weight/func coupling lives in one place instead of two hand-synced lists.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .content import r_content
from .edit import r_edit
from .style import make_style_reward

# A builder receives the shared context and returns a TRL reward callable.
RewardBuilder = Callable[..., Callable]

REWARD_REGISTRY: dict[str, RewardBuilder] = {}


def register_reward(name: str) -> Callable[[RewardBuilder], RewardBuilder]:
    """Decorator: register a reward *builder* under ``name``."""

    def deco(builder: RewardBuilder) -> RewardBuilder:
        REWARD_REGISTRY[name] = builder
        return builder

    return deco


@register_reward("style")
def _build_style(classifier=None, cls_tokenizer=None, max_length: int = 128, **_: Any):
    if classifier is None or cls_tokenizer is None:
        raise ValueError("reward 'style' requires a trained classifier + its tokenizer.")
    return make_style_reward(classifier, cls_tokenizer, max_length)


@register_reward("content")
def _build_content(**_: Any):
    return r_content


@register_reward("edit")
def _build_edit(**_: Any):
    return r_edit


# Default reward set when config omits `rewards` (paper-faithful: style + content).
DEFAULT_REWARDS: list[dict[str, Any]] = [
    {"name": "style", "weight": 1.0},
    {"name": "content", "weight": 0.5},
]


def build_reward_fns(
    reward_specs: list[dict[str, Any]] | None = None,
    *,
    classifier=None,
    cls_tokenizer=None,
    max_length: int = 128,
) -> tuple[list[Callable], list[float]]:
    """Resolve a list of ``{name, weight}`` specs into ``(reward_funcs, reward_weights)``.

    Unknown names raise immediately with the registered set listed.
    """
    specs = reward_specs if reward_specs else DEFAULT_REWARDS

    funcs: list[Callable] = []
    weights: list[float] = []
    for spec in specs:
        name = spec["name"]
        weight = float(spec.get("weight", 1.0))
        try:
            builder = REWARD_REGISTRY[name]
        except KeyError:
            raise ValueError(
                f"Unknown reward {name!r}. Registered: {sorted(REWARD_REGISTRY)}."
            ) from None
        funcs.append(
            builder(
                classifier=classifier,
                cls_tokenizer=cls_tokenizer,
                max_length=max_length,
            )
        )
        weights.append(weight)
    return funcs, weights
