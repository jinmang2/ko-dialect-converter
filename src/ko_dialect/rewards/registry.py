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

from ._utils import rescale_to_unit
from .content import r_content, r_copy_margin, r_overcorrection
from .edit import r_edit, r_edit_precision, r_edit_recall
from .fluency import make_fluency_reward
from .length import make_length_reward
from .reconstruction import make_reconstruction_reward
from .style import make_style_reward

# A builder receives the shared context and returns a TRL reward callable.
RewardBuilder = Callable[..., Callable]

REWARD_REGISTRY: dict[str, RewardBuilder] = {}

# Native output range of each reward, used by ``normalize=True`` to rescale every
# reward onto a common [0, 1] basis before weighting (see ``rescale_to_unit``).
# r_style is the outlier ([-1, 1]); the rest are already unit-scaled.
REWARD_OUTPUT_RANGES: dict[str, tuple[float, float]] = {
    "style": (-1.0, 1.0),
    "content": (0.0, 1.0),
    "edit": (0.0, 1.0),
    "edit_precision": (0.0, 1.0),
    "edit_recall": (0.0, 1.0),
    "copy_margin": (0.0, 1.0),
    "overcorrection": (0.0, 1.0),
    "reconstruction": (0.0, 1.0),
    "fluency": (0.0, 1.0),
    "length": (0.0, 1.0),
}


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


@register_reward("edit_precision")
def _build_edit_precision(**_: Any):
    return r_edit_precision


@register_reward("edit_recall")
def _build_edit_recall(**_: Any):
    return r_edit_recall


@register_reward("copy_margin")
def _build_copy_margin(**_: Any):
    return r_copy_margin


@register_reward("overcorrection")
def _build_overcorrection(**_: Any):
    return r_overcorrection


@register_reward("fluency")
def _build_fluency(
    ref_model=None,
    ref_tokenizer=None,
    max_length: int = 128,
    fluency_scale: float = 4.0,
    **_: Any,
):
    if ref_model is None or ref_tokenizer is None:
        raise ValueError(
            "reward 'fluency' requires a frozen reference LM (ref_model) + its "
            "tokenizer (ref_tokenizer); reuse the GRPO base model handle."
        )
    return make_fluency_reward(
        ref_model, ref_tokenizer, max_length=max_length, scale=fluency_scale
    )


@register_reward("reconstruction")
def _build_reconstruction(
    ref_model=None,
    ref_tokenizer=None,
    recon_max_new_tokens: int = 64,
    recon_batch_size: int = 8,
    **_: Any,
):
    if ref_model is None or ref_tokenizer is None:
        raise ValueError(
            "reward 'reconstruction' requires a back-translator LM (ref_model) + its "
            "tokenizer (ref_tokenizer); reuse the frozen SFT base handle. This reward "
            "reverse-generates per step (6GB phase-2, smoke-gated — see plan R3)."
        )
    return make_reconstruction_reward(
        ref_model,
        ref_tokenizer,
        max_new_tokens=recon_max_new_tokens,
        batch_size=recon_batch_size,
    )


@register_reward("length")
def _build_length(
    length_max_ratio: float = 1.5, length_tolerance: float = 0.2, **_: Any
):
    return make_length_reward(max_ratio=length_max_ratio, tolerance=length_tolerance)


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
    normalize: bool = False,
    **builder_kwargs: Any,
) -> tuple[list[Callable], list[float]]:
    """Resolve a list of ``{name, weight}`` specs into ``(reward_funcs, reward_weights)``.

    Unknown names raise immediately with the registered set listed.

    When ``normalize=True`` each reward is wrapped so its output is rescaled to a
    common [0, 1] range (via :data:`REWARD_OUTPUT_RANGES`), so the configured weights
    govern the mixture instead of each reward's native scale. Extra ``builder_kwargs``
    (e.g. ``length_max_ratio``) are forwarded to every builder; builders ignore the
    ones they don't use.
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
        fn = builder(
            classifier=classifier,
            cls_tokenizer=cls_tokenizer,
            max_length=max_length,
            **builder_kwargs,
        )
        if normalize:
            lo, hi = REWARD_OUTPUT_RANGES.get(name, (0.0, 1.0))
            fn = rescale_to_unit(fn, lo, hi)
        funcs.append(fn)
        weights.append(weight)
    return funcs, weights
