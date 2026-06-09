from __future__ import annotations

from ko_dialect.rewards._utils import (
    normalize_eojeol,
    rescale_to_unit,
    tokenize_words,
)
from ko_dialect.rewards.edit import r_edit
from ko_dialect.rewards.length import make_length_reward


# --- eojeol normalization (reward-hacking guard) --------------------------------


def test_normalize_eojeol_strips_boundary_punctuation():
    assert normalize_eojeol("갔어,") == "갔어"
    assert normalize_eojeol('"갔어".') == "갔어"
    assert normalize_eojeol("  갔어!  ") == "갔어"
    # internal characters are preserved
    assert normalize_eojeol("3.5") == "3.5"


def test_tokenize_words_normalizes_each_token():
    assert tokenize_words("내가 갔어, 집에.") == {"내가", "갔어", "집에"}


def test_r_edit_robust_to_trailing_punctuation():
    """A standard word present with trailing punctuation must NOT count as removed,
    and a gold dialect word with punctuation must still count as a hit."""
    eojeol_map = [{"standard": "갔다", "dialect": "갔어예"}]
    # Output keeps the standard word punctuated AND uses the dialect word punctuated.
    completions = ["나는 갔다. 그리고 갔어예!"]
    naive_set = set(completions[0].split())  # {"나는", "갔다.", "그리고", "갔어예!"}
    # Sanity: naive whitespace split would mis-handle both words.
    assert "갔다" not in naive_set
    assert "갔어예" not in naive_set

    rewards = r_edit(
        prompts=[""],
        completions=completions,
        standard=["나는 갔다"],
        dialect=["나는 갔어예"],
        dialect_eojeol_map=[eojeol_map],
        direction=["std2dia"],
    )
    # std word "갔다" is still present (normalized) -> removed fraction = 0
    # dia word "갔어예" is present (normalized) -> hit fraction = 1
    # reward = 0.7 * 0 + 0.3 * 1 = 0.3
    assert rewards == [0.3]


def test_r_edit_rewards_actual_removal():
    eojeol_map = [{"standard": "갔다", "dialect": "갔어예"}]
    completions = ["나는 갔어예."]  # removed std, used dialect
    rewards = r_edit(
        prompts=[""],
        completions=completions,
        standard=["나는 갔다"],
        dialect=["나는 갔어예"],
        dialect_eojeol_map=[eojeol_map],
        direction=["std2dia"],
    )
    # removed=1/1, hit=1/1 -> 0.7 + 0.3 = 1.0
    assert rewards == [1.0]


# --- length reward (DAPO-style verbosity guard) ---------------------------------


def test_length_reward_full_when_within_tolerance():
    r_length = make_length_reward(max_ratio=1.5, tolerance=0.2)
    out = r_length(
        prompts=[""],
        completions=["가나다라"],  # len 4 vs gold 4 -> ratio 1.0
        standard=["WXYZ"],
        dialect=["가나다라"],
        direction=["std2dia"],
    )
    assert out == [1.0]


def test_length_reward_zero_when_overlong():
    r_length = make_length_reward(max_ratio=1.5, tolerance=0.2)
    out = r_length(
        prompts=[""],
        completions=["가나다라마바사아"],  # len 8 vs gold 4 -> ratio 2.0 >= 1.5
        standard=["WXYZ"],
        dialect=["가나다라"],
        direction=["std2dia"],
    )
    assert out == [0.0]


def test_length_reward_linear_decay():
    r_length = make_length_reward(max_ratio=1.5, tolerance=0.0)
    # gold len 10, gen len 12 -> ratio 1.2, free=1.0, max=1.5
    # (1.5 - 1.2) / (1.5 - 1.0) = 0.6
    out = r_length(
        prompts=[""],
        completions=["가" * 12],
        standard=["x" * 10],
        dialect=["나" * 10],
        direction=["std2dia"],
    )
    assert abs(out[0] - 0.6) < 1e-9


def test_length_reward_uses_direction_to_pick_reference():
    r_length = make_length_reward(max_ratio=1.5, tolerance=0.2)
    # dia2std -> reference is the standard string
    out = r_length(
        prompts=[""],
        completions=["가나다라"],
        standard=["가나다라"],
        dialect=["가나다라마바사아"],
        direction=["dia2std"],
    )
    assert out == [1.0]


def test_length_reward_rejects_bad_bounds():
    import pytest

    with pytest.raises(ValueError, match="max_ratio"):
        make_length_reward(max_ratio=1.0, tolerance=0.2)


# --- reward rescaling -----------------------------------------------------------


def test_rescale_to_unit_maps_style_range():
    def raw(*_a, **_k):
        return [-1.0, 0.0, 1.0]

    wrapped = rescale_to_unit(raw, -1.0, 1.0)
    assert wrapped() == [0.0, 0.5, 1.0]


def test_rescale_clamps_out_of_range():
    def raw(*_a, **_k):
        return [-2.0, 2.0]

    wrapped = rescale_to_unit(raw, -1.0, 1.0)
    assert wrapped() == [0.0, 1.0]


def test_rescale_noop_on_zero_span():
    def raw(*_a, **_k):
        return [0.3]

    assert rescale_to_unit(raw, 1.0, 1.0)([]) == [0.3]
