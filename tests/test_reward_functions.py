from __future__ import annotations

from ko_dialect.rewards._utils import (
    normalize_eojeol,
    rescale_to_unit,
    tokenize_words,
)
from ko_dialect.rewards.content import r_copy_margin
from ko_dialect.rewards.edit import r_edit, r_edit_precision, r_edit_recall
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


# --- edit precision / recall (collateral-damage guard) --------------------------


def test_r_edit_precision_perfect_when_nontarget_preserved():
    # src=std "나는 어제 갔다", target eojeol = 갔다->갔어예.
    # Non-target words {나는, 어제} must survive; gen keeps them -> precision 1.0.
    eojeol_map = [{"standard": "갔다", "dialect": "갔어예"}]
    out = r_edit_precision(
        prompts=[""],
        completions=["나는 어제 갔어예"],
        standard=["나는 어제 갔다"],
        dialect=["나는 어제 갔어예"],
        dialect_eojeol_map=[eojeol_map],
        direction=["std2dia"],
    )
    assert out == [1.0]


def test_r_edit_precision_penalises_collateral_damage():
    # Model mangles a non-target word (어제 -> 어쩨): {나는, 어제} -> only 나는 kept.
    eojeol_map = [{"standard": "갔다", "dialect": "갔어예"}]
    out = r_edit_precision(
        prompts=[""],
        completions=["나는 어쩨 갔어예"],
        standard=["나는 어제 갔다"],
        dialect=["나는 어제 갔어예"],
        dialect_eojeol_map=[eojeol_map],
        direction=["std2dia"],
    )
    assert out == [0.5]  # 1 of 2 non-target words preserved


def test_r_edit_recall_requires_correct_gold_form():
    eojeol_map = [{"standard": "갔다", "dialect": "갔어예"}]
    # Correct gold dialect form present -> recall 1.0
    hit = r_edit_recall(
        prompts=[""],
        completions=["나는 갔어예"],
        standard=["나는 갔다"],
        dialect=["나는 갔어예"],
        dialect_eojeol_map=[eojeol_map],
        direction=["std2dia"],
    )
    assert hit == [1.0]
    # Word merely dropped (no gold form) -> recall 0.0 (cannot be hacked by deletion)
    miss = r_edit_recall(
        prompts=[""],
        completions=["나는"],
        standard=["나는 갔다"],
        dialect=["나는 갔어예"],
        dialect_eojeol_map=[eojeol_map],
        direction=["std2dia"],
    )
    assert miss == [0.0]


# --- copy-margin (Gangwon copy-bias guard) --------------------------------------


def test_copy_margin_low_when_copying_source():
    # gold ≈ source (Gangwon-like). Echoing the source -> margin ~0 -> ~0.5.
    out = r_copy_margin(
        prompts=[""],
        completions=["나는 학교에 간다"],  # == source
        standard=["나는 학교에 간다"],
        dialect=["나는 학교에 간다요"],
        direction=["std2dia"],
    )
    assert out[0] < 0.6  # near the 0.5 copy floor, not a high content score


def test_copy_margin_high_when_moving_to_gold():
    # Output matches gold and differs from source -> positive margin -> > 0.5.
    out = r_copy_margin(
        prompts=[""],
        completions=["내가 학교에 가니더"],  # == gold
        standard=["나는 학교에 간다"],
        dialect=["내가 학교에 가니더"],
        direction=["std2dia"],
    )
    assert out[0] > 0.5


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
