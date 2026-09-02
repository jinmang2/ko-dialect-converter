"""Tests for capping low-headroom GRPO prompts.

GRPO scores a *group* of completions per prompt and normalises by the group's spread. When
the gold differs from the source by one word, every completion lands on nearly the same
reward, the advantage variance collapses, and the prompt returns almost no gradient for a
full `num_generations` of compute. Measured on the 5-region build, 28.5% of prompts are
like that (Chungcheong 39.8%).

The cap is deliberately a cap and not a filter: those prompts are what teaches "leave it
alone", and removing them all invites the over-correction failure `r_overcorrection`
exists to punish — the same trade-off as `identity_ratio` in `build_ger_dataset`.
"""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from ko_dialect.data import build_grpo_dataset, eojeol_edit_count


def _rows(n_low: int = 0, n_high: int = 0):
    rows = []
    for i in range(n_low):
        rows.append(
            {  # one eojeol apart
                "standard": f"오늘 날씨가 참 좋습니다 {i}",
                "dialect": f"오늘 날씨가 참 좋습니더 {i}",
                "do": "gyeongsangdo",
                "is_identical": False,
                "dialect_eojeol_map": [],
            }
        )
    for i in range(n_high):
        rows.append(
            {  # several eojeols apart
                "standard": f"그런데 왜 추우냐 얼어 죽겠어 {i}",
                "dialect": f"경헌디 무사 추우냐 얼엉 죽으켜 {i}",
                "do": "jejudo",
                "is_identical": False,
                "dialect_eojeol_map": [],
            }
        )
    return DatasetDict({"train": Dataset.from_list(rows)})


def test_edit_count_is_symmetric_and_alignment_based():
    assert eojeol_edit_count("가 나 다", "가 나 다") == 0
    assert eojeol_edit_count("가 나 다", "가 라 다") == 1
    assert eojeol_edit_count("가 라 다", "가 나 다") == 1
    # An insertion must count as one edit, not shift every following word.
    assert eojeol_edit_count("가 나 다", "가 나 다 라") == 1


def test_default_keeps_every_prompt(mock_tokenizer):
    out = build_grpo_dataset(_rows(n_low=40, n_high=10), mock_tokenizer)
    assert len(out["train"]) == 50


def test_cap_limits_the_low_headroom_share(mock_tokenizer):
    out = build_grpo_dataset(_rows(n_low=400, n_high=100), mock_tokenizer, low_headroom_ratio=0.2)
    kept = out["train"]
    low = sum(1 for r in kept if eojeol_edit_count(r["standard"], r["dialect"]) <= 1)
    assert low / len(kept) == pytest.approx(0.2, abs=0.02)
    assert low > 0  # capped, not filtered out


def test_ratio_zero_drops_them_and_one_keeps_all(mock_tokenizer):
    none_kept = build_grpo_dataset(
        _rows(n_low=30, n_high=10), mock_tokenizer, low_headroom_ratio=0.0
    )
    all_kept = build_grpo_dataset(
        _rows(n_low=30, n_high=10), mock_tokenizer, low_headroom_ratio=1.0
    )
    assert len(none_kept["train"]) == 10
    assert len(all_kept["train"]) == 40


def test_high_headroom_prompts_are_never_dropped(mock_tokenizer):
    out = build_grpo_dataset(_rows(n_low=500, n_high=7), mock_tokenizer, low_headroom_ratio=0.1)
    high = sum(1 for r in out["train"] if eojeol_edit_count(r["standard"], r["dialect"]) > 1)
    assert high == 7


def test_max_edits_threshold_is_configurable(mock_tokenizer):
    # With a threshold of 5, the "high" Jeju rows also count as low-headroom.
    out = build_grpo_dataset(
        _rows(n_low=10, n_high=10),
        mock_tokenizer,
        low_headroom_ratio=0.0,
        low_headroom_max_edits=5,
    )
    assert len(out["train"]) == 0


def test_invalid_ratio_is_rejected(mock_tokenizer):
    with pytest.raises(ValueError, match="low_headroom_ratio"):
        build_grpo_dataset(_rows(n_low=2, n_high=2), mock_tokenizer, low_headroom_ratio=1.5)


def test_capping_is_seed_stable(mock_tokenizer):
    kw = {"low_headroom_ratio": 0.3, "seed": 5}
    a = build_grpo_dataset(_rows(n_low=100, n_high=50), mock_tokenizer, **kw)["train"]["prompt"]
    b = build_grpo_dataset(_rows(n_low=100, n_high=50), mock_tokenizer, **kw)["train"]["prompt"]
    assert a == b
