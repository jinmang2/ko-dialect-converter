"""Tests for building GRPO prompts in the deployed (dia2std) direction.

The whole reward stack already reads a per-row ``direction`` and swaps gold/source, so
the only thing that pinned this project to std2dia was one hardcoded argument in
stage0. These tests pin the behaviour that argument now controls — in particular that
the source really is the dialect side, since getting it backwards would train the model
on prompts whose "input" is the answer.
"""

from __future__ import annotations

from datasets import Dataset, DatasetDict

from ko_dialect.data import build_grpo_dataset
from ko_dialect.data.template import ChatTemplate


def _ds():
    return DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "standard": "밥 먹었니",
                        "dialect": "밥 뭇나",
                        "do": "gyeongsangdo",
                        "is_identical": False,
                        "dialect_eojeol_map": [],
                    }
                ]
            )
        }
    )


def test_dia2std_prompts_take_the_dialect_side_as_input(mock_tokenizer):
    out = build_grpo_dataset(_ds(), mock_tokenizer, ChatTemplate(), direction="dia2std")
    row = out["train"][0]
    assert row["direction"] == "dia2std"
    assert "밥 뭇나" in row["prompt"]  # the dialect is the input …
    assert "밥 먹었니" not in row["prompt"]  # … and the standard answer is not leaked


def test_std2dia_remains_the_default_and_is_unchanged(mock_tokenizer):
    out = build_grpo_dataset(_ds(), mock_tokenizer, ChatTemplate())
    row = out["train"][0]
    assert row["direction"] == "std2dia"
    assert "밥 먹었니" in row["prompt"]
    assert "밥 뭇나" not in row["prompt"]


def test_both_directions_keep_the_reward_columns(mock_tokenizer):
    # Rewards need standard+dialect regardless of direction to compute gold vs source.
    for direction in ("std2dia", "dia2std"):
        row = build_grpo_dataset(_ds(), mock_tokenizer, direction=direction)["train"][0]
        assert row["standard"] == "밥 먹었니"
        assert row["dialect"] == "밥 뭇나"
        assert "dialect_eojeol_map" in row


def test_prompt_wording_matches_the_requested_direction(mock_tokenizer):
    dia2std = build_grpo_dataset(_ds(), mock_tokenizer, direction="dia2std")["train"][0]["prompt"]
    std2dia = build_grpo_dataset(_ds(), mock_tokenizer, direction="std2dia")["train"][0]["prompt"]
    assert "표준어로" in dia2std
    assert "사투리로" in std2dia
