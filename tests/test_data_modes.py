from __future__ import annotations

import pytest
from datasets import Dataset

from ko_dialect.data.dataset import build_sft_dataset


@pytest.fixture
def raw_ds():
    return Dataset.from_list(
        [
            {
                "standard": "안녕하세요",
                "dialect": "안녕하세유",
                "do": "gangwondo",
                "is_identical": False,
            },
            {
                "standard": "밥 먹었니",
                "dialect": "밥 먹었나",
                "do": "gyeongsangdo",
                "is_identical": False,
            },
        ]
    )


def test_structured_mode_emits_raw_columns(raw_ds, mock_tokenizer):
    out = build_sft_dataset(raw_ds, mock_tokenizer, output_mode="structured")
    cols = set(out["train"].column_names)
    assert cols == {"source", "target", "do", "direction"}
    # both_directions=True → 2 rows per source example
    assert len(out["train"]) == 4


def test_text_mode_emits_rendered_text(raw_ds, mock_tokenizer):
    out = build_sft_dataset(raw_ds, mock_tokenizer, output_mode="text")
    assert "text" in out["train"].column_names
    assert isinstance(out["train"][0]["text"], str)


def test_invalid_output_mode_raises(raw_ds, mock_tokenizer):
    with pytest.raises(ValueError, match="output_mode"):
        build_sft_dataset(raw_ds, mock_tokenizer, output_mode="bogus")


def test_single_direction_halves_rows(raw_ds, mock_tokenizer):
    out = build_sft_dataset(
        raw_ds, mock_tokenizer, both_directions=False, output_mode="structured"
    )
    assert len(out["train"]) == 2
    assert all(d == "std2dia" for d in out["train"]["direction"])
