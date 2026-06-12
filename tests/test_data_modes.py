from __future__ import annotations

import pytest
from datasets import Dataset

from ko_dialect.data import dataset as data_dataset
from ko_dialect.data.prosody import PROSODY_MARKER_POLICY


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
    out = data_dataset.build_sft_dataset(raw_ds, mock_tokenizer, output_mode="structured")
    cols = set(out["train"].column_names)
    assert cols == {"source", "target", "do", "direction"}
    # both_directions=True → 2 rows per source example
    assert len(out["train"]) == 4


def test_text_mode_emits_rendered_text(raw_ds, mock_tokenizer):
    out = data_dataset.build_sft_dataset(raw_ds, mock_tokenizer, output_mode="text")
    assert "text" in out["train"].column_names
    assert isinstance(out["train"][0]["text"], str)


def test_invalid_output_mode_raises(raw_ds, mock_tokenizer):
    with pytest.raises(ValueError, match="output_mode"):
        data_dataset.build_sft_dataset(raw_ds, mock_tokenizer, output_mode="bogus")


def test_single_direction_halves_rows(raw_ds, mock_tokenizer):
    out = data_dataset.build_sft_dataset(
        raw_ds, mock_tokenizer, both_directions=False, output_mode="structured"
    )
    assert len(out["train"]) == 2
    assert all(d == "std2dia" for d in out["train"]["direction"])


def test_add_sentence_final_marker_before_punctuation():
    assert data_dataset.add_sentence_final_marker("밥 먹었나?", "<UP>") == "밥 먹었나<UP>?"
    assert data_dataset.add_sentence_final_marker("밥 먹었나", "<DOWN>") == "밥 먹었나<DOWN>"
    assert data_dataset.add_sentence_final_marker("밥 먹었나?", None) == "밥 먹었나?"


def test_prosody_policy_is_versioned():
    assert PROSODY_MARKER_POLICY["version"] == 1
    assert PROSODY_MARKER_POLICY["up_threshold_delta_ratio"] == 0.15
    assert PROSODY_MARKER_POLICY["placement"].startswith("before")


def test_include_prosody_uses_marked_dialect_side(mock_tokenizer):
    raw = Dataset.from_list(
        [
            {
                "standard": "밥 먹었니?",
                "dialect": "밥 먹었나?",
                "prosody_marker": "<UP>",
                "do": "gyeongsangdo",
                "is_identical": False,
            },
        ]
    )

    out = data_dataset.build_sft_dataset(
        raw,
        mock_tokenizer,
        output_mode="structured",
        include_prosody=True,
    )

    rows = out["train"]
    assert rows[0]["direction"] == "std2dia"
    assert rows[0]["target"] == "밥 먹었나<UP>?"
    assert rows[1]["direction"] == "dia2std"
    assert rows[1]["source"] == "밥 먹었나<UP>?"


def test_include_prosody_warns_when_marker_is_missing(mock_tokenizer, caplog):
    raw = Dataset.from_list(
        [
            {
                "standard": "밥 먹었니?",
                "dialect": "밥 먹었나?",
                "prosody_marker": None,
                "do": "gyeongsangdo",
                "is_identical": False,
            },
        ]
    )

    out = data_dataset.build_sft_dataset(
        raw,
        mock_tokenizer,
        output_mode="structured",
        include_prosody=True,
    )

    assert out["train"][0]["target"] == "밥 먹었나?"
    assert "no prosody_marker" in caplog.text


def test_prosody_marker_coverage_uses_sft_filters():
    raw = Dataset.from_list(
        [
            {
                "standard": "밥 먹었니?",
                "dialect": "밥 먹었나?",
                "prosody_marker": "<UP>",
                "do": "gyeongsangdo",
                "is_identical": False,
            },
            {
                "standard": "뭐 하니?",
                "dialect": "뭐 하노?",
                "prosody_marker": None,
                "do": "gyeongsangdo",
                "is_identical": False,
            },
            {
                "standard": "제외",
                "dialect": "제외",
                "prosody_marker": "<KEEP>",
                "do": "gyeongsangdo",
                "is_identical": True,
            },
            {
                "standard": "제외",
                "dialect": "제외",
                "prosody_marker": "<KEEP>",
                "do": "unknown",
                "is_identical": False,
            },
        ]
    )

    assert data_dataset.prosody_marker_coverage(raw) == {
        "train": {
            "total_rows": 2,
            "marked_rows": 1,
            "missing_rows": 1,
            "coverage_ratio": 0.5,
        }
    }
