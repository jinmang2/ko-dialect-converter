"""Tests for the GER (ASR error correction) dataset builder.

``stt_hypothesis → standard`` is the only direction whose input exists at inference
time; the other two require a gold dialect transcript nobody has in production. Two of
its rules are deliberate and easy to "simplify" away later, so they are pinned here:

1. It must NOT inherit ``build_sft_dataset``'s ``filter_identical``. ASR errors are
   independent of how dialectal the speech was, so rows with ``standard == dialect``
   (~39% of the corpus) are valid GER supervision.
2. It must keep *some* rows where the ASR was already correct. Dropping all of them
   trains a forced-edit habit — the same failure ``r_overcorrection`` guards against.
"""

from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from ko_dialect.data import build_ger_dataset
from ko_dialect.data.template import ChatTemplate


def _rows(n_identity: int = 0, n_edit: int = 0, **overrides):
    rows = []
    for i in range(n_edit):
        rows.append(
            {
                "standard": f"밥 먹었니 {i}",
                "dialect": f"밥 뭇나 {i}",
                "stt_hypothesis": f"밥 무따 {i}",
                "do": "gyeongsangdo",
                "is_identical": False,
            }
        )
    for i in range(n_identity):
        rows.append(
            {
                "standard": f"안녕하세요 {i}",
                "dialect": f"안녕하세요 {i}",  # identical → build_sft_dataset would drop it
                "stt_hypothesis": f"안녕하세요 {i}",
                "do": "gangwondo",
                "is_identical": True,
            }
        )
    for r in rows:
        r.update(overrides)
    return DatasetDict({"train": Dataset.from_list(rows)})


def test_identical_dialect_rows_are_kept(mock_tokenizer):
    # standard == dialect, but the ASR still got it wrong → real GER supervision.
    ds = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "standard": "밥 먹었습니다",
                        "dialect": "밥 먹었습니다",
                        "stt_hypothesis": "밥 머겄습니다",
                        "do": "chungcheongdo",
                        "is_identical": True,
                    }
                ]
            )
        }
    )
    out = build_ger_dataset(ds, mock_tokenizer, ChatTemplate())
    assert len(out["train"]) == 1


def test_already_correct_asr_rows_are_capped_not_erased(mock_tokenizer):
    out = build_ger_dataset(_rows(n_identity=200, n_edit=200), mock_tokenizer, identity_ratio=0.5)
    kept = len(out["train"])
    # 200 edits always kept; ~half of the 200 identity rows survive.
    assert 200 < kept < 400


def test_identity_ratio_zero_drops_them_and_one_keeps_all(mock_tokenizer):
    none_kept = build_ger_dataset(
        _rows(n_identity=50, n_edit=10), mock_tokenizer, identity_ratio=0.0
    )
    all_kept = build_ger_dataset(
        _rows(n_identity=50, n_edit=10), mock_tokenizer, identity_ratio=1.0
    )
    assert len(none_kept["train"]) == 10
    assert len(all_kept["train"]) == 60


def test_rows_without_an_asr_hypothesis_are_dropped(mock_tokenizer):
    ds = _rows(n_edit=3)
    ds["train"] = ds["train"].map(
        lambda x, i: {"stt_hypothesis": None if i == 0 else x["stt_hypothesis"]}, with_indices=True
    )
    out = build_ger_dataset(ds, mock_tokenizer)
    assert len(out["train"]) == 2


def test_structured_mode_records_the_stt2std_direction(mock_tokenizer):
    out = build_ger_dataset(_rows(n_edit=1), mock_tokenizer, output_mode="structured")
    row = out["train"][0]
    assert row["direction"] == "stt2std"
    assert row["source"] == "밥 무따 0"  # the ASR hypothesis, not the gold dialect
    assert row["target"] == "밥 먹었니 0"


def test_prompt_names_the_task_as_asr_correction(mock_tokenizer):
    out = build_ger_dataset(_rows(n_edit=1), mock_tokenizer)
    text = out["train"][0]["text"]
    assert "음성인식" in text and "표준어" in text
    assert "경상도" in text  # region name, not the raw `gyeongsangdo` code


def test_missing_stt_column_raises_instead_of_yielding_nothing(mock_tokenizer):
    ds = _rows(n_edit=2)
    ds["train"] = ds["train"].remove_columns("stt_hypothesis")
    # v1 corpora have no ASR field; an empty dataset would look like a successful build.
    with pytest.raises(ValueError, match="stt_hypothesis"):
        build_ger_dataset(ds, mock_tokenizer)


def test_unsupported_region_rows_are_filtered(mock_tokenizer):
    out = build_ger_dataset(_rows(n_edit=4, do="atlantis"), mock_tokenizer)
    assert len(out["train"]) == 0


def test_identity_selection_is_seed_stable(mock_tokenizer):
    a = build_ger_dataset(
        _rows(n_identity=100, n_edit=0), mock_tokenizer, identity_ratio=0.3, seed=7
    )
    b = build_ger_dataset(
        _rows(n_identity=100, n_edit=0), mock_tokenizer, identity_ratio=0.3, seed=7
    )
    assert len(a["train"]) == len(b["train"])


def test_invalid_identity_ratio_is_rejected(mock_tokenizer):
    with pytest.raises(ValueError, match="identity_ratio"):
        build_ger_dataset(_rows(n_edit=1), mock_tokenizer, identity_ratio=1.5)
