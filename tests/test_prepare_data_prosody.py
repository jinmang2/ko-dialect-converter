from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_PREPARE_DATA_PATH = Path(__file__).parents[1] / "scripts" / "prepare_data.py"
_SPEC = importlib.util.spec_from_file_location("prepare_data", _PREPARE_DATA_PATH)
assert _SPEC is not None
prepare_data = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(prepare_data)

_process_single_json = prepare_data._process_single_json
prosody_marker = prepare_data.prosody_marker
summarize_intonation = prepare_data.summarize_intonation
write_json_file = prepare_data.write_json_file


def test_summarize_intonation_adds_range_and_marker():
    prosody = summarize_intonation([100.0, 112.0, 130.0])

    assert prosody == {
        "f0_mean": 114.0,
        "f0_std": 12.33,
        "f0_start": 100.0,
        "f0_end": 130.0,
        "f0_delta": 0.3,
        "f0_range": 30.0,
    }
    assert prosody_marker(prosody) == "<UP>"


def test_prosody_marker_classification_order():
    assert prosody_marker({"f0_delta": -0.2, "f0_range": 80.0}) == "<DOWN>"
    assert prosody_marker({"f0_delta": 0.0, "f0_range": 80.0}) == "<WAVE>"
    assert prosody_marker({"f0_delta": 0.0, "f0_range": 10.0}) == "<KEEP>"
    assert prosody_marker(None) is None


def test_new_format_parser_emits_prosody_fields(tmp_path):
    sample = {
        "fileName": "say_sample",
        "transcription": {
            "sentences": [
                {
                    "sentenceId": "1",
                    "startTime": "00:00:00.000",
                    "endTime": "00:00:01.000",
                    "standard": "밥 먹었니?",
                    "dialect": "밥 먹었니?",
                    "intonations": [100.0, 118.0, 140.0],
                }
            ],
            "segments": [
                {
                    "startTime": "00:00:00.100",
                    "endTime": "00:00:00.500",
                    "dialect": "밥",
                    "standard": None,
                },
                {
                    "startTime": "00:00:00.600",
                    "endTime": "00:00:00.900",
                    "dialect": "먹었니?",
                    "standard": None,
                },
            ],
        },
        "annotation": {"intents": [], "emotions": []},
    }
    path = tmp_path / "TL_01._강원도_01._1인발화_자유발화" / "Training"
    path.mkdir(parents=True)
    json_path = path / "say_sample.json"
    json_path.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")

    rows = _process_single_json(json_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["do"] == "gangwondo"
    assert row["split"] == "train"
    assert row["speech_kind"] == "say"
    assert row["is_identical"] is True
    assert row["dialect_eojeol_map"] == []
    assert row["prosody_marker"] == "<UP>"
    assert "dialect_prosody" not in row


def test_write_json_file_preserves_korean_text(tmp_path):
    path = tmp_path / "manifest.json"

    write_json_file(path, {"dataset": "방언", "prosody_marker_policy": {"version": 1}})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "dataset": "방언",
        "prosody_marker_policy": {"version": 1},
    }
