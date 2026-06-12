from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_STAGE0_PATH = Path(__file__).parents[1] / "scripts" / "stage0_build_datasets.py"
_SPEC = importlib.util.spec_from_file_location("stage0_build_datasets", _STAGE0_PATH)
assert _SPEC is not None
stage0_build_datasets = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(stage0_build_datasets)

load_raw_manifest = stage0_build_datasets.load_raw_manifest
validate_prosody_manifest = stage0_build_datasets.validate_prosody_manifest


def test_load_raw_manifest_reads_adjacent_sidecar(tmp_path):
    raw_dir = tmp_path / "dialect_raw_new"
    raw_dir.mkdir()
    manifest = raw_dir.with_name("dialect_raw_new_manifest.json")
    manifest.write_text(
        json.dumps({"prosody_marker_policy": {"version": 1}}),
        encoding="utf-8",
    )

    assert load_raw_manifest(str(raw_dir)) == {"prosody_marker_policy": {"version": 1}}


def test_validate_prosody_manifest_requires_matching_policy():
    validate_prosody_manifest({"prosody_marker_policy": {"version": 1}})

    with pytest.raises(FileNotFoundError, match="requires a raw dataset manifest"):
        validate_prosody_manifest(None)

    with pytest.raises(ValueError, match="prosody_marker_policy"):
        validate_prosody_manifest({"prosody_marker_policy": None})

    with pytest.raises(ValueError, match="does not match"):
        validate_prosody_manifest({"prosody_marker_policy": {"version": 999}})
