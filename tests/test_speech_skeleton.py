"""Speech-module skeleton contract tests (CPU, no model download).

Pins the seam built in docs/SPEECH_MODULE_DESIGN.md: importing the package is cheap and
heavy ML deps load lazily; config mirrors the YAML; interfaces fail loudly before a model
is wired rather than at import.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ko_dialect.speech import AudioProsodyExtractor, DialectStandardizerASR, SpeechConfig

_REPO = Path(__file__).parents[1]
_YAML = _REPO / "configs" / "speech" / "default.yaml"


def test_import_does_not_pull_heavy_deps_or_download():
    # In a fresh interpreter, importing the package must NOT import faster_whisper/torch
    # (so it never triggers a weight download just by being imported).
    code = (
        "import sys; import ko_dialect.speech; "
        "assert 'faster_whisper' not in sys.modules, 'faster_whisper imported at module load'; "
        "assert 'torch' not in sys.modules, 'torch imported at module load'; "
        "print('ok')"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(_REPO)
    )
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout


def test_config_defaults_are_6gb_safe():
    cfg = SpeechConfig()
    assert cfg.direction == "dialect2standard"
    assert cfg.precision == "fp16"  # never bf16 on the RTX 2060 (Turing)
    assert cfg.compute_type == "int8"
    assert cfg.language == "ko"
    assert cfg.sample_rate == 16000


def test_no_bf16_configured_value():
    # The RTX 2060 (Turing) has no BF16. Guard the configured VALUES (not comments, which may
    # legitimately say "no bf16"): precision must be fp16 and no YAML value may be bf16.
    cfg = SpeechConfig()
    assert cfg.precision == "fp16"
    loaded = yaml.safe_load(_YAML.read_text(encoding="utf-8"))
    assert "bf16" not in {str(v).lower() for v in loaded.values()}


def test_yaml_keys_are_all_valid_config_fields():
    import dataclasses

    loaded = yaml.safe_load(_YAML.read_text(encoding="utf-8"))
    valid = {f.name for f in dataclasses.fields(SpeechConfig)}
    unknown = set(loaded) - valid
    assert not unknown, f"YAML has keys not in SpeechConfig: {unknown}"


def test_config_load_roundtrips_from_yaml():
    cfg = SpeechConfig.load(_YAML)
    assert cfg.asr_model == "large-v3-turbo"
    assert cfg.regions == ["gangwondo", "gyeongsangdo"]
    # overrides win
    cfg2 = SpeechConfig.load(_YAML, direction="dialect2dialect")
    assert cfg2.direction == "dialect2dialect"


def test_asr_interface_fails_before_load():
    asr = DialectStandardizerASR()
    assert asr.is_loaded is False
    with pytest.raises(RuntimeError):
        asr.transcribe("nonexistent.wav")


def test_transcribe_to_standard_guards_direction():
    asr = DialectStandardizerASR(SpeechConfig(direction="dialect2dialect"))
    with pytest.raises(ValueError):
        asr.transcribe_to_standard("nonexistent.wav")


def test_prosody_extractor_fails_before_load_but_rule_layer_works():
    ex = AudioProsodyExtractor()
    assert ex.is_loaded is False
    with pytest.raises(RuntimeError):
        ex.extract_f0([0.0])
    # Rule layer reuses ko_dialect.data.prosody (model-free) — rising contour -> <UP>.
    assert AudioProsodyExtractor.markers_from_f0([100.0, 112.0, 130.0]) == "<UP>"
