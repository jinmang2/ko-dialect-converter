"""Config for the speech (dialect-audio -> standard-text) module.

Mirrors ``configs/speech/default.yaml`` and follows the same load/override contract as
``ko_dialect.evaluation.config.EvalConfig``. Defaults are 6GB RTX 2060 (fp16, no BF16) safe.
See ``docs/SPEECH_MODULE_DESIGN.md``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "configs/speech/default.yaml"


@dataclass
class SpeechConfig:
    """Speech-module knobs. Mirrors ``configs/speech/default.yaml``."""

    # Backbone / task
    asr_model: str = "large-v3-turbo"  # faster-whisper size id (Whisper-large-v3-turbo, ko ✓)
    direction: str = "dialect2standard"  # target=standard_form; "dialect2dialect"=pure ASR
    language: str = "ko"

    # Inference (faster-whisper / CTranslate2)
    device: str = "cuda"
    compute_type: str = "int8"  # int8 ≈ 1.6GB for turbo; fp16 also ok on Turing

    # Training (LoRA on the HF Whisper backbone) — fp16 ONLY (RTX 2060 has no BF16)
    hf_backbone: str = "openai/whisper-large-v3-turbo"
    precision: str = "fp16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    batch_size: int = 8
    grad_accum: int = 4

    # Audio
    sample_rate: int = 16000
    max_audio_seconds: float = 30.0

    # Prosody (real-audio F0; validates the JSON-derived markers)
    prosody_extractor: str = "pesto"  # "pesto" | "fcpe" | "none"

    # Data / eval scope (reuses the existing raw dialect dataset + region convention)
    raw_dataset_path: str = "outputs/dialect_raw_new"
    regions: list[str] = field(default_factory=lambda: ["gangwondo", "gyeongsangdo"])

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> SpeechConfig:
        """Load YAML defaults (if present), then apply non-None keyword overrides.

        Unknown YAML keys are ignored so an older config keeps working as the dataclass grows.
        """
        data: dict[str, Any] = {}
        cfg_path = Path(path or DEFAULT_CONFIG_PATH)
        if cfg_path.exists():
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            valid = {f.name for f in dataclasses.fields(cls)}
            data = {k: v for k, v in loaded.items() if k in valid}
        cfg = cls(**data)
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
