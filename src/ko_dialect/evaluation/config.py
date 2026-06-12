"""Config-driven evaluation defaults.

Training is config-driven (Hydra ``configs/*.yaml``); evaluation was not — every script
hard-coded ``n=150``, ``batch_size=16``, the classifier path, the region list, etc. in its
own ``fire`` signature, so changing a shared default meant editing five scripts. This holds
those defaults in one ``configs/eval/default.yaml`` + dataclass; ``scripts/eval.py`` and the
specialised scripts read it and let CLI flags override per-field.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "configs/eval/default.yaml"


@dataclass
class EvalConfig:
    """Shared evaluation knobs. Mirrors ``configs/eval/default.yaml``."""

    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    classifier_path: str = "outputs/classifier_clean"
    cls_tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    raw_dataset_path: str = "outputs/dialect_raw_new"
    runs_dir: str = "outputs"
    split: str = "valid"
    n_samples: int = 150
    batch_size: int = 16
    max_new_tokens: int = 64
    regions: list[str] = field(default_factory=lambda: ["gangwondo", "gyeongsangdo"])
    # metrics to surface in reports; empty => the full registry order
    metrics: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> EvalConfig:
        """Load YAML defaults (if the file exists) then apply non-None keyword overrides.

        Unknown YAML keys are ignored with a soft warning rather than crashing, so an older
        config file keeps working as the dataclass grows.
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
