"""Resolve a Trainer output directory to the *right* checkpoint.

``outputs/<stage>`` accumulates ``checkpoint-*`` subdirs across runs. Picking the
highest step number is wrong once two runs share a directory: an old run that
reached ``checkpoint-14000`` (worse) outranks a newer, better ``checkpoint-1400``.

Resolution order:
1. The path itself if it already holds weights/adapter.
2. ``best_model_checkpoint`` recorded in the newest ``trainer_state.json`` under
   the path (this is what ``load_best_model_at_end`` selected — the metric winner).
3. The checkpoint with the newest mtime (the most recent run's last save).
4. Fallback: highest step number (legacy behaviour).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

WEIGHT_MARKERS = (
    "adapter_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def _has_weights(p: Path) -> bool:
    return any((p / marker).exists() for marker in WEIGHT_MARKERS)


def _best_from_trainer_state(ckpts: list[Path]) -> Path | None:
    """Read the newest checkpoint's trainer_state and return its best_model_checkpoint."""
    newest_state = max(
        (c / "trainer_state.json" for c in ckpts if (c / "trainer_state.json").exists()),
        key=lambda f: f.stat().st_mtime,
        default=None,
    )
    if newest_state is None:
        return None
    try:
        state = json.loads(newest_state.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    best = state.get("best_model_checkpoint")
    if not best:
        return None
    best_path = Path(best)
    if not best_path.is_absolute():
        best_path = newest_state.parent.parent / best_path.name
    if best_path.exists() and _has_weights(best_path):
        metric = state.get("best_metric")
        logger.info("Resolved best checkpoint %s (best_metric=%s)", best_path, metric)
        return best_path
    return None


def resolve_best_checkpoint(model_path: str | Path) -> str:
    """Resolve ``model_path`` to a concrete checkpoint dir (see module docstring)."""
    p = Path(model_path)
    if _has_weights(p):
        return str(p)

    ckpts = [c for c in p.glob("checkpoint-*") if c.is_dir() and _has_weights(c)]
    if not ckpts:
        return str(p)

    best = _best_from_trainer_state(ckpts)
    if best is not None:
        return str(best)

    newest = max(ckpts, key=lambda d: d.stat().st_mtime)
    logger.info("No best_model_checkpoint recorded; using newest-by-mtime %s", newest)
    return str(newest)
