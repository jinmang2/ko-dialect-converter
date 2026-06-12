from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointStage:
    tag: str
    adapter: str | None


_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_step(path: str | Path) -> int | None:
    """Return the numeric step for a `checkpoint-N` path, or None if invalid."""
    match = _CHECKPOINT_RE.match(Path(path).name)
    return int(match.group(1)) if match else None


def discover_grpo_stages(
    grpo_dir: str | Path, *, include_sft: bool = True
) -> list[CheckpointStage]:
    """Return SFT baseline, sorted GRPO checkpoints, and final adapter if present."""
    root = Path(grpo_dir)
    stages: list[CheckpointStage] = []
    if include_sft:
        stages.append(CheckpointStage("SFT(0)", None))

    checkpoints = [
        path
        for path in root.glob("checkpoint-*")
        if path.is_dir() and checkpoint_step(path) is not None
    ]
    for path in sorted(checkpoints, key=lambda p: checkpoint_step(p) or -1):
        stages.append(CheckpointStage(f"step-{checkpoint_step(path)}", str(path)))

    if (root / "adapter_config.json").exists():
        stages.append(CheckpointStage("final", str(root)))
    return stages


def clamp_select_count(total: int, requested: int) -> int:
    if requested < 0:
        raise ValueError("n must be non-negative.")
    return min(requested, total)


def metric_value(metrics: dict[str, Any], key: str, *, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if not isinstance(value, (int, float)) or math.isnan(float(value)):
        return default
    return float(value)


def best_by_metric(
    rows: list[tuple[str, dict[str, Any]]], metric: str
) -> tuple[str, dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot select a best checkpoint from an empty result set.")
    return max(rows, key=lambda row: metric_value(row[1], metric))


def summarize_trainer_state(
    path: str | Path,
    *,
    zero_std_threshold: float = 0.3,
) -> dict[str, Any]:
    """Summarize GRPO training health from a Hugging Face `trainer_state.json`."""
    state_path = Path(path)
    data = json.loads(state_path.read_text(encoding="utf-8"))
    history = [row for row in data.get("log_history", []) if "step" in row]
    last = history[-1] if history else {}

    nan_grad_steps = [
        row["step"]
        for row in history
        if isinstance(row.get("grad_norm"), float) and math.isnan(row["grad_norm"])
    ]
    high_zero_std_steps = [
        row["step"]
        for row in history
        if isinstance(row.get("frac_reward_zero_std"), (int, float))
        and row["frac_reward_zero_std"] > zero_std_threshold
    ]

    return {
        "path": str(state_path),
        "global_step": data.get("global_step"),
        "last_step": last.get("step"),
        "reward": last.get("reward"),
        "reward_std": last.get("reward_std"),
        "frac_reward_zero_std": last.get("frac_reward_zero_std"),
        "kl": last.get("kl"),
        "clipped_ratio": last.get("completions/clipped_ratio"),
        "nan_grad_steps": nan_grad_steps,
        "high_zero_std_steps": high_zero_std_steps,
    }
