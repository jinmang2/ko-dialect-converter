"""Structured W&B panels for richer training/eval monitoring.

TRL's GRPOTrainer already logs per-reward-function means automatically, but two things the
project cares about were never surfaced to W&B: the **prosody marker distribution** (is an
arm collapsing to one boundary tone?) and **eval-metric panels** (the leaderboard / A/B
scripts logged nothing). These builders turn raw counts / metric dicts into namespaced,
W&B-loggable payloads; ``log_panels`` writes them and is a safe no-op when no run is active
(CI, ``logger=disabled``), so callers never need to guard the import.

The builders are pure and unit-tested; ``log_panels`` is the only side-effecting function.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def marker_distribution_panel(counts: dict[str, int], prefix: str = "prosody") -> dict[str, float]:
    """Normalise prosody marker counts into a ``{prefix}/share/<marker>`` panel (+ totals).

    Emits both the fraction (so a collapse to one marker is visible as a share → 1.0) and the
    raw count per marker. ``None``/missing markers are bucketed under ``none``.
    """
    total = sum(counts.values())
    panel: dict[str, float] = {f"{prefix}/total": float(total)}
    for marker, n in counts.items():
        key = str(marker).strip("<>").lower() if marker not in (None, "None") else "none"
        panel[f"{prefix}/share/{key}"] = round(n / total, 4) if total else 0.0
        panel[f"{prefix}/count/{key}"] = float(n)
    return panel


def metrics_panel(metrics: dict[str, Any], section: str) -> dict[str, float]:
    """Flatten a metric dict into ``{section}/<metric>`` numeric panels (drops non-numbers)."""
    panel: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            panel[f"{section}/{k}"] = float(v)
    return panel


def wandb_run_active() -> bool:
    """True only if wandb is importable and has a live run (else logging is a no-op)."""
    try:
        import wandb
    except ImportError:
        return False
    return wandb.run is not None


def log_panels(payload: dict[str, Any], step: int | None = None) -> bool:
    """Log a panel dict to the active W&B run; no-op (returns False) when none is active."""
    if not payload or not wandb_run_active():
        return False
    import wandb

    wandb.log(payload, step=step)
    return True
