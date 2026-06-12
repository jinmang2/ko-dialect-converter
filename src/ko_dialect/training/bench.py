"""Pure helpers for the training-throughput benchmark (unsloth vs DeepSpeed).

GPU-free so the throughput/VRAM arithmetic is unit-tested away from the training loop in
``scripts/bench_train.py``. Mirrors the ``evaluation.serving`` split (pure stats in the
library, the model loop in the script).
"""

from __future__ import annotations

from typing import Any


def summarize_train_throughput(
    step_times_s: list[float],
    tokens_per_step: int,
    *,
    peak_vram_mb: float | None = None,
    warmup: int = 1,
) -> dict[str, Any]:
    """Summarise per-step wall times into steps/sec + tokens/sec, dropping ``warmup`` steps.

    The first step(s) include CUDA graph capture / allocator warmup and JIT compilation, so
    they are excluded from the steady-state rate (still reported as ``n_steps_total``).
    """
    if not step_times_s:
        raise ValueError("summarize_train_throughput() needs at least one step time.")
    steady = step_times_s[warmup:] or step_times_s
    n = len(steady)
    total = sum(steady)
    mean_s = total / n
    out: dict[str, Any] = {
        "n_steps_total": len(step_times_s),
        "n_steps_measured": n,
        "step_time_s_mean": round(mean_s, 4),
        "steps_per_sec": round(n / total, 3) if total > 0 else 0.0,
        "tokens_per_sec": round(tokens_per_step / mean_s, 1) if mean_s > 0 else 0.0,
    }
    if peak_vram_mb is not None:
        out["peak_vram_mb"] = round(peak_vram_mb, 1)
    return out


def compare_throughput(runs: dict[str, dict], metric: str = "tokens_per_sec") -> dict[str, Any]:
    """Rank backend throughput runs and report the winner + relative speedups.

    ``runs`` maps a backend label (e.g. ``"unsloth"``, ``"deepspeed"``) to the dict returned
    by :func:`summarize_train_throughput`. Returns the winner and each backend's ratio to it
    on ``metric`` (higher is better) — the honest "did DeepSpeed actually help?" answer.
    """
    if not runs:
        raise ValueError("compare_throughput() needs at least one run.")
    best = max(runs, key=lambda k: runs[k].get(metric, 0.0))
    best_val = runs[best].get(metric, 0.0) or 1.0
    ratios = {k: round((v.get(metric, 0.0)) / best_val, 3) for k, v in runs.items()}
    return {"metric": metric, "winner": best, "ratio_to_winner": ratios}
