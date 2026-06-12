"""Latency/throughput helpers for the inference-serving benchmark.

Pure, GPU-free statistics so the percentile/throughput maths is unit-tested away from
the model-loading path in ``scripts/bench_serving.py``.
"""

from __future__ import annotations

from typing import Any


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list. ``q`` in [0, 100]."""
    if not sorted_values:
        raise ValueError("percentile() needs at least one value.")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be within [0, 100].")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (q / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def summarize_latencies(
    latencies_s: list[float],
    new_tokens: list[int] | None = None,
) -> dict[str, Any]:
    """Summarise per-request latencies into p50/p95/mean + throughput.

    ``new_tokens`` (generated token counts, request-aligned) enables a tokens/sec
    figure; when omitted only request-level numbers are reported.
    """
    if not latencies_s:
        raise ValueError("summarize_latencies() needs at least one latency.")
    ordered = sorted(latencies_s)
    total_time = sum(latencies_s)
    n = len(latencies_s)
    out: dict[str, Any] = {
        "n_requests": n,
        "latency_ms_p50": round(percentile(ordered, 50) * 1000, 2),
        "latency_ms_p95": round(percentile(ordered, 95) * 1000, 2),
        "latency_ms_mean": round((total_time / n) * 1000, 2),
        "latency_ms_max": round(ordered[-1] * 1000, 2),
        "requests_per_sec": round(n / total_time, 3) if total_time > 0 else 0.0,
    }
    if new_tokens:
        total_tokens = sum(new_tokens)
        out["total_new_tokens"] = total_tokens
        out["tokens_per_sec"] = (
            round(total_tokens / total_time, 2) if total_time > 0 else 0.0
        )
    return out
