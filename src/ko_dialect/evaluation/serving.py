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
        out["tokens_per_sec"] = round(total_tokens / total_time, 2) if total_time > 0 else 0.0
    return out


def directory_size_mb(path: str) -> float:
    """Total on-disk size (MB) of a model directory or single file."""
    from pathlib import Path

    p = Path(path)
    if p.is_file():
        total = p.stat().st_size
    else:
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)


def quantization_tradeoff(variants: list[dict], baseline: str = "fp16") -> dict:
    """Summarise a PTQ/quantization sweep as a quality × size × latency trade-off.

    Each variant is ``{"name", "size_mb", "latency_ms_p50", "chrf", "copy_margin", ...}``
    (the fields ``bench_serving.py`` already produces, plus ``size_mb``). Returns rows
    annotated with deltas vs the named ``baseline`` variant (the unquantized fp16 model):
    ``size_pct`` (← smaller is better), ``speedup`` (latency baseline/variant, ↑ better),
    and ``chrf_drop`` / ``copy_margin_drop`` (quality lost to quantization, ↓ better) — so
    the genuine PTQ question, "how much quality for how much size/latency", is answered in
    one place. EXPERIMENTS §3: PTQ first, escalate to QAT only if PTQ degrades too much.
    """
    by_name = {v["name"]: v for v in variants}
    base = by_name.get(baseline)
    rows = []
    for v in variants:
        row = dict(v)
        if base and v["name"] != baseline:
            if base.get("size_mb"):
                row["size_pct"] = round(100 * v.get("size_mb", 0) / base["size_mb"], 1)
            if v.get("latency_ms_p50"):
                row["speedup"] = round(base.get("latency_ms_p50", 0) / v["latency_ms_p50"], 2)
            if base.get("chrf") is not None and v.get("chrf") is not None:
                row["chrf_drop"] = round(base["chrf"] - v["chrf"], 3)
            if base.get("copy_margin") is not None and v.get("copy_margin") is not None:
                row["copy_margin_drop"] = round(base["copy_margin"] - v["copy_margin"], 3)
        rows.append(row)
    return {"baseline": baseline, "rows": rows}


def format_tradeoff_table(summary: dict) -> str:
    """Render :func:`quantization_tradeoff` output as a markdown table."""
    header = "| variant | size_mb | size% | speedup | p50 ms | chrf | chrf_drop | copy_margin |"
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        lines.append(
            f"| {r['name']} | {r.get('size_mb', '—')} | {r.get('size_pct', '—')} | "
            f"{r.get('speedup', '—')} | {r.get('latency_ms_p50', '—')} | "
            f"{r.get('chrf', '—')} | {r.get('chrf_drop', '—')} | {r.get('copy_margin', '—')} |"
        )
    return "\n".join(lines)
