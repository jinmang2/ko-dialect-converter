"""Assemble one measurement row (brief §6 schema) and append it to JSONL.

Combines: parsed llama-bench speed (parse_bench) + on-disk size + peak RSS
(mem_logger.sh JSON) + the serve config flags. Quality (chrF/recon_bleu) is left
None here and merged later by eval/score_offline.py — phone generates, desktop scores.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import fire
from parse_bench import parse_text, summarize
from schema import MEASURED, BenchRow, read_jsonl, write_jsonl


def _ondisk_mb(model_path: str | None) -> float | None:
    if model_path and Path(model_path).exists():
        return round(Path(model_path).stat().st_size / (1024 * 1024), 1)
    return None


def _peak_rss_mb(rss_json: str | None) -> float | None:
    """Read peak RSS from mem_logger.sh output (its final {"peak_rss_mb": ...} line)."""
    if not rss_json or not Path(rss_json).exists():
        return None
    peak = None
    for line in Path(rss_json).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "peak_rss_mb" in obj:
            peak = obj["peak_rss_mb"]
        elif "vm_hwm_mb" in obj:  # series samples → take max as fallback
            peak = max(peak or 0.0, obj["vm_hwm_mb"])
    return round(peak, 1) if peak is not None else None


def log_run(
    variant: str,
    bench_log: str | None = None,
    out_jsonl: str = "ondevice/bench/logs/measurements.jsonl",
    model_path: str | None = None,
    rss_json: str | None = None,
    device: str | None = None,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    flash_attn: bool = False,
    n_ctx: int = 4096,
    prefix_cache: bool = False,
    n_threads: int = 6,
    runtime: str = "llama.cpp-cpu",
    notes: str = MEASURED,
    append: bool = True,
) -> None:
    """Build one BenchRow and (append-)write it to ``out_jsonl``."""
    speed: dict[str, Any] = {}
    if bench_log and Path(bench_log).exists():
        speed = summarize(parse_text(Path(bench_log).read_text(encoding="utf-8")))

    row = BenchRow(
        variant=variant,
        runtime=runtime,
        device=device or os.environ.get("ONDEVICE_DEVICE"),
        n_prompt_tokens=speed.get("n_prompt_tokens"),
        n_gen_tokens=speed.get("n_gen_tokens"),
        prefill_tok_s=speed.get("prefill_tok_s"),
        decode_tok_s=speed.get("decode_tok_s"),
        peak_rss_mb=_peak_rss_mb(rss_json),
        ondisk_mb=_ondisk_mb(model_path),
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        flash_attn=flash_attn,
        n_ctx=n_ctx,
        prefix_cache=prefix_cache,
        n_threads=speed.get("n_threads") or n_threads,
        notes=notes,
    )

    existing = read_jsonl(out_jsonl) if append and Path(out_jsonl).exists() else []
    write_jsonl([*existing, row.to_json()], out_jsonl)
    print(f"logged {variant} → {out_jsonl}  ({len(existing) + 1} rows)")
    print(json.dumps(row.to_json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    fire.Fire(log_run)
