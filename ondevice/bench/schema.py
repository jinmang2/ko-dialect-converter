"""Single source of truth for the on-device measurement row (brief §6).

One JSONL row = one (variant × config × prompt/gen size) measurement. Speed axes
come from `llama-bench` (parse_bench.py), memory from mem_logger.sh, quality
(chrF / recon_bleu) is merged in later from the desktop scorer (eval/score_offline.py)
— the phone never scores (decision O3).

Keeping the schema in one module means parse_bench / logger / score_offline / report
can't silently drift apart.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Honesty tags (brief §10): every number is one or the other.
PREDICTED = "PREDICTED"
MEASURED = "MEASURED"


@dataclass
class BenchRow:
    """Brief §6 schema. ``None`` = not-yet-filled (kept explicit, never silently 0)."""

    variant: str  # Q4_K_M, Q5_K_M, ...
    runtime: str = "llama.cpp-cpu"
    device: str | None = None  # SM-S938N (getprop ro.product.model)
    n_prompt_tokens: int | None = None
    n_gen_tokens: int | None = None
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    ttft_ms: float | None = None
    peak_rss_mb: float | None = None
    ondisk_mb: float | None = None
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    flash_attn: bool = False
    n_ctx: int = 4096
    prefix_cache: bool = False
    chrF: float | None = None  # noqa: N815 — matches the project-wide metric name (brief §6 schema)
    recon_bleu: float | None = None
    thermal_series: list[float] = field(default_factory=list)
    n_threads: int = 6
    notes: str = MEASURED  # PREDICTED | MEASURED

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(rows: list[BenchRow] | list[dict[str, Any]], path: str | Path) -> Path:
    """Append-safe write of schema rows to JSONL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            obj = r.to_json() if isinstance(r, BenchRow) else r
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def median_drop_warmup(values: list[float], drop_warmup: bool = True) -> float | None:
    """Median over repetitions, discarding the first (cold-cache) run (brief §6).

    ``llama-bench -r N`` already warms up and averages internally; this is for the
    case where we aggregate several *separate* invocations ourselves.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if drop_warmup and len(vals) > 1:
        vals = vals[1:]
    return statistics.median(vals)
