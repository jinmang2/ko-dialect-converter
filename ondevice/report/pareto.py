"""Size × speed × quality Pareto over the on-device variants (brief §8/§9).

Mirrors the dominance logic of `ko_dialect.evaluation.leaderboard.pareto_frontier`
(a run is dominated when another is ≥ on every axis and > on ≥1), but with an explicit
per-axis direction map so we can mix higher-is-better (decode_tok_s, recon_bleu, chrF)
with lower-is-better (ondisk_mb, peak_rss_mb) without coupling to the GRPO metric
registry's orient() table.

The frontier is the *set of defensible choices* — there is no single winner when size,
speed, and quality trade off (brief: don't force one rank).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from schema import read_jsonl  # noqa: E402

# axis -> +1 (higher better) | -1 (lower better)
DIRECTION = {
    "decode_tok_s": +1,
    "prefill_tok_s": +1,
    "recon_bleu": +1,
    "chrF": +1,
    "ondisk_mb": -1,
    "peak_rss_mb": -1,
}
DEFAULT_AXES = ("decode_tok_s", "recon_bleu", "ondisk_mb")


def _oriented(metrics: dict[str, Any], axis: str) -> float:
    v = metrics.get(axis)
    if not isinstance(v, (int, float)) or float(v) != float(v):
        return float("-inf")
    return DIRECTION[axis] * float(v)


def pareto_frontier(rows: list[tuple[str, dict[str, Any]]], axes: tuple[str, ...]) -> list[str]:
    def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
        ge = all(_oriented(a, k) >= _oriented(b, k) for k in axes)
        gt = any(_oriented(a, k) > _oriented(b, k) for k in axes)
        return ge and gt

    frontier = []
    for i, (tag, m) in enumerate(rows):
        if not any(dominates(o, m) for j, (_, o) in enumerate(rows) if j != i):
            frontier.append(tag)
    return frontier


def _representative(measurements: list[dict]) -> dict[str, dict]:
    """One row per variant for the Tier-1 Pareto: the plain f16-KV, no-prefix config.

    (Tier-2 KV-cache experiments live in separate rows and are analysed there, not in
    the size×speed×quality frontier.)
    """
    out: dict[str, dict] = {}
    for r in measurements:
        if (
            r.get("cache_type_k") == "f16"
            and r.get("cache_type_v") == "f16"
            and not r.get("prefix_cache")
        ):
            out.setdefault(r["variant"], r)
    # fall back to whatever exists if nothing matched the strict filter
    if not out:
        for r in measurements:
            out.setdefault(r["variant"], r)
    return out


_COLS = [
    "variant",
    "ondisk_mb",
    "decode_tok_s",
    "prefill_tok_s",
    "peak_rss_mb",
    "chrF",
    "recon_bleu",
    "notes",
]


def report(
    measurements: str = "ondevice/bench/logs/measurements.jsonl",
    axes: str = ",".join(DEFAULT_AXES),
    out_json: str = "ondevice/report/pareto.json",
) -> None:
    """Print the variant table with ★ frontier markers; dump frontier JSON."""
    axis_t = tuple(a.strip() for a in axes.split(",") if a.strip())
    for a in axis_t:
        if a not in DIRECTION:
            raise ValueError(f"unknown axis {a!r}; known: {sorted(DIRECTION)}")

    rows_by_variant = _representative(read_jsonl(measurements))
    rows = list(rows_by_variant.items())
    frontier = set(pareto_frontier(rows, axis_t))

    # ordering: largest→smallest on disk (the natural quant-sweep axis)
    ordered = sorted(rows_by_variant.values(), key=lambda r: -(r.get("ondisk_mb") or 0))
    head = "  ".join(f"{c:>12}" for c in _COLS)
    print(f"\nPARETO over {axis_t}  (★ = frontier)\n{head}")
    print("-" * len(head))
    for r in ordered:
        star = " ★" if r["variant"] in frontier else ""
        cells = []
        for c in _COLS:
            v = r.get(c)
            cells.append(f"{v:>12.2f}" if isinstance(v, float) else f"{str(v):>12}")
        print("  ".join(cells) + star)
    print(f"\nfrontier: {sorted(frontier)}")

    payload = {
        "axes": list(axis_t),
        "direction": {a: DIRECTION[a] for a in axis_t},
        "frontier": sorted(frontier),
        "variants": ordered,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    fire.Fire(report)
