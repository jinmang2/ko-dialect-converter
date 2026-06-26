"""Plots for the on-device report (brief §8.2): size–speed–quality Pareto +
prefill/decode decomposition bar. Reads the measurement JSONL, writes PNG+SVG.

matplotlib only (no seaborn) to keep deps minimal. Frontier points are filled+ringed;
dominated points hollow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from schema import read_jsonl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pareto import DEFAULT_AXES, _representative, pareto_frontier  # noqa: E402


def _save(fig, out_stem: str) -> None:
    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {out}.png / .svg")


def pareto_scatter(rows_by_variant: dict, frontier: set, out_stem: str) -> None:
    import matplotlib.pyplot as plt

    rows = list(rows_by_variant.values())
    xs = [r.get("ondisk_mb") for r in rows]
    ys = [r.get("decode_tok_s") for r in rows]
    qs = [r.get("recon_bleu") for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        xs,
        ys,
        c=[q if q is not None else float("nan") for q in qs],
        s=170,
        cmap="viridis",
        edgecolors=["crimson" if r["variant"] in frontier else "none" for r in rows],
        linewidths=2.2,
        zorder=3,
    )
    for r, x, y in zip(rows, xs, ys):
        if x is None or y is None:
            continue
        star = "★ " if r["variant"] in frontier else ""
        ax.annotate(
            f"{star}{r['variant']}", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9
        )
    ax.set_xlabel("on-disk size (MB)  ← smaller better")
    ax.set_ylabel("decode tok/s  → higher better")
    ax.set_title("On-device Pareto: size × speed (color = recon_bleu)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("recon_bleu (quality)")
    ax.grid(alpha=0.3, zorder=0)
    _save(fig, out_stem)
    plt.close(fig)


def prefill_decode_bar(rows_by_variant: dict, out_stem: str) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(rows_by_variant.values(), key=lambda r: -(r.get("ondisk_mb") or 0))
    labels = [r["variant"] for r in rows]
    prefill = [r.get("prefill_tok_s") or 0 for r in rows]
    decode = [r.get("decode_tok_s") or 0 for r in rows]
    x = range(len(labels))
    w = 0.4

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - w / 2 for i in x], prefill, w, label="prefill (pp) tok/s")
    ax.bar([i + w / 2 for i in x], decode, w, label="decode (tg) tok/s")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("tok/s")
    ax.set_title("Prefill vs decode throughput by variant")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_stem)
    plt.close(fig)


def make(
    measurements: str = "ondevice/bench/logs/measurements.jsonl",
    out_dir: str = "ondevice/report/figs",
    axes: str = ",".join(DEFAULT_AXES),
) -> None:
    """Render both figures from the measurement JSONL."""
    rows_by_variant = _representative(read_jsonl(measurements))
    axis_t = tuple(a.strip() for a in axes.split(",") if a.strip())
    frontier = set(pareto_frontier(list(rows_by_variant.items()), axis_t))
    pareto_scatter(rows_by_variant, frontier, f"{out_dir}/pareto")
    prefill_decode_bar(rows_by_variant, f"{out_dir}/prefill_decode")


if __name__ == "__main__":
    fire.Fire(make)
