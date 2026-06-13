"""Consolidate eval-log JSON artifacts into one markdown results report.

The hand-maintained tables in ``docs/EXPERIMENTS.md`` drift from the actual run artifacts
(this is how a stale 4-bit footprint number slipped in once). This renders the live
``outputs/eval_logs/*.json`` — leaderboards, quantization trade-offs, classifier reports —
into a single generated section. Pure functions (classify + render) are unit-tested; the
filesystem discovery lives in ``scripts/report.py``.
"""

from __future__ import annotations

from typing import Any

from .metric_registry import header_label

# Leaderboard columns to surface (with direction arrows from the registry).
_LB_COLS = ("reconstruction_bleu", "copy_margin", "chrf", "tdr", "eojeol_accuracy")
_QUANT_COLS = ("size_mb", "latency_ms_p50", "chrf", "copy_margin")


def classify_artifact(data: Any) -> str | None:
    """Classify a parsed eval-log JSON into a known report kind, or None if unrecognised."""
    if not isinstance(data, dict):
        return None
    if "ranking" in data and "rows" in data:
        return "leaderboard"
    if "baseline" in data and isinstance(data.get("rows"), list):
        return "quantization"
    if "macro_f1" in data:
        return "classifier"
    return None


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.3f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
    return str(value)


def render_leaderboard(data: dict) -> str:
    target = data.get("target_do", "?")
    n = data.get("n_samples", "?")
    sel = data.get("select_by", "?")
    best = data.get("best_run", "?")
    ranking = data.get("ranking", [])
    rows = data.get("rows", {})
    lines = [
        f"#### Leaderboard — {target} (n={n}, ranked by {sel}↑) · best: **{best}**",
        "",
        "| run | " + " | ".join(header_label(c) for c in _LB_COLS) + " |",
        "|---|" + "---|" * len(_LB_COLS),
    ]
    for tag in ranking:
        metrics = rows.get(tag, {})
        cells = " | ".join(_fmt(metrics.get(c)) for c in _LB_COLS)
        marker = " ★" if tag == best else ""
        lines.append(f"| {tag}{marker} | {cells} |")
    return "\n".join(lines)


def render_quantization(data: dict) -> str:
    baseline = data.get("baseline", "?")
    scope = ""
    if data.get("model"):
        scope = f" — `{data['model']}`"
        if data.get("target_do"):
            scope += f" [{data['target_do']}]"
    lines = [
        f"#### Quantization trade-off{scope} (baseline: {baseline})",
        "",
        "| variant | " + " | ".join(_QUANT_COLS) + " |",
        "|---|" + "---|" * len(_QUANT_COLS),
    ]
    for row in data.get("rows", []):
        cells = " | ".join(_fmt(row.get(c)) for c in _QUANT_COLS)
        lines.append(f"| {row.get('name', '?')} | {cells} |")
    return "\n".join(lines)


def render_classifier(data: dict) -> str:
    ckpt = data.get("checkpoint", "?")
    split = data.get("split", "?")
    return (
        f"#### Classifier — `{ckpt}` [{split}]: "
        f"macro-F1 {_fmt(data.get('macro_f1'))}, acc {_fmt(data.get('accuracy'))}"
    )


_RENDERERS = {
    "leaderboard": ("Cross-run leaderboards", render_leaderboard),
    "quantization": ("Quantization", render_quantization),
    "classifier": ("Dialect classifier", render_classifier),
}


def build_report(artifacts: list[tuple[str, dict]], *, title: str = "KoDialect — results") -> str:
    """Render ``(kind, data)`` artifacts into one markdown report, grouped by kind.

    Order within a group follows input order (callers pass latest-first). Unknown kinds are
    skipped. Empty input yields a header with a 'no artifacts' note so the output is stable.
    """
    sections: list[str] = [f"# {title}", "", "_Generated from `outputs/eval_logs/*.json`._", ""]
    by_kind: dict[str, list[dict]] = {}
    for kind, data in artifacts:
        if kind in _RENDERERS:
            by_kind.setdefault(kind, []).append(data)

    if not by_kind:
        sections.append("_No recognised eval artifacts found._")
        return "\n".join(sections)

    for kind, (heading, renderer) in _RENDERERS.items():
        items = by_kind.get(kind)
        if not items:
            continue
        sections.append(f"## {heading}")
        sections.append("")
        for data in items:
            sections.append(renderer(data))
            sections.append("")
    return "\n".join(sections).rstrip() + "\n"
