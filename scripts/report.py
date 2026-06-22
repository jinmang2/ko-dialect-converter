#!/usr/bin/env python3
"""Generate a consolidated results report from outputs/eval_logs/*.json.

Keeps a single up-to-date results page in sync with the actual run artifacts (the
hand-maintained EXPERIMENTS tables drift). Discovers eval-log JSONs, classifies each, keeps
the latest per (kind, scope), and renders one markdown file.

    python scripts/report.py                        # -> docs/RESULTS.md
    python scripts/report.py --out stdout            # print to stdout
    python scripts/report.py --eval_logs outputs/eval_logs --out docs/RESULTS.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import fire

from ko_dialect.evaluation.report import (
    build_report,
    classify_artifact,
    drop_superseded_leaderboards,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _scope_key(kind: str, data: dict) -> str:
    """Identity used to keep only the latest artifact per logical scope."""
    if kind == "leaderboard_multi":
        regions = ",".join(data.get("regions", []))
        return f"leaderboard_multi:{data.get('select_by', '?')}:{regions}"
    if kind == "leaderboard":
        return f"leaderboard:{data.get('target_do', '?')}"
    if kind == "quantization":
        # Prefer the explicit run identity; fall back to variant names for older artifacts.
        model = data.get("model") or "_".join(r.get("name", "") for r in data.get("rows", []))
        return f"quantization:{model}:{data.get('target_do', '?')}"
    if kind == "classifier":
        return f"classifier:{data.get('checkpoint', '?')}:{data.get('split', '?')}"
    return kind


def main(eval_logs: str = "outputs/eval_logs", out: str = "docs/RESULTS.md") -> None:
    """Scan ``eval_logs`` for JSON artifacts and write a consolidated markdown report."""
    paths = sorted(Path(eval_logs).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest: dict[str, tuple[str, dict]] = {}
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        kind = classify_artifact(data)
        if kind is None:
            continue
        key = _scope_key(kind, data)
        if key not in latest:  # paths are newest-first, so the first seen is latest
            latest[key] = (kind, data)

    artifacts = drop_superseded_leaderboards(list(latest.values()))
    report = build_report(artifacts)
    logger.info("Recognised %d artifact(s): %s", len(artifacts), ", ".join(sorted(latest)))

    if str(out) == "stdout":
        print(report)
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(report, encoding="utf-8")
        logger.info("Wrote %s", out)


if __name__ == "__main__":
    fire.Fire(main)
