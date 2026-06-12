#!/usr/bin/env python3
"""Add a ``prosody_marker`` column to an existing raw dialect dataset.

``outputs/dialect_raw_new`` already carries a per-sentence ``prosody`` F0 summary but
predates the marker stage, so it has no ``prosody_marker`` column (which
``build_sft_dataset(include_prosody=True)`` reads). Rather than re-parse the source
AI-Hub JSONs (~600k rows), this maps ``prosody_marker(prosody)`` over the existing
summary and writes a new raw dataset + the manifest sidecar that stage0 validates.

As of marker policy v2 all four classes fire from the stored summary: <WAVE> keys on the
F0 coefficient of variation (f0_std/f0_mean), which is present, instead of the absent
f0_range. The manifest records the active policy + version under ``prosody_marker_policy``.

Usage:
    python scripts/derive_prosody_markers.py \
        --raw outputs/dialect_raw_new --out outputs/dialect_raw_prosody
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import fire

from ko_dialect.data.dataset import load_dialect_dataset
from ko_dialect.data.prosody import PROSODY_MARKER_POLICY, prosody_marker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("derive_prosody")


def main(
    raw: str = "outputs/dialect_raw_new",
    out: str = "outputs/dialect_raw_prosody",
    num_proc: int = 4,
) -> None:
    """Derive prosody_marker from the prosody column and save a marked raw dataset."""
    ds = load_dialect_dataset(raw)
    if "prosody" not in next(iter(ds.values())).column_names:
        raise SystemExit(f"{raw} has no 'prosody' column to derive markers from.")

    def add_marker(batch: dict) -> dict:
        return {"prosody_marker": [prosody_marker(p) for p in batch["prosody"]]}

    marked = ds.map(add_marker, batched=True, num_proc=num_proc, desc="prosody_marker")

    dist = {}
    for split in marked:
        c = Counter(marked[split]["prosody_marker"])
        dist[split] = {str(k): v for k, v in c.items()}
        logger.info("[%s] marker distribution: %s", split, dist[split])

    out_path = Path(out)
    marked.save_to_disk(str(out_path))

    manifest = {
        "prosody_marker_policy": PROSODY_MARKER_POLICY,
        "derived_from": {
            "raw": str(raw),
            "method": "mapped prosody_marker(prosody) over existing F0 summary",
            "note": "v2 policy: <WAVE> keys on f0_std/f0_mean (cv); all four classes fire",
        },
        "marker_distribution": dist,
    }
    manifest_path = out_path.with_name(f"{out_path.name}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s and manifest %s", out_path, manifest_path)


if __name__ == "__main__":
    fire.Fire(main)
