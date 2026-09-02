#!/usr/bin/env python3
"""Re-split a built raw dataset so that no speaker appears in more than one split.

Why: AI-Hub ships its own ``Training``/``Validation`` directories and ``prepare_data.py``
inherits that partition verbatim. Measured on ``outputs/dialect_raw_new``, **113 of the
623 validation speakers (18.1%) also appear in training** — so a held-out score is partly
rewarding a memorised idiolect rather than generalisation to an unseen speaker. For a
style-transfer task this matters more than usual: a speaker's dialect habits are exactly
the thing the model is supposed to learn to convert, and the same person's utterances on
both sides is the closest thing to label leakage this corpus has.

This does NOT overwrite the original build. It writes a new dataset directory, so every
previously reported number stays reproducible against the original split, and the
speaker-disjoint split is an **opt-in second view** you can report alongside it.

Grouping is by speaker, so a 2인발화 (two-speaker) row belongs to a *speaker set*; rows
whose speakers straddle a proposed boundary are assigned by connected component — two
speakers who ever share a file are kept together, otherwise the disjointness claim is
false for exactly the dialogue rows.

Usage:
    python scripts/resplit_speaker_disjoint.py --raw_dataset_path outputs/dialect_raw_new
    python scripts/resplit_speaker_disjoint.py --raw_dataset_path outputs/dialect_raw_new \
        --out outputs/dialect_raw_new_spk --valid_ratio 0.1 --verify
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fire  # noqa: E402

from ko_dialect.ids import speakers_of  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("resplit_speaker_disjoint")


class _Union:
    """Union-find over speaker ids, so co-occurring speakers land in the same split."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _assign_groups(
    group_rows: dict[str, list[int]],
    group_key: dict[str, str],
    valid_ratio: float,
) -> set[str]:
    """Greedy largest-first bin packing → the set of groups assigned to valid.

    Deterministic: groups are ordered by (-size, key). Balances per-region so a region
    does not vanish from valid — a plain global split can starve the smaller region.
    """
    target: dict[str, float] = defaultdict(float)
    current: dict[str, float] = defaultdict(float)
    for group, rows in group_rows.items():
        target[group_key[group]] += len(rows) * valid_ratio

    valid_groups: set[str] = set()
    ordered = sorted(group_rows, key=lambda g: (-len(group_rows[g]), g))
    for group in ordered:
        key = group_key[group]
        if current[key] + len(group_rows[group]) / 2 <= target[key]:
            valid_groups.add(group)
            current[key] += len(group_rows[group])
    return valid_groups


def main(
    raw_dataset_path: str = "outputs/dialect_raw_new",
    out: str | None = None,
    valid_ratio: float = 0.1,
    verify: bool = True,
    manifest: bool = True,
) -> None:
    """Rebuild the dataset with a speaker-disjoint train/valid partition.

    Args:
        raw_dataset_path: Existing built dataset (a ``save_to_disk`` DatasetDict).
        out: Destination. Defaults to ``<raw_dataset_path>_spk``.
        valid_ratio: Target fraction of rows in valid, balanced per region.
        verify: Re-check disjointness after writing and fail loudly if violated.
        manifest: Copy the source manifest and stamp the split policy onto it.
    """
    from datasets import DatasetDict, concatenate_datasets, load_from_disk

    src = Path(raw_dataset_path)
    dst = Path(out) if out else src.with_name(src.name + "_spk")
    ds_dict = load_from_disk(str(src))

    merged = concatenate_datasets([ds_dict[s] for s in sorted(ds_dict)])
    logger.info("merged %s -> %d rows", sorted(ds_dict), len(merged))

    ids, dos = merged["id"], merged["do"]
    union = _Union()
    unassigned: list[int] = []
    for i, sample_id in enumerate(ids):
        speakers = speakers_of(sample_id)
        if not speakers:
            unassigned.append(i)
            continue
        for other in speakers[1:]:
            union.union(speakers[0], other)

    if unassigned:
        logger.warning(
            "%d rows carry no speaker id (old-format ids). They are put in TRAIN, since "
            "an unattributable row cannot be certified speaker-disjoint.",
            len(unassigned),
        )

    group_rows: dict[str, list[int]] = defaultdict(list)
    group_key: dict[str, str] = {}
    for i, (sample_id, do) in enumerate(zip(ids, dos, strict=True)):
        speakers = speakers_of(sample_id)
        if not speakers:
            continue
        group = union.find(speakers[0])
        group_rows[group].append(i)
        group_key.setdefault(group, do)

    valid_groups = _assign_groups(group_rows, group_key, valid_ratio)
    valid_idx = [i for g in valid_groups for i in group_rows[g]]
    valid_set = set(valid_idx)
    train_idx = [i for i in range(len(merged)) if i not in valid_set]

    logger.info(
        "groups=%d (valid %d) -> train %d rows / valid %d rows (%.1f%%)",
        len(group_rows),
        len(valid_groups),
        len(train_idx),
        len(valid_idx),
        100 * len(valid_idx) / max(len(merged), 1),
    )

    new_dict = DatasetDict(
        {"train": merged.select(sorted(train_idx)), "valid": merged.select(sorted(valid_idx))}
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    new_dict.save_to_disk(str(dst))
    logger.info("wrote %s", dst)

    if manifest:
        src_manifest = src.with_name(f"{src.name}_manifest.json")
        payload: dict[str, Any] = {}
        if src_manifest.is_file():
            payload = json.loads(src_manifest.read_text(encoding="utf-8"))
        payload.update(
            {
                "dataset": dst.name,
                "derived_from": src.name,
                "split_policy": "speaker_disjoint",
                "valid_ratio_target": valid_ratio,
                "speaker_groups": len(group_rows),
                "rows_without_speaker_id": len(unassigned),
            }
        )
        out_manifest = dst.with_name(f"{dst.name}_manifest.json")
        out_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", out_manifest)

    if verify:
        seen: dict[str, set[str]] = {}
        for split in ("train", "valid"):
            found: set[str] = set()
            for sample_id in new_dict[split]["id"]:
                found.update(speakers_of(sample_id))
            seen[split] = found
        shared = seen["train"] & seen["valid"]
        print("\n" + "=" * 70)
        print(f"VERIFY  train {len(seen['train']):,} speakers / valid {len(seen['valid']):,}")
        print(f"        shared = {len(shared)}")
        print("=" * 70)
        if shared:
            raise SystemExit(f"FAILED: {len(shared)} speakers still shared: {sorted(shared)[:8]}")
        print("OK — speaker-disjoint.")
        per_region: dict[str, int] = defaultdict(int)
        for do in new_dict["valid"]["do"]:
            per_region[do] += 1
        print(f"valid per region: {dict(sorted(per_region.items()))}")


if __name__ == "__main__":
    fire.Fire(main)
