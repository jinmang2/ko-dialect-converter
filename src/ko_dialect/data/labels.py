"""Canonical dialect-region label map — single source of truth.

The TextCNN classifier (the GRPO style reward) and every region-aware metric/eval must
agree on these integer ids. Kept dependency-light (stdlib only) so reward and evaluation
modules can import it without pulling in ``datasets``/``torch``.

Ids are STABLE and append-only: renumbering an existing region would misalign every
previously trained classifier checkpoint. ``standard`` is 0; dialect regions follow.
"""

from __future__ import annotations

from collections.abc import Iterable

DIALECT_LABELS: dict[str, int] = {
    "standard": 0,
    "gangwondo": 1,
    "gyeongsangdo": 2,
    "jeollado": 3,
    "jejudo": 4,
    "chungcheongdo": 5,
}

LABEL_STANDARD: int = DIALECT_LABELS["standard"]

# Dialect regions only (excludes the standard class).
DO_TO_LABEL: dict[str, int] = {k: v for k, v in DIALECT_LABELS.items() if k != "standard"}

SUPPORTED_DO: frozenset[str] = frozenset(DO_TO_LABEL)


def num_labels_for(labels: Iterable[int]) -> int:
    """Smallest classifier output dimension covering every id in ``labels``.

    Data-driven so a gangwon/gyeongsang-only build stays 3-class while a full 5-region
    build becomes 6-class — existing checkpoints keep working without a global bump.
    Empty input yields 1 (never crashes).
    """
    return max(labels, default=0) + 1


def dialect_region(do: str) -> str:
    """Short region key for grouping in reports (``gangwondo`` -> ``gangwon``).

    Was duplicated in scripts/compare_sft_grpo.py and scripts/eval_grpo_checkpoints.py.
    Anything outside the two regions with data today collapses to ``other``, so a report
    keeps working when a new region lands before its bucket labels do.
    """
    lowered = do.lower()
    if "gangwon" in lowered:
        return "gangwon"
    if "gyeongsang" in lowered:
        return "gyeongsang"
    return "other"
