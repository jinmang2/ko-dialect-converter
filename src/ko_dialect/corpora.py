"""AI-Hub dialect corpus registry — which corpora exist, and where they live on disk.

That knowledge used to be split across two places that could drift apart: a
description-only ``DIALECT_DATASETS`` dict in ``scripts/aihub_fetch.py`` and a hardcoded
default path in ``scripts/prepare_data.py``. Downloading a corpus therefore meant typing
a destination by hand, which is exactly the step that can put a v1 tree under a v2 root.

That mistake is not loud. ``prepare_data.py`` picks its parser **per tree**
(``--use_old_format``), so an old-format JSON reached by the new-format parser yields
``do="unknown"`` rows that ``stage0``'s ``SUPPORTED_DO`` filter then drops in silence —
a whole corpus disappearing with no error (DATA_EXPANSION_RUNBOOK §1.4 B2). So the
invariant here is **version == directory**: a corpus's on-disk root is derived from its
registry entry, never typed.

Lives at the package root, NOT under ``ko_dialect.data``: that package's ``__init__``
imports the collator, which imports torch. The download and inspection scripts that use
this registry need neither, and on the 6 GB dev box paying for a torch import to resolve
a path is not free. For the same reason the region names below are plain strings rather
than an import of ``ko_dialect.data.labels`` — ``tests/test_corpora.py`` asserts they
stay a subset of ``SUPPORTED_DO``, so the duplication cannot drift unnoticed.

Paths resolve at **call** time, not import time, so ``$KO_DIALECT_DATA_ROOT`` can be
pointed elsewhere (another disk, a test's tmp_path) without reimporting the module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Corpus generations. The value doubles as the on-disk directory name under ``data/aihub``.
V1_2020 = "v1_2020"
V2_2022 = "v2_2022"
VERSIONS: tuple[str, ...] = (V1_2020, V2_2022)


@dataclass(frozen=True)
class Corpus:
    """One AI-Hub dataset: its key, its regions, and how it must be parsed."""

    datasetkey: str
    title: str
    regions: tuple[str, ...]
    version: str
    old_format: bool
    deidentified: bool
    label_gb: float

    @property
    def root(self) -> Path:
        """Extraction root for this corpus — shared with every corpus of the same version."""
        return tree_root(self.version)


# Sizes measured 2026-08-04 via `aihub_fetch.py plan` (labelling files only; the 원천
# audio shards are 100–580 GB per dataset and deliberately out of scope).
#
# `deidentified` tracks AI-Hub's 2026-07 re-upload of privacy-scrubbed data: every
# dataset here got one **except 71517**, so for 71517 the copy already on disk is the
# current release, not a stale one.
CORPORA: dict[str, Corpus] = {
    "118": Corpus("118", "한국어 방언 발화(강원도)", ("gangwondo",), V1_2020, True, True, 0.22),
    "119": Corpus("119", "한국어 방언 발화(경상도)", ("gyeongsangdo",), V1_2020, True, True, 0.33),
    "120": Corpus("120", "한국어 방언 발화(전라도)", ("jeollado",), V1_2020, True, True, 0.09),
    "121": Corpus("121", "한국어 방언 발화(제주도)", ("jejudo",), V1_2020, True, True, 0.36),
    "122": Corpus("122", "한국어 방언 발화(충청도)", ("chungcheongdo",), V1_2020, True, True, 0.34),
    "71517": Corpus(
        "71517",
        "중·노년층 한국어 방언 데이터(강원도, 경상도) — 139-1",
        ("gangwondo", "gyeongsangdo"),
        V2_2022,
        False,
        False,
        4.3,
    ),
    "71558": Corpus(
        "71558",
        "중·노년층 한국어 방언 데이터(충청도, 전라도, 제주도) — 139-2",
        ("chungcheongdo", "jeollado", "jejudo"),
        V2_2022,
        False,
        True,
        4.5,
    ),
}


def data_root() -> Path:
    """Root for downloaded data. ``$KO_DIALECT_DATA_ROOT`` overrides ``<repo>/data``."""
    override = os.environ.get("KO_DIALECT_DATA_ROOT")
    return Path(override).expanduser() if override else REPO_ROOT / "data"


def aihub_root() -> Path:
    """Where ``aihubshell`` downloads land, one subdirectory per corpus version."""
    return data_root() / "aihub"


def tree_root(version: str) -> Path:
    """Extraction root for ``version`` — the path handed to ``prepare_data.py``."""
    if version not in VERSIONS:
        raise ValueError(f"unknown corpus version {version!r}; expected one of {VERSIONS}")
    return aihub_root() / version


def corpus(datasetkey: str | int) -> Corpus:
    """Look up one corpus by AI-Hub dataset key."""
    key = str(datasetkey)
    if key not in CORPORA:
        raise KeyError(f"unknown AI-Hub dataset key {key!r}; known: {sorted(CORPORA)}")
    return CORPORA[key]


def corpora_for(version: str | None = None, old_format: bool | None = None) -> list[Corpus]:
    """Registry entries filtered by version and/or parser format, in registry order."""
    if version is not None and version not in VERSIONS:
        raise ValueError(f"unknown corpus version {version!r}; expected one of {VERSIONS}")
    return [
        c
        for c in CORPORA.values()
        if (version is None or c.version == version)
        and (old_format is None or c.old_format == old_format)
    ]


def uses_old_format(version: str) -> bool:
    """Whether ``prepare_data.py --use_old_format`` applies to a whole version tree.

    Raises if a version ever mixes formats: that would break the version==directory
    invariant this module exists to protect, and callers have no correct action left.
    """
    formats = {c.old_format for c in corpora_for(version=version)}
    if len(formats) != 1:
        raise ValueError(
            f"version {version!r} has mixed parser formats ({formats}) — one tree cannot "
            "be parsed by one parser; split it into separate version directories."
        )
    return formats.pop()


def regions_for(version: str | None = None) -> list[str]:
    """Every dialect region covered by the selected corpora, deduplicated and sorted."""
    return sorted({region for c in corpora_for(version=version) for region in c.regions})
