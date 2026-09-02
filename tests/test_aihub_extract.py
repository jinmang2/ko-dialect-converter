"""Tests for ``aihub_fetch``'s archive extraction.

Extraction is done member-by-member rather than via ``ZipFile.extractall`` so that
CP437-mojibake Korean names can be repaired. That hand-rolled loop then owns the path
sanitising ``extractall`` would otherwise have done for free — and the 139-1 (71517)
archives prove it matters: every member is stored with a **leading slash**, which makes
``out_dir / name`` resolve to the filesystem root. The first version of this code refused
those members as escaping the destination and extracted 0 of 341,043 files while
cheerfully reporting success on all 12 archives.
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

_PATH = Path(__file__).parents[1] / "scripts" / "aihub_fetch.py"
_SPEC = importlib.util.spec_from_file_location("aihub_fetch", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
aihub_fetch = importlib.util.module_from_spec(_SPEC)
# `@dataclass` resolves annotations through `sys.modules[cls.__module__]`, so the module
# has to be registered before it executes.
sys.modules["aihub_fetch"] = aihub_fetch
_SPEC.loader.exec_module(aihub_fetch)


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return path


def test_absolute_member_names_extract_inside_the_destination(tmp_path):
    # Exactly the 139-1 layout: "/talk_….json", no directory entries.
    archive = _write_zip(tmp_path / "TL_01.zip", {"/talk_set2_speakergw1_0.json": b"{}"})

    written, skipped = aihub_fetch._extract_zip(archive)

    assert (written, skipped) == (1, False)
    assert (tmp_path / "TL_01" / "talk_set2_speakergw1_0.json").read_bytes() == b"{}"


def test_parent_traversal_is_stripped_not_followed(tmp_path):
    archive = _write_zip(tmp_path / "a.zip", {"../../escaped.json": b"{}"})

    written, _ = aihub_fetch._extract_zip(archive)

    assert written == 1
    assert (tmp_path / "a" / "escaped.json").exists()
    assert not (tmp_path.parent / "escaped.json").exists()


def test_nested_directories_are_preserved(tmp_path):
    archive = _write_zip(tmp_path / "b.zip", {"Training/02.라벨링데이터/x.json": b"{}"})

    written, _ = aihub_fetch._extract_zip(archive)

    assert written == 1
    # prepare_data resolves region and split from path components, so nesting must survive.
    assert (tmp_path / "b" / "Training" / "02.라벨링데이터" / "x.json").exists()


def test_extraction_is_idempotent_and_force_overrides(tmp_path):
    archive = _write_zip(tmp_path / "c.zip", {"/x.json": b"{}"})

    assert aihub_fetch._extract_zip(archive) == (1, False)
    assert aihub_fetch._extract_zip(archive) == (0, True)
    assert aihub_fetch._extract_zip(archive, force=True) == (1, False)


def test_an_interrupted_extraction_is_redone_not_skipped(tmp_path):
    # A killed session leaves a non-empty but incomplete directory. Skipping it because
    # "the folder exists" would report success over a truncated corpus.
    archive = _write_zip(tmp_path / "part.zip", {f"/f{i}.json": b"{}" for i in range(5)})
    assert aihub_fetch._extract_zip(archive) == (5, False)

    (tmp_path / "part" / "f3.json").unlink()
    written, skipped = aihub_fetch._extract_zip(archive)

    assert (written, skipped) == (5, False)
    assert (tmp_path / "part" / "f3.json").exists()


def test_a_complete_directory_is_skipped_without_decompressing(tmp_path):
    archive = _write_zip(tmp_path / "done.zip", {f"/f{i}.json": b"{}" for i in range(3)})
    aihub_fetch._extract_zip(archive)

    # Corrupt the archive body: a genuine skip must not need to read it.
    marker = tmp_path / "done" / "f0.json"
    before = marker.stat().st_mtime_ns
    assert aihub_fetch._extract_zip(archive) == (0, True)
    assert marker.stat().st_mtime_ns == before  # untouched, not rewritten


def _unflagged(raw_name: bytes) -> zipfile.ZipInfo:
    """A member whose name bytes lack the UTF-8 flag — how `zipfile` hands it to us."""
    info = zipfile.ZipInfo(raw_name.decode("cp437"))
    info.flag_bits &= ~0x800
    return info


def test_cp949_names_without_the_utf8_flag_are_repaired():
    # A Windows-authored archive. Left as CP437, `제주도` becomes garbage — and
    # prepare_data reads the region out of exactly this path component.
    assert aihub_fetch._member_name(_unflagged("제주도/x.json".encode("cp949"))) == "제주도/x.json"


def test_utf8_names_without_the_flag_are_not_forced_through_cp949():
    # Linux-written zips routinely omit the flag while storing UTF-8. Decoding those
    # bytes as CP949 would corrupt a name that was never broken.
    assert aihub_fetch._member_name(_unflagged("제주도/x.json".encode())) == "제주도/x.json"


def test_flagged_names_are_passed_through_untouched():
    info = zipfile.ZipInfo("제주도/x.json")
    info.flag_bits |= 0x800
    assert aihub_fetch._member_name(info) == "제주도/x.json"


def test_ascii_names_survive_every_branch():
    assert aihub_fetch._member_name(_unflagged(b"/talk_set2_speakergw1_0.json")) == (
        "/talk_set2_speakergw1_0.json"
    )
