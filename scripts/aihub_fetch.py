#!/usr/bin/env python3
"""Browse and selectively fetch AI-Hub dialect datasets via ``aihubshell``.

Why this exists (vs ``scripts/download_from_aihub.sh``): that wrapper grabs **every**
``TL_``/``VL_`` file of a dataset in one shot. For the dialect corpora that is 4.5 GB of
labels — and one careless ``-filekey`` typo pulls a 28 GB audio shard. This script makes
the size explicit *before* downloading, refuses anything over a threshold unless you opt
in, and can pull the single smallest labelling file so you can eyeball the schema first.

Auth: the AI-Hub API key is read from ``.env.local`` (``AIHUB_API_KEY=...``) or the
``AIHUB_API_KEY`` / ``AIHUB_APIKEY`` environment variables. ``.env.local`` is gitignored.
Listing (``datasets``/``tree``/``plan``) works **without** a key; only ``download``/
``sample`` need one, plus a currently-valid data-use approval on aihub.or.kr.

Usage:
    python scripts/aihub_fetch.py datasets --grep 방언        # find dataset keys
    python scripts/aihub_fetch.py plan --datasetkey 71558     # label vs audio, sizes, keys
    python scripts/aihub_fetch.py sample --datasetkey 71558   # smallest label zip only
    python scripts/aihub_fetch.py fetch --version v2_2022     # every label of one generation
    python scripts/aihub_fetch.py download --datasetkey 71558 --kind label --split valid
    python scripts/aihub_fetch.py extract --version v2_2022   # unzip what was downloaded
    python scripts/aihub_fetch.py inspect --path data/aihub_samples  # schema of one JSON
    python scripts/aihub_fetch.py diff_trees --a old/ --b new/       # are two copies equal?
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import fire

from ko_dialect.corpora import CORPORA, VERSIONS, corpora_for, corpus, tree_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aihub_fetch")

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_LOCAL = REPO_ROOT / ".env.local"

# Dataset keys and where each one belongs on disk both come from the registry, so a
# download destination is derived rather than typed — see `ko_dialect.corpora`.
DIALECT_DATASETS: dict[str, str] = {
    key: f"{c.title} — {c.version}" + ("" if c.deidentified else " [비식별화판 없음]")
    for key, c in CORPORA.items()
}

# Sized so a full labelling set passes (71558 labels = 4.5 GB, the largest) while any
# single 원천데이터 shard (12–41 GB of audio) is refused until you raise it on purpose.
LARGE_MB_DEFAULT = 6_000.0

_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*(KB|MB|GB|TB)\s*$", re.IGNORECASE)
_UNIT_MB = {"KB": 1 / 1024, "MB": 1.0, "GB": 1024.0, "TB": 1024.0 * 1024}


@dataclass
class TreeEntry:
    """One downloadable file in an AI-Hub dataset file tree."""

    name: str
    size_mb: float
    filekey: str
    kind: str = "unknown"  # "label" | "source" | "unknown"
    split: str = "unknown"  # "train" | "valid" | "unknown"

    @property
    def size_human(self) -> str:
        return f"{self.size_mb / 1024:.1f} GB" if self.size_mb >= 1024 else f"{self.size_mb:.0f} MB"


@dataclass
class Tree:
    datasetkey: str
    entries: list[TreeEntry] = field(default_factory=list)

    def select(self, kind: str | None = None, split: str | None = None) -> list[TreeEntry]:
        out = self.entries
        if kind:
            out = [e for e in out if e.kind == kind]
        if split:
            out = [e for e in out if e.split == split]
        return out


# --------------------------------------------------------------------------- #
# aihubshell plumbing
# --------------------------------------------------------------------------- #
def _resolve_aihubshell(path: str | None = None) -> str:
    """Locate the ``aihubshell`` script (arg → $AIHUBSHELL_PATH → ./ → ~/ → $PATH)."""
    for candidate in (path, os.environ.get("AIHUBSHELL_PATH"), "./aihubshell", "~/aihubshell"):
        if not candidate:
            continue
        resolved = Path(candidate).expanduser()
        if resolved.is_file():
            return str(resolved.resolve())
    found = shutil.which("aihubshell")
    if found:
        return found
    raise FileNotFoundError(
        "aihubshell not found. Install it from https://aihub.or.kr (API 이용안내) and put "
        "it at ~/aihubshell, or pass --aihubshell_path."
    )


def _load_api_key() -> str | None:
    """Key from the environment, else ``AIHUB_API_KEY`` in ``.env.local``."""
    for var in ("AIHUB_API_KEY", "AIHUB_APIKEY"):
        if os.environ.get(var):
            return os.environ[var]
    if ENV_LOCAL.is_file():
        for line in ENV_LOCAL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() in ("AIHUB_API_KEY", "AIHUB_APIKEY"):
                return value.strip().strip("\"'")
    return None


def _run(args: list[str], cwd: Path | None = None, timeout: int = 1800) -> str:
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = result.stdout + result.stderr
    # aihubshell exits 0 even on auth failure, so surface the API's own message.
    for marker in ("인증실패", "승인 유효기간", "Download failed"):
        if marker in output:
            logger.error("AI-Hub rejected the request:\n%s", output.strip()[-600:])
            break
    return output


def _shell(
    *shell_args: str,
    aihubshell_path: str | None = None,
    with_key: bool = False,
    cwd: Path | None = None,
    timeout: int = 1800,
) -> str:
    args = ["bash", _resolve_aihubshell(aihubshell_path), *shell_args]
    if with_key:
        key = _load_api_key()
        if not key:
            raise RuntimeError(
                "No AI-Hub API key. Put `AIHUB_API_KEY=<your key>` in .env.local "
                "(gitignored) or export AIHUB_API_KEY."
            )
        args += ["-aihubapikey", key]
    return _run(args, cwd=cwd, timeout=timeout)


# --------------------------------------------------------------------------- #
# Tree parsing
# --------------------------------------------------------------------------- #
def _parse_size_mb(raw: str) -> float:
    m = _SIZE_RE.match(raw)
    if not m:
        return 0.0
    return float(m.group(1)) * _UNIT_MB[m.group(2).upper()]


def parse_tree(text: str, datasetkey: str = "") -> Tree:
    """Parse ``aihubshell -mode l -datasetkey K`` output into typed entries.

    File lines look like ``├─<name>.zip | 518 MB | 572524``; everything else is a
    directory line. Files inherit ``kind``/``split`` from the nearest preceding
    directory line that names a category (라벨링/원천) or a split (Training/Validation).
    """
    tree = Tree(datasetkey=datasetkey)
    kind, split = "unknown", "unknown"
    for line in text.splitlines():
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3 or not parts[-1].isdigit():
                continue
            name = re.sub(r"^[\s│├└─]+", "", parts[0]).strip()
            tree.entries.append(
                TreeEntry(
                    name=name,
                    size_mb=_parse_size_mb(parts[-2]),
                    filekey=parts[-1],
                    kind=kind,
                    split=split,
                )
            )
            continue
        # Directory line: update the inherited context.
        if "Training" in line:
            split = "train"
        elif "Validation" in line:
            split = "valid"
        if "라벨링" in line:
            kind = "label"
        elif "원천" in line:
            kind = "source"
    return tree


# --------------------------------------------------------------------------- #
# Local filesystem helpers
# --------------------------------------------------------------------------- #
def _resolve_dest(dest: str | Path | None, datasetkey: str) -> Path:
    """Absolute download destination: an explicit path, else the registry's version tree."""
    if dest is None:
        try:
            path = corpus(datasetkey).root
        except KeyError:
            path = REPO_ROOT / "data" / "aihub"
            logger.warning(
                "dataset %s is not in the registry; falling back to %s", datasetkey, path
            )
    else:
        path = Path(dest)
        if not path.is_absolute():
            path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _member_name(info: zipfile.ZipInfo) -> str:
    """A zip member's name, undoing the CP437 mojibake of Windows-authored archives.

    AI-Hub's zips are built on Windows without the UTF-8 flag, so ``zipfile`` decodes the
    Korean names as CP437. Left alone, a component like ``제주도`` becomes garbage — and
    ``prepare_data`` resolves both region and split by scanning path components for Korean
    markers, so the damage would surface as ``do="unknown"`` rows rather than an error.
    """
    if info.flag_bits & 0x800:  # bit 11: names are already UTF-8
        return info.filename
    try:
        raw = info.filename.encode("cp437")
    except UnicodeEncodeError:
        return info.filename
    # UTF-8 first: a zip written on Linux often omits the flag while still storing UTF-8,
    # and forcing CP949 on those bytes would corrupt names that were never broken.
    for encoding in ("utf-8", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return info.filename


def _member_relpath(info: zipfile.ZipInfo) -> Path:
    """A member's name as a path guaranteed to stay *inside* the destination.

    ``ZipFile.extractall`` does this sanitising for you; extracting member-by-member
    (which this module does, to fix the encoding above) means doing it by hand. The
    139-1 archives make it load-bearing: every member is stored as ``/talk_….json``,
    with a leading slash. ``out_dir / "/talk_….json"`` is ``/talk_….json`` — pathlib
    discards the left side on an absolute right side — so without this, extraction
    either escapes the tree or, once the escape is refused, silently writes nothing.
    """
    parts = [
        part
        for part in _member_name(info).replace("\\", "/").split("/")
        if part not in ("", ".", "..") and not part.endswith(":")  # drop drive letters
    ]
    return Path(*parts) if parts else Path()


def _extract_zip(zip_path: Path, force: bool = False) -> tuple[int, bool]:
    """Extract one archive next to itself. Returns ``(files_written, was_skipped)``.

    "Already extracted" means **the file count matches the archive**, not merely that the
    output directory is non-empty. An extraction interrupted partway — a killed session,
    a full disk — leaves a non-empty directory, and treating that as done would report
    success over a truncated corpus. Comparing counts costs one central-directory read
    plus a metadata walk, and self-heals directories extracted before this check existed.
    """
    out_dir = zip_path.with_suffix("")
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            info for info in zf.infolist() if not info.is_dir() and _member_relpath(info) != Path()
        ]
        if not force and out_dir.is_dir():
            on_disk = sum(1 for p in out_dir.rglob("*") if p.is_file())
            if on_disk == len(members):
                return 0, True
            if on_disk:
                logger.warning(
                    "%s: %d files on disk but archive holds %d — re-extracting "
                    "(previous run was interrupted)",
                    out_dir.name,
                    on_disk,
                    len(members),
                )

        out_dir.mkdir(parents=True, exist_ok=True)
        resolved_root = out_dir.resolve()
        written = 0
        for info in members:
            target = out_dir / _member_relpath(info)
            # Backstop only — _member_relpath already removed every escaping component.
            if not target.resolve().is_relative_to(resolved_root):
                logger.warning("skipping member escaping %s: %s", out_dir.name, info.filename)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1
    return written, False


def _version_roots(path: str | None, version: str | None) -> list[Path]:
    """The tree(s) a local-filesystem command should operate on."""
    if path:
        root = Path(path)
        return [root if root.is_absolute() else REPO_ROOT / root]
    versions = [version] if version else list(VERSIONS)
    return [tree_root(v) for v in versions]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def datasets(grep: str = "방언", aihubshell_path: str | None = None) -> None:
    """List AI-Hub datasets whose title matches ``grep`` (no API key needed)."""
    out = _shell("-mode", "l", aihubshell_path=aihubshell_path, timeout=300)
    hits = [line for line in out.splitlines() if grep in line]
    print(f"\n=== AI-Hub datasets matching {grep!r} ({len(hits)}) ===")
    for line in hits:
        print(" ", line.strip())
    print("\n=== known dialect keys ===")
    for key, desc in DIALECT_DATASETS.items():
        print(f"  {key:>6s}  {desc}")


def tree(datasetkey: str | int, aihubshell_path: str | None = None) -> None:
    """Print the raw file tree of one dataset (no API key needed)."""
    print(_shell("-mode", "l", "-datasetkey", str(datasetkey), aihubshell_path=aihubshell_path))


def plan(datasetkey: str | int, aihubshell_path: str | None = None) -> None:
    """Summarise a dataset: label vs audio, per-split totals, and every file key.

    Run this **before** any download — it is the only way to see that a single
    원천데이터 shard is 28 GB while the labels you actually need are 500 MB.
    """
    datasetkey = str(datasetkey)
    raw = _shell("-mode", "l", "-datasetkey", datasetkey, aihubshell_path=aihubshell_path)
    parsed = parse_tree(raw, datasetkey)
    if not parsed.entries:
        print("No files parsed — check the dataset key or the aihubshell output above.")
        return

    print(f"\n=== dataset {datasetkey}: {DIALECT_DATASETS.get(datasetkey, '')} ===")
    print(f"{len(parsed.entries)} files\n")
    for kind in ("label", "source", "unknown"):
        group = parsed.select(kind=kind)
        if not group:
            continue
        total = sum(e.size_mb for e in group)
        tag = {"label": "라벨링(텍스트)", "source": "원천(음성)", "unknown": "미분류"}[kind]
        print(f"--- {tag}: {len(group)} files, {total / 1024:.1f} GB total ---")
        for e in sorted(group, key=lambda x: x.size_mb):
            print(f"  filekey={e.filekey:>8s}  {e.size_human:>8s}  [{e.split:5s}]  {e.name}")
        print()

    labels = parsed.select(kind="label")
    if labels:
        smallest = min(labels, key=lambda e: e.size_mb)
        print(
            f"smallest label file: filekey={smallest.filekey} ({smallest.size_human}) "
            f"→ python scripts/aihub_fetch.py sample --datasetkey {datasetkey}"
        )


def download(
    datasetkey: str | int,
    filekey: str | None = None,
    kind: str | None = "label",
    split: str | None = None,
    dest: str | None = None,
    allow_large_mb: float = LARGE_MB_DEFAULT,
    dry_run: bool = False,
    aihubshell_path: str | None = None,
) -> None:
    """Download selected files. Needs an API key **and** a valid AI-Hub approval.

    Args:
        filekey: Explicit comma-separated file keys. Overrides ``kind``/``split``.
        kind: "label" (text, default) or "source" (audio — huge).
        split: Restrict to "train" or "valid".
        dest: Defaults to the registry's tree for this dataset's version — keeping
            old-format and new-format corpora in separate trees, which is what lets
            ``prepare_data.py`` pick the right parser for a whole directory.
        allow_large_mb: Refuse if the selection exceeds this many MB. Raise it
            deliberately to pull audio.
        dry_run: Resolve and print the selection without downloading.
    """
    datasetkey = str(datasetkey)
    raw = _shell("-mode", "l", "-datasetkey", datasetkey, aihubshell_path=aihubshell_path)
    parsed = parse_tree(raw, datasetkey)

    if filekey:
        wanted = {k.strip() for k in str(filekey).replace(",", " ").split() if k.strip()}
        chosen = [e for e in parsed.entries if e.filekey in wanted]
        missing = wanted - {e.filekey for e in chosen}
        if missing:
            logger.warning("file keys not present in this dataset's tree: %s", sorted(missing))
    else:
        chosen = parsed.select(kind=kind, split=split)

    if not chosen:
        print("Nothing selected. Run `plan` to see what's available.")
        return

    total = sum(e.size_mb for e in chosen)
    print(f"\nselected {len(chosen)} file(s), {total / 1024:.2f} GB:")
    for e in sorted(chosen, key=lambda x: x.size_mb):
        print(f"  filekey={e.filekey:>8s}  {e.size_human:>8s}  [{e.kind}/{e.split}]  {e.name}")

    if total > allow_large_mb:
        raise SystemExit(
            f"\nREFUSED: {total / 1024:.2f} GB exceeds allow_large_mb="
            f"{allow_large_mb / 1024:.2f} GB. Re-run with --allow_large_mb if intended."
        )
    if dry_run:
        print("\n(dry run — nothing downloaded)")
        return

    dest_path = _resolve_dest(dest, datasetkey)
    keys = ",".join(e.filekey for e in chosen)
    logger.info("downloading into %s ...", dest_path)
    print(
        _shell(
            "-mode",
            "d",
            "-datasetkey",
            datasetkey,
            "-filekey",
            keys,
            with_key=True,
            cwd=dest_path,
            aihubshell_path=aihubshell_path,
            timeout=21600,
        )
    )


def sample(
    datasetkey: str | int,
    dest: str = "data/aihub_samples",
    aihubshell_path: str | None = None,
) -> None:
    """Download only the single smallest labelling file — enough to verify the schema."""
    datasetkey = str(datasetkey)
    raw = _shell("-mode", "l", "-datasetkey", datasetkey, aihubshell_path=aihubshell_path)
    parsed = parse_tree(raw, datasetkey)
    labels = parsed.select(kind="label")
    if not labels:
        print("No labelling files found — run `plan` and pick a file key manually.")
        return
    smallest = min(labels, key=lambda e: e.size_mb)
    logger.info("smallest label file: %s (%s)", smallest.name, smallest.size_human)
    download(
        datasetkey,
        filekey=smallest.filekey,
        dest=f"{dest}/{datasetkey}",
        aihubshell_path=aihubshell_path,
    )


def inspect(path: str = "data/aihub_samples", max_files: int = 2, unzip: bool = True) -> None:
    """Print the JSON schema of a downloaded sample: keys, one sentence, prosody fields.

    This is the gate before trusting ``prepare_data.py`` on a *new* dataset: 139-2
    (71558) is assumed to share 139-1's schema, and this is how that gets verified
    rather than believed.
    """
    root = (REPO_ROOT / path) if not Path(path).is_absolute() else Path(path)
    if not root.exists():
        raise SystemExit(f"{root} does not exist — download a sample first.")

    if unzip:
        for zip_path in sorted(root.rglob("*.zip"))[:max_files]:
            out_dir = zip_path.with_suffix("")
            if not out_dir.exists():
                logger.info("unzipping %s", zip_path.name)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(out_dir)

    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        raise SystemExit(f"No .json under {root}.")
    print(f"{len(json_files)} JSON files under {root}\n")

    for jf in json_files[:max_files]:
        print("=" * 78)
        print(jf.relative_to(root))
        print("=" * 78)
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — malformed AI-Hub JSON is expected
            print(f"  !! unparseable ({exc}) — prepare_data.clean_json handles these")
            continue
        print("  top-level keys:", sorted(data)[:20])
        transcription = data.get("transcription", {})
        print("  transcription keys:", sorted(transcription)[:20])
        sentences = transcription.get("sentences", [])
        segments = transcription.get("segments", [])
        print(f"  sentences={len(sentences)} segments={len(segments)}")
        if sentences:
            s = sentences[0]
            print("  sentence[0] keys:", sorted(s))
            for k in ("standard_form", "dialect_form", "startTime", "endTime"):
                if k in s:
                    print(f"    {k}: {str(s[k])[:120]}")
            tones = s.get("intonations")
            if tones is None:
                print("    !! no `intonations` — prosody markers would be unavailable")
            else:
                print(f"    intonations: n={len(tones)} head={tones[:8]}")
        if segments:
            print("  segment[0] keys:", sorted(segments[0]))


def extract(
    path: str | None = None,
    version: str | None = None,
    force: bool = False,
) -> None:
    """Unzip every downloaded archive under a tree, in place and idempotently.

    ``aihubshell`` leaves ``.zip`` files behind; nothing else in this repo unzips them in
    bulk (``inspect`` opens at most two, to eyeball a schema), so a full download used to
    need a manual unzip step. Archives are extracted next to themselves and left on disk,
    which makes a re-extract free if a parse turns out wrong.

    Args:
        path: An explicit tree. Defaults to every registry version tree that exists.
        version: Restrict to one corpus version ("v1_2020" / "v2_2022").
        force: Re-extract even where an output directory already has content.
    """
    roots = [r for r in _version_roots(path, version) if r.exists()]
    if not roots:
        raise SystemExit("Nothing to extract — download first (`aihub_fetch.py fetch`).")

    total_written = total_skipped = 0
    for root in roots:
        zips = sorted(root.rglob("*.zip"))
        print(f"\n=== {root} — {len(zips)} archive(s) ===")
        for zip_path in zips:
            written, skipped = _extract_zip(zip_path, force=force)
            if skipped:
                total_skipped += 1
                print(f"  skip   {zip_path.name}  (already extracted)")
            else:
                total_written += written
                print(f"  ok     {zip_path.name}  → {written} files")

    print(f"\nextracted {total_written} files; skipped {total_skipped} archive(s)")
    for root in roots:
        print(f"  {root}: {len(list(root.rglob('*.json')))} JSON files on disk")


def fetch(
    version: str = "all",
    dry_run: bool = False,
    do_extract: bool = True,
    allow_large_mb: float = LARGE_MB_DEFAULT,
    aihubshell_path: str | None = None,
) -> None:
    """Download **every** labelling file of a corpus generation, then extract it.

    The destination for each dataset comes from the registry, so old-format and
    new-format corpora cannot end up in one tree — which matters because
    ``prepare_data.py`` chooses its parser per tree, and the wrong choice fails silently
    (``do="unknown"`` rows that stage0 then drops).

    Args:
        version: "v1_2020", "v2_2022", or "all".
        dry_run: Print the per-dataset selection without downloading.
        do_extract: Unzip after downloading (skip to defer the disk cost).
    """
    versions = list(VERSIONS) if version == "all" else [version]
    selected = [c for v in versions for c in corpora_for(version=v)]
    if not selected:
        raise SystemExit(f"No corpora for version={version!r}; known: {list(VERSIONS)}")

    print(f"\n=== fetching labels for {len(selected)} dataset(s) ===")
    for c in selected:
        print(f"  {c.datasetkey:>6s}  {c.label_gb:>4.1f} GB  → {c.root}  ({c.title})")

    for c in selected:
        print(f"\n----- {c.datasetkey}: {c.title} -----")
        download(
            c.datasetkey,
            kind="label",
            allow_large_mb=allow_large_mb,
            dry_run=dry_run,
            aihubshell_path=aihubshell_path,
        )

    if dry_run or not do_extract:
        return
    for v in versions:
        extract(version=v)


def diff_trees(
    a: str,
    b: str,
    pattern: str = "*.json",
    show: int = 15,
) -> None:
    """Compare two extracted copies of the same corpus by content hash.

    Written for the one question a re-download has to answer before the older copy can be
    deleted: *is the new copy the same data?* Files are keyed by **name**, not by relative
    path, because two copies of one AI-Hub dataset routinely differ in directory layout
    (bundle subdirectories, numbered split prefixes) while carrying identical files.

    Exits non-zero when the trees differ, so it can gate a deletion in a shell script.
    """
    root_a, root_b = (Path(p) if Path(p).is_absolute() else REPO_ROOT / p for p in (a, b))
    for root in (root_a, root_b):
        if not root.exists():
            raise SystemExit(f"{root} does not exist.")

    manifests: list[dict[str, Path]] = []
    for root in (root_a, root_b):
        by_name: dict[str, Path] = {}
        collisions = 0
        for file in root.rglob(pattern):
            if file.name in by_name:
                collisions += 1
            by_name[file.name] = file
        if collisions:
            logger.warning(
                "%s: %d duplicate file names — name-keyed comparison hides those copies",
                root,
                collisions,
            )
        print(f"{root}: {len(by_name)} unique {pattern} names")
        manifests.append(by_name)

    names_a, names_b = (set(m) for m in manifests)
    only_a, only_b = sorted(names_a - names_b), sorted(names_b - names_a)
    shared = sorted(names_a & names_b)

    logger.info("hashing %d shared files on both sides ...", len(shared))
    differing = []
    for i, name in enumerate(shared, 1):
        if _sha256(manifests[0][name]) != _sha256(manifests[1][name]):
            differing.append(name)
        if i % 20_000 == 0:
            logger.info("  %d/%d hashed", i, len(shared))

    print(f"\n=== diff {root_a.name} ↔ {root_b.name} ===")
    print(f"  shared names   : {len(shared)}")
    print(f"  only in A      : {len(only_a)}")
    print(f"  only in B      : {len(only_b)}")
    print(f"  content differs: {len(differing)}")
    for label, items in (("only in A", only_a), ("only in B", only_b), ("differs", differing)):
        for name in items[:show]:
            print(f"    [{label}] {name}")
        if len(items) > show:
            print(f"    ... and {len(items) - show} more")

    if only_a or only_b or differing:
        raise SystemExit("\nTREES DIFFER — do not delete either copy on this result.")
    print("\nIDENTICAL — every file matches by content.")


if __name__ == "__main__":
    fire.Fire(
        {
            "datasets": datasets,
            "tree": tree,
            "plan": plan,
            "fetch": fetch,
            "download": download,
            "extract": extract,
            "sample": sample,
            "inspect": inspect,
            "diff_trees": diff_trees,
        }
    )
