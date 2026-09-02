"""Pin the corpus registry's invariants — the ones whose violation fails *silently*.

`ko_dialect.corpora` exists so a download destination is derived rather than typed, and
so one directory tree holds exactly one parser format. Both properties are load-bearing:
`prepare_data.py` picks its parser per tree, and a mismatched parser yields `do="unknown"`
rows that stage0's `SUPPORTED_DO` filter silently drops (DATA_EXPANSION_RUNBOOK §1.4 B2).
Nothing raises; a corpus just disappears. So these are tests, not comments.
"""

from __future__ import annotations

import pytest

from ko_dialect import corpora
from ko_dialect.corpora import CORPORA, V1_2020, V2_2022, VERSIONS


def test_every_key_matches_its_entry():
    # A dict key that disagrees with the entry's datasetkey would send `download` to
    # AI-Hub with one key and file it on disk under another.
    for key, entry in CORPORA.items():
        assert key == entry.datasetkey


def test_registry_covers_the_seven_dialect_datasets():
    assert set(CORPORA) == {"118", "119", "120", "121", "122", "71517", "71558"}


def test_regions_are_canonical_classifier_labels():
    # The registry duplicates region names as plain strings to stay torch-free; this is
    # the check that keeps that duplication from drifting away from labels.py.
    from ko_dialect.data.labels import SUPPORTED_DO

    for entry in CORPORA.values():
        assert set(entry.regions) <= SUPPORTED_DO, entry.datasetkey


def test_both_generations_together_cover_all_five_regions():
    assert len(corpora.regions_for()) == 5
    # v2 alone also reaches five: 139-1 (강원·경상) + 139-2 (충청·전라·제주).
    assert len(corpora.regions_for(V2_2022)) == 5


def test_each_version_tree_has_exactly_one_parser_format():
    assert corpora.uses_old_format(V1_2020) is True
    assert corpora.uses_old_format(V2_2022) is False


def test_uses_old_format_rejects_a_mixed_tree(monkeypatch):
    mixed = dict(CORPORA)
    mixed["118"] = CORPORA["118"].__class__(**{**CORPORA["118"].__dict__, "old_format": False})
    monkeypatch.setattr(corpora, "CORPORA", mixed)
    # Silently picking one format here is what the registry exists to prevent.
    with pytest.raises(ValueError, match="mixed parser formats"):
        corpora.uses_old_format(V1_2020)


def test_only_71517_lacks_a_deidentified_reupload():
    # This is why the local 139-1 copy is the current release rather than a stale one.
    not_deidentified = [c.datasetkey for c in CORPORA.values() if not c.deidentified]
    assert not_deidentified == ["71517"]


def test_corpora_for_filters_by_version_and_format():
    assert [c.datasetkey for c in corpora.corpora_for(version=V2_2022)] == ["71517", "71558"]
    assert len(corpora.corpora_for(old_format=True)) == 5
    assert len(corpora.corpora_for()) == 7


def test_corpus_lookup_accepts_int_and_reports_unknown_keys():
    assert corpora.corpus(71558).datasetkey == "71558"
    with pytest.raises(KeyError, match="unknown AI-Hub dataset key"):
        corpora.corpus("99999")


def test_version_directories_are_distinct_and_named_after_the_version():
    roots = {v: corpora.tree_root(v) for v in VERSIONS}
    assert len(set(roots.values())) == len(VERSIONS)
    for version, root in roots.items():
        assert root.name == version


def test_corpus_root_agrees_with_tree_root():
    for entry in CORPORA.values():
        assert entry.root == corpora.tree_root(entry.version)


def test_unknown_version_is_rejected():
    with pytest.raises(ValueError, match="unknown corpus version"):
        corpora.tree_root("v3_2099")
    with pytest.raises(ValueError, match="unknown corpus version"):
        corpora.corpora_for(version="v3_2099")


def test_data_root_env_override_applies_at_call_time(monkeypatch, tmp_path):
    # Call-time resolution is what lets a test (or a second disk) redirect the tree
    # without reimporting the module.
    monkeypatch.setenv("KO_DIALECT_DATA_ROOT", str(tmp_path))
    assert corpora.data_root() == tmp_path
    assert corpora.tree_root(V1_2020) == tmp_path / "aihub" / V1_2020

    monkeypatch.delenv("KO_DIALECT_DATA_ROOT")
    assert corpora.data_root() == corpora.REPO_ROOT / "data"
