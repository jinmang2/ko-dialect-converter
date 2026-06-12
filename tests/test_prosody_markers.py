from __future__ import annotations

from ko_dialect.data.prosody import (
    PROSODY_MARKER_POLICY,
    add_sentence_final_marker,
    strip_markers,
)


def test_strip_markers_removes_all_marker_types():
    for marker in PROSODY_MARKER_POLICY["markers"]:
        assert strip_markers(f"문장{marker}") == "문장"


def test_strip_markers_tidies_whitespace_and_punctuation():
    assert strip_markers("밥 뭇나<UP>?") == "밥 뭇나?"
    assert strip_markers("어데 가노<DOWN>") == "어데 가노"
    assert strip_markers("보통 말<KEEP> 함더<WAVE>") == "보통 말 함더"


def test_strip_markers_noop_on_clean_text():
    assert strip_markers("no markers here") == "no markers here"
    assert strip_markers("") == ""


def test_add_then_strip_roundtrips():
    # adding a marker then stripping returns the original (modulo strip())
    original = "밥 먹었니?"
    marked = add_sentence_final_marker(original, "<UP>")
    assert marked == "밥 먹었니<UP>?"
    assert strip_markers(marked) == original
