"""Tests for old_dialect (AI-Hub 한국어 방언 발화) transcript cleaning.

The old corpus stores AI-Hub annotation conventions inside ``standard_form`` /
``dialect_form``. The refactor that produced ``scripts/prepare_data.py`` carried over
the structural parsing but dropped the text-normalization step, so residual tags
((( )), &PII&, {non-verbal}, residual (dia)/(std) dual transcription, #interjection)
leaked into the training pairs. These tests pin the recovered ``clean_old_transcript``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PREPARE_DATA_PATH = Path(__file__).parents[1] / "scripts" / "prepare_data.py"
_SPEC = importlib.util.spec_from_file_location("prepare_data", _PREPARE_DATA_PATH)
assert _SPEC is not None
prepare_data = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(prepare_data)

clean_old_transcript = prepare_data.clean_old_transcript


def test_residual_dual_transcription_resolves_per_side():
    raw = "혹시 어떤 방법이 좋은지 (쫌)/(조금) 알려주실 수 있으세요?"
    assert (
        clean_old_transcript(raw, "standard")
        == "혹시 어떤 방법이 좋은지 조금 알려주실 수 있으세요?"
    )
    assert (
        clean_old_transcript(raw, "dialect") == "혹시 어떤 방법이 좋은지 쫌 알려주실 수 있으세요?"
    )


def test_multiple_dual_transcriptions_in_one_utterance():
    raw = "그것도 (쫌)/(조금) 군살이 (쫌)/(조금) 많이 붙어가지구"
    assert clean_old_transcript(raw, "standard") == "그것도 조금 군살이 조금 많이 붙어가지구"
    assert clean_old_transcript(raw, "dialect") == "그것도 쫌 군살이 쫌 많이 붙어가지구"


def test_dual_after_pii_token():
    raw = "&company-name2&(허고)/(하고) 하여튼 두 군데는"
    assert clean_old_transcript(raw, "standard") == "하고 하여튼 두 군데는"
    assert clean_old_transcript(raw, "dialect") == "허고 하여튼 두 군데는"


def test_unintelligible_double_paren_removed():
    assert clean_old_transcript("학교 다녀야 돼 (())", "standard") == "학교 다녀야 돼"
    assert clean_old_transcript("더 일하는(()) 분들이 와", "dialect") == "더 일하는 분들이 와"


def test_pii_anonymization_token_removed():
    # Only the &...& token is removed; a dangling josa (랑) is left as-is — it appears
    # identically on both sides, so it does not bias the dialect transform.
    raw = "&company-name&랑 그~ 신세계 백화점 지하 음식코너 같은 데서"
    assert clean_old_transcript(raw, "standard") == "랑 그~ 신세계 백화점 지하 음식코너 같은 데서"


def test_non_verbal_brace_removed():
    raw = "엄청 귀엽잖아. {laughing}"
    assert clean_old_transcript(raw, "dialect") == "엄청 귀엽잖아."


def test_interjection_hash_marker_stripped_keeps_word():
    assert clean_old_transcript("#오메 너도 늙었다", "standard") == "오메 너도 늙었다"
    assert clean_old_transcript("#아따 진짜 나", "dialect") == "아따 진짜 나"


def test_filler_lengthening_tilde_is_preserved():
    # '~' is prosodic lengthening present identically on both sides — not a tag.
    raw = "어~ 작년에는 그래도"
    assert clean_old_transcript(raw, "standard") == "어~ 작년에는 그래도"


def test_whitespace_is_collapsed_and_stripped():
    raw = " 학교 다녀야 돼  (())   분들이 "
    assert clean_old_transcript(raw, "standard") == "학교 다녀야 돼 분들이"


def test_clean_to_empty_returns_empty_string():
    assert clean_old_transcript("(())", "standard") == ""
    assert clean_old_transcript("&name& {laughing}", "dialect") == ""


def test_orphan_bracket_from_cross_utterance_span_is_stripped():
    # Dual-transcription parens sometimes straddle the utterance boundary, leaving a
    # lone ( or ) whose partner lives in another utterance — drop the stray char.
    assert (
        clean_old_transcript("달걀만 갖고는 쫌) 부족(하드라고", "dialect")
        == "달걀만 갖고는 쫌 부족하드라고"
    )


def test_clean_handles_none_and_blank():
    assert clean_old_transcript(None, "standard") == ""
    assert clean_old_transcript("", "dialect") == ""
