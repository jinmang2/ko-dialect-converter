from __future__ import annotations

from ko_dialect.data.template import DO_NAME, ChatTemplate


def test_do_name_mapping():
    assert DO_NAME["gangwondo"] == "강원도"
    assert DO_NAME["gyeongsangdo"] == "경상도"


def test_format_user_std2dia(mock_tokenizer):
    t = ChatTemplate()
    result = t.format_user("안녕하세요", "gangwondo", "std2dia")
    assert "강원도" in result
    assert "안녕하세요" in result
    assert "사투리" in result


def test_format_user_dia2std(mock_tokenizer):
    t = ChatTemplate()
    result = t.format_user("안녕하세유", "gyeongsangdo", "dia2std")
    assert "경상도" in result
    assert "안녕하세유" in result
    assert "표준어" in result


def test_format_messages_structure(mock_tokenizer):
    t = ChatTemplate()
    msgs = t.format_messages("hello", "반갑다", "gangwondo", "std2dia")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant"]
    assert msgs[2]["content"] == "반갑다"


def test_apply_returns_string(mock_tokenizer):
    t = ChatTemplate()
    result = t.apply(mock_tokenizer, "hello", "반갑다", "gangwondo", "std2dia")
    assert isinstance(result, str)
    assert "[system]" in result
    assert "[assistant]" in result


def test_build_prompt_has_generation_prompt(mock_tokenizer):
    t = ChatTemplate()
    prompt = t.build_prompt(mock_tokenizer, "hello", "gangwondo", "std2dia")
    assert "[assistant]" in prompt
    assert "[user]" in prompt
