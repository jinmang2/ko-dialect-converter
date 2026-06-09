from __future__ import annotations

import pytest

from ko_dialect.data.template import (
    TEMPLATE_REGISTRY,
    ChatTemplate,
    get_template,
    register_template,
)


def test_default_template_is_chat_template():
    t = get_template("default")
    assert isinstance(t, ChatTemplate)


def test_prosody_template_has_distinct_system_prompt():
    default = get_template("default")
    prosody = get_template("prosody")
    assert prosody.system != default.system
    assert "억양" in prosody.system


def test_get_template_unknown_raises_with_listing():
    with pytest.raises(ValueError, match="Unknown template"):
        get_template("does-not-exist")


def test_register_template_roundtrip():
    custom = ChatTemplate(system="테스트 시스템 프롬프트")
    register_template("unit-test-tmpl", custom)
    try:
        assert get_template("unit-test-tmpl") is custom
    finally:
        TEMPLATE_REGISTRY.pop("unit-test-tmpl", None)
