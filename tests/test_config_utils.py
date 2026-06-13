from __future__ import annotations

from dataclasses import dataclass

import pytest
from omegaconf import OmegaConf

from ko_dialect.config_utils import from_omegaconf


@dataclass
class _Cfg:
    a: int = 1
    b: str = "x"
    c: float = 0.5


def test_merges_section_and_keeps_defaults():
    section = OmegaConf.create({"a": 7})
    cfg = from_omegaconf(_Cfg, section)
    assert cfg.a == 7
    assert cfg.b == "x"  # default preserved


def test_ignores_unknown_keys_from_section():
    section = OmegaConf.create({"a": 2, "unknown": 99})
    cfg = from_omegaconf(_Cfg, section)
    assert cfg.a == 2


def test_later_sections_and_overrides_win():
    s1 = OmegaConf.create({"a": 1, "b": "first"})
    s2 = OmegaConf.create({"b": "second"})
    cfg = from_omegaconf(_Cfg, s1, s2, c=9.0)
    assert cfg.b == "second"
    assert cfg.c == 9.0


def test_unknown_override_raises():
    with pytest.raises(TypeError, match="no field"):
        from_omegaconf(_Cfg, nonexistent=1)


def test_none_sections_skipped():
    cfg = from_omegaconf(_Cfg, None, {"a": 3})
    assert cfg.a == 3
