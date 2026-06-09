"""Helpers to build dataclass configs from Hydra/OmegaConf sections.

Replaces the per-field copying each stage script used to do (3× boilerplate, and the
source of silent-drop bugs when a YAML knob had no matching constructor argument).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

T = TypeVar("T")


def from_omegaconf(cls: type[T], *sections: DictConfig | dict | None, **overrides: Any) -> T:
    """Construct dataclass ``cls`` from one or more OmegaConf/dict sections.

    Later sections and explicit ``overrides`` win. Keys absent from the dataclass are
    dropped with a debug log (so an unrelated section like ``logger`` can be passed
    wholesale). Keys present in the dataclass but absent from the sections keep their
    dataclass defaults — nothing is silently required.
    """
    field_names = {f.name for f in dataclasses.fields(cls)}
    merged: dict[str, Any] = {}
    for section in sections:
        if section is None:
            continue
        data = OmegaConf.to_container(section, resolve=True) if isinstance(section, DictConfig) else dict(section)
        for key, value in data.items():
            if key in field_names:
                merged[key] = value
            else:
                logger.debug("from_omegaconf(%s): ignoring unknown key %r", cls.__name__, key)
    for key, value in overrides.items():
        if key not in field_names:
            raise TypeError(f"{cls.__name__} has no field {key!r} (override).")
        merged[key] = value
    return cls(**merged)
