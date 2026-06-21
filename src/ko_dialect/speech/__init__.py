"""Speech module: dialect audio -> standard Korean text (+ real-audio prosody).

Lightweight by design — importing this package pulls in NO heavy ML deps and downloads no
weights. faster-whisper / torch / PESTO are imported lazily inside the relevant methods.
See ``docs/SPEECH_MODULE_DESIGN.md`` for the full design.
"""

from __future__ import annotations

from .asr import DialectStandardizerASR
from .config import SpeechConfig
from .prosody_audio import AudioProsodyExtractor

__all__ = [
    "SpeechConfig",
    "DialectStandardizerASR",
    "AudioProsodyExtractor",
]
