"""Dialect-audio -> standard-text ASR wrapper (direct standardization).

Thin, lazy interface over faster-whisper. Heavy deps (``faster_whisper``, ``torch``) are
imported *inside* methods so ``import ko_dialect.speech`` is cheap and never downloads a
model. Training (Whisper-turbo + LoRA) and the audio data pipeline live in ``scripts/``
(see ``docs/SPEECH_MODULE_DESIGN.md``); this class is the inference/serving seam.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import SpeechConfig

if TYPE_CHECKING:  # only for type checkers; never imported at runtime/import time
    import numpy as np


class DialectStandardizerASR:
    """Transcribe dialect speech directly into standard Korean text.

    Usage::

        asr = DialectStandardizerASR(SpeechConfig())
        asr.load()                      # downloads/loads weights (network, GPU)
        text = asr.transcribe("utt.wav")

    Until :meth:`load` is called, :meth:`transcribe` raises — construction and import are
    side-effect free.
    """

    def __init__(self, config: SpeechConfig | None = None) -> None:
        self.config = config or SpeechConfig()
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Lazily construct the faster-whisper model (imports + weight download happen here)."""
        from faster_whisper import WhisperModel  # lazy: keep module import cheap

        self._model = WhisperModel(
            self.config.asr_model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def transcribe(self, audio: str | np.ndarray) -> str:
        """Return the model's text for ``audio`` (path or 16kHz mono float array)."""
        if self._model is None:
            raise RuntimeError(
                "ASR model not loaded — call load() first (downloads/loads weights)."
            )
        segments, _info = self._model.transcribe(audio, language=self.config.language)
        return "".join(seg.text for seg in segments).strip()

    def transcribe_to_standard(self, audio: str | np.ndarray) -> str:
        """Semantic alias: with a ``direction='dialect2standard'`` model this yields standard text.

        Guards against accidentally serving a pure-ASR (``dialect2dialect``) checkpoint as a
        standardizer.
        """
        if self.config.direction != "dialect2standard":
            raise ValueError(
                "transcribe_to_standard requires direction='dialect2standard', "
                f"got {self.config.direction!r}."
            )
        return self.transcribe(audio)
