"""Real-audio F0 extraction for prosody markers.

Lets us extract F0 directly from speech (PESTO default, FCPE alternative) and reuse the
EXISTING marker rules in ``ko_dialect.data.prosody`` to re-derive ``<UP>/<DOWN>/<KEEP>``
markers — so the JSON-metadata-derived markers can be validated against real pitch.

The F0 model (PESTO/torch) is imported lazily inside :meth:`load`/:meth:`extract_f0`;
``markers_from_f0`` is pure-python and works on any F0 series without the model.
See ``docs/SPEECH_MODULE_DESIGN.md`` §5.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .config import SpeechConfig

if TYPE_CHECKING:
    import numpy as np


class AudioProsodyExtractor:
    """Extract an F0 contour from audio and map it to a prosody marker.

    Construction and import are side-effect free; :meth:`load` performs the heavy import.
    """

    def __init__(self, config: SpeechConfig | None = None) -> None:
        self.config = config or SpeechConfig()
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Lazily load the configured F0 extractor (``pesto`` default)."""
        extractor = self.config.prosody_extractor
        if extractor == "pesto":
            import pesto  # lazy

            self._model = pesto
        elif extractor == "fcpe":
            import torchfcpe  # lazy

            self._model = torchfcpe.spawn_bundled_infer_model()
        else:
            raise ValueError(
                f"Unknown prosody_extractor={extractor!r} (expected 'pesto' | 'fcpe')."
            )

    def extract_f0(self, audio: np.ndarray, sample_rate: int | None = None) -> list[float]:
        """Return the F0 (Hz) contour for ``audio`` at ``sample_rate``. Requires :meth:`load` first.

        Contract only — the concrete PESTO/FCPE call is wired in P4 (see design doc §5/§8) once
        audio lands; ``markers_from_f0`` already provides the model-free rule layer.
        """
        if self._model is None:
            raise RuntimeError("F0 model not loaded — call load() first.")
        raise NotImplementedError("F0 inference is wired in P4; use markers_from_f0 for the rule layer.")

    @staticmethod
    def markers_from_f0(f0_series: Sequence[float]) -> str | None:
        """Map an F0 (Hz) contour to a prosody marker using the existing rule set.

        Uses only the pure-python rule functions (no F0 model), so the real-audio path produces
        markers identical in semantics to the JSON-derived ones. Note: the FIRST call imports
        ``ko_dialect.data``, whose package init pulls torch (via ``collator``); that cost is kept
        out of ``import ko_dialect.speech`` on purpose (deferred to call time), but it is not
        torch-free at call time.
        """
        # Deferred import: keeps `import ko_dialect.speech` light. ko_dialect.data.__init__ pulls
        # torch/datasets, so this lands torch in sys.modules at call time (not at module load).
        from ko_dialect.data.prosody import prosody_marker, summarize_intonation

        prosody = summarize_intonation(list(f0_series))
        return prosody_marker(prosody)
