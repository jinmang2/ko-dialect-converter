from __future__ import annotations

import re
from typing import Any

PROSODY_MARKER_POLICY_VERSION = 2

# v1 — the original scheme that produced the first prosody A/B (docs/EXPERIMENTS §2).
# Kept verbatim so that result stays reproducible (``prosody_marker(p, version=1)``).
# Two problems motivated v2:
#   1. Korean F0 declines naturally over an utterance (sampled f0_delta median ≈ −0.21),
#      so a symmetric ±0.15 delta band tagged ~60% of rows <DOWN> — it confused ordinary
#      declination with a falling boundary tone.
#   2. <WAVE> keyed on ``f0_range``, which the stored sentence summary does not carry, so
#      <WAVE> could never fire (a dead class).
PROSODY_MARKER_POLICY_V1: dict[str, Any] = {
    "version": 1,
    "markers": ["<UP>", "<DOWN>", "<WAVE>", "<KEEP>"],
    "valid_f0_min_hz": 30.0,
    "min_valid_points": 3,
    "up_threshold_delta_ratio": 0.15,
    "down_threshold_delta_ratio": -0.15,
    "wave_threshold_range_hz": 50.0,
    "placement": "before sentence-final punctuation, otherwise append to text",
}

# v2 — K-ToBI-grounded, declination-aware. Markers map to Korean Intonation-Phrase
# boundary tones (Jun, S.-A. 2000, *K-ToBI Labelling Conventions*; Jun 2005, *Prosodic
# Typology*): an IP ends in a boundary tone (%):
#   <UP>   ≈ H%            rising boundary tone (interrogative / continuation)
#   <DOWN> ≈ L%            steep fall *beyond* natural declination (emphatic / final)
#   <WAVE> ≈ LH%/HL%/LHL%  complex contour — high F0 dispersion (coefficient of variation)
#   <KEEP> ≈ M% / level    sustained/level tone (incl. the Gyeongsang "끝까지 유지" pattern)
# Thresholds tuned on the 614k-row F0 summary so all four classes are non-degenerate
# (UP≈19% DOWN≈23% WAVE≈6% KEEP≈52%); <WAVE> keys on f0_std/f0_mean (always present)
# rather than the absent f0_range.
PROSODY_MARKER_POLICY: dict[str, Any] = {
    "version": PROSODY_MARKER_POLICY_VERSION,
    "markers": ["<UP>", "<DOWN>", "<WAVE>", "<KEEP>"],
    "scheme": "K-ToBI IP boundary tones (Jun 2000): UP=H%, DOWN=L%, WAVE=contour, KEEP=level",
    "valid_f0_min_hz": 30.0,
    "min_valid_points": 3,
    "up_threshold_delta_ratio": 0.05,
    "down_threshold_delta_ratio": -0.40,
    "wave_threshold_cv": 0.30,
    "declination_aware": "thresholds offset for natural F0 declination (sampled median delta ≈ -0.21)",
    "placement": "before sentence-final punctuation, otherwise append to text",
    "reference": "Jun (2000) K-ToBI; Jun (2005) Prosodic Typology of Korean",
}


def summarize_intonation(intonations: list[float]) -> dict | None:
    """raw F0 시계열 -> 요약 통계 (None 값/0 값 제외)"""
    valid = [p for p in intonations if p and p > PROSODY_MARKER_POLICY["valid_f0_min_hz"]]
    n = len(valid)
    if len(valid) < PROSODY_MARKER_POLICY["min_valid_points"]:
        return None

    mean = sum(valid) / n
    var = sum((p - mean) ** 2 for p in valid) / n

    return {
        "f0_mean": round(mean, 2),
        "f0_std": round(var**0.5, 2),
        "f0_start": round(valid[0], 2),
        "f0_end": round(valid[-1], 2),
        "f0_delta": round((valid[-1] - valid[0]) / valid[0], 3) if valid[0] > 0 else 0,
        "f0_range": round(max(valid) - min(valid), 2),
    }


def f0_coefficient_of_variation(prosody: dict) -> float:
    """F0 dispersion = std/mean — the contour-strength signal for <WAVE>.

    Uses ``f0_std``/``f0_mean`` (always present in the stored summary). Falls back to
    ``f0_range``/``f0_mean`` only if a re-parsed summary carries f0_range but not f0_std.
    """
    mean = prosody.get("f0_mean") or 0.0
    if not mean:
        return 0.0
    std = prosody.get("f0_std")
    if std is None:
        rng = prosody.get("f0_range")
        return (rng / mean) if rng is not None else 0.0
    return std / mean


def _prosody_marker_v1(prosody: dict) -> str:
    policy = PROSODY_MARKER_POLICY_V1
    delta = prosody.get("f0_delta", 0.0)
    if delta >= policy["up_threshold_delta_ratio"]:
        return "<UP>"
    if delta <= policy["down_threshold_delta_ratio"]:
        return "<DOWN>"
    if prosody.get("f0_range", 0.0) >= policy["wave_threshold_range_hz"]:
        return "<WAVE>"
    return "<KEEP>"


def prosody_marker(
    prosody: dict | None, *, version: int = PROSODY_MARKER_POLICY_VERSION
) -> str | None:
    """Map a sentence-level F0 summary to a K-ToBI-grounded boundary-tone marker.

    v2 (default) is declination-aware and keys <WAVE> on the coefficient of variation;
    pass ``version=1`` to reproduce the original prosody A/B scheme.
    """
    if not prosody:
        return None
    if version == 1:
        return _prosody_marker_v1(prosody)

    policy = PROSODY_MARKER_POLICY
    delta = prosody.get("f0_delta", 0.0)
    if delta >= policy["up_threshold_delta_ratio"]:
        return "<UP>"
    if delta <= policy["down_threshold_delta_ratio"]:
        return "<DOWN>"
    if f0_coefficient_of_variation(prosody) >= policy["wave_threshold_cv"]:
        return "<WAVE>"
    return "<KEEP>"


def slice_intonation_by_time(
    intonations: list[float],
    sent_start_s: float,
    sent_end_s: float,
    seg_start_s: float,
    seg_end_s: float,
) -> list[float]:
    """Return the F0 sub-series covering ``[seg_start_s, seg_end_s]`` within a sentence.

    The AI-Hub data stores one flat F0 series per *sentence* (uniformly sampled across the
    sentence span); word-level segments carry only ``startTime``/``endTime``. Per-eojeol F0
    is therefore recovered by slicing the sentence series at the segment's time window —
    the basis for per-eojeol (Phase-2) prosody markers.
    """
    n = len(intonations)
    dur = sent_end_s - sent_start_s
    if n == 0 or dur <= 0:
        return []
    fps = n / dur
    i0 = max(0, int(round((seg_start_s - sent_start_s) * fps)))
    i1 = min(n, int(round((seg_end_s - sent_start_s) * fps)))
    return intonations[i0:i1] if i1 > i0 else []


def eojeol_prosody_markers(
    intonations: list[float],
    sent_start_s: float,
    sent_end_s: float,
    eojeols: list[dict],
    *,
    version: int = PROSODY_MARKER_POLICY_VERSION,
) -> list[dict]:
    """Per-eojeol F0 markers for a sentence.

    ``eojeols`` is a list of ``{"word": str, "start_s": float, "end_s": float}``. Returns one
    ``{"word", "marker", "prosody"}`` per eojeol; ``marker`` is None when the eojeol's slice
    has too few valid F0 points to summarise.
    """
    out: list[dict] = []
    for ej in eojeols:
        sub = slice_intonation_by_time(
            intonations, sent_start_s, sent_end_s, ej["start_s"], ej["end_s"]
        )
        summary = summarize_intonation(sub)
        out.append(
            {
                "word": ej["word"],
                "marker": prosody_marker(summary, version=version),
                "prosody": summary,
            }
        )
    return out


def apply_eojeol_markers(marked: list[dict]) -> str:
    """Render per-eojeol markers as an inline-marked dialect string (``내가<UP> 최근에<KEEP>``)."""
    return " ".join(f"{m['word']}{m['marker']}" if m.get("marker") else m["word"] for m in marked)


def apply_eojeol_markers_to_text(dialect_text: str, eojeol_prosody: list[dict] | None) -> str:
    """Append per-eojeol markers to the *original* dialect string, faithfully.

    Markers are attached to each word of ``dialect_text`` from ``eojeol_prosody`` (the
    ``dialect_eojeol_prosody`` column) **only when the two align 1:1** — i.e. the marker list
    has exactly one entry per dialect word. On any misalignment (segment count ≠ word count,
    ~23% of rows from N:M mappings / ghost spaces) or empty input, the text is returned
    unchanged. This guarantees the SFT target is the true dialect words plus markers, never a
    reconstruction that could drop or reorder words — inter-word whitespace is canonicalised
    to single spaces (stripping markers recovers the gold dialect up to that normalisation).
    Callers count fallbacks via :func:`eojeol_markers_aligned`.
    """
    words = dialect_text.split()
    if not eojeol_prosody or len(eojeol_prosody) != len(words):
        return dialect_text
    return " ".join(
        f"{word}{m['marker']}" if m.get("marker") else word
        for word, m in zip(words, eojeol_prosody)
    )


def eojeol_markers_aligned(dialect_text: str, eojeol_prosody: list[dict] | None) -> bool:
    """True when per-eojeol markers align 1:1 with the dialect words (so they were applied)."""
    return bool(eojeol_prosody) and len(eojeol_prosody) == len(dialect_text.split())


def add_sentence_final_marker(text: str, marker: str | None) -> str:
    """Insert a prosody marker before sentence-final punctuation when present."""
    text = text.strip()
    if not text or not marker:
        return text

    punctuation = ".?!。！？"
    if text[-1] in punctuation:
        return f"{text[:-1]}{marker}{text[-1]}"
    return f"{text}{marker}"


_MARKER_RE = re.compile("|".join(re.escape(m) for m in PROSODY_MARKER_POLICY["markers"]))


def strip_markers(text: str) -> str:
    """Remove any prosody markers from generated text and tidy whitespace.

    A prosody-supervised model emits markers (e.g. ``밥 뭇나<UP>?``); they must be
    stripped before scoring against marker-free gold so the A/B compares dialect content,
    not the marker tokens themselves.
    """
    cleaned = _MARKER_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()
