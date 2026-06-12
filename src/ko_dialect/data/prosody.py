from __future__ import annotations

from typing import Any

PROSODY_MARKER_POLICY_VERSION = 1
PROSODY_MARKER_POLICY: dict[str, Any] = {
    "version": PROSODY_MARKER_POLICY_VERSION,
    "markers": ["<UP>", "<DOWN>", "<WAVE>", "<KEEP>"],
    "valid_f0_min_hz": 30.0,
    "min_valid_points": 3,
    "up_threshold_delta_ratio": 0.15,
    "down_threshold_delta_ratio": -0.15,
    "wave_threshold_range_hz": 50.0,
    "placement": "before sentence-final punctuation, otherwise append to text",
}


def summarize_intonation(intonations: list[float]) -> dict | None:
    """raw F0 시계열 -> 요약 통계 (None 값/0 값 제외)"""
    valid = [
        p for p in intonations if p and p > PROSODY_MARKER_POLICY["valid_f0_min_hz"]
    ]
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


def prosody_marker(prosody: dict | None) -> str | None:
    """Map sentence-level F0 summary to a compact text marker."""
    if not prosody:
        return None

    delta = prosody.get("f0_delta", 0.0)
    if delta >= PROSODY_MARKER_POLICY["up_threshold_delta_ratio"]:
        return "<UP>"
    if delta <= PROSODY_MARKER_POLICY["down_threshold_delta_ratio"]:
        return "<DOWN>"
    if prosody.get("f0_range", 0.0) >= PROSODY_MARKER_POLICY["wave_threshold_range_hz"]:
        return "<WAVE>"
    return "<KEEP>"


def add_sentence_final_marker(text: str, marker: str | None) -> str:
    """Insert a prosody marker before sentence-final punctuation when present."""
    text = text.strip()
    if not text or not marker:
        return text

    punctuation = ".?!。！？"
    if text[-1] in punctuation:
        return f"{text[:-1]}{marker}{text[-1]}"
    return f"{text}{marker}"
