"""Bootstrap confidence intervals + paired significance for leaderboard metrics.

A point estimate like "SFT recon_bleu 38.4 vs arm2 37.1" is uninterpretable without
knowing whether the gap survives sampling noise on a 150-sentence slice. This adds the
canonical MT significance machinery:

  * bootstrap_ci          — percentile bootstrap CI of a per-sentence metric's mean
  * paired_bootstrap      — Koehn (2004) paired bootstrap resampling: resample sentence
                            indices *jointly* for both systems B times and count how
                            often each wins, giving a two-sided p-value for "system ≠
                            baseline" that respects the per-sentence pairing.

Reference: Philipp Koehn, "Statistical Significance Tests for Machine Translation
Evaluation", EMNLP 2004 — the paired bootstrap is reliable down to ~300 sentences and
is the de-facto standard for chrF/BLEU system comparison.

Pure stdlib (``random`` only) so it is deterministic under ``seed`` and unit-tested
without a GPU. Operates on *per-sentence* scores (e.g. sentence-level chrF); resampling
the mean of those is the standard, segment-aligned approximation to a corpus-metric
bootstrap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _resample_mean(values: list[float], idx: list[int]) -> float:
    return sum(values[i] for i in idx) / len(idx)


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    confidence: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mean": round(self.mean, 4),
            "ci_lo": round(self.lo, 4),
            "ci_hi": round(self.hi, 4),
            "confidence": self.confidence,
        }


def bootstrap_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 1234,
) -> BootstrapCI:
    """Percentile-bootstrap CI for the mean of ``values`` (per-sentence scores)."""
    if not values:
        raise ValueError("bootstrap_ci needs at least one value.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(_resample_mean(values, idx))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[max(0, int(alpha * n_boot))]
    hi = means[min(n_boot - 1, int((1.0 - alpha) * n_boot))]
    return BootstrapCI(mean=_mean(values), lo=lo, hi=hi, confidence=confidence)


@dataclass(frozen=True)
class PairedResult:
    delta: float  # mean(system) - mean(baseline)
    p_value: float  # two-sided: prob the sign of delta is noise
    win_rate: float  # fraction of resamples where system > baseline
    n_boot: int

    @property
    def significant_05(self) -> bool:
        return self.p_value < 0.05

    def as_dict(self) -> dict[str, float | bool | int]:
        return {
            "delta": round(self.delta, 4),
            "p_value": round(self.p_value, 4),
            "win_rate": round(self.win_rate, 4),
            "significant_05": self.significant_05,
            "n_boot": self.n_boot,
        }


def paired_bootstrap(
    system: list[float],
    baseline: list[float],
    *,
    n_boot: int = 1000,
    seed: int = 1234,
) -> PairedResult:
    """Koehn (2004) paired bootstrap: is ``system`` better than ``baseline``?

    Both score lists must be aligned per sentence (same order, same length). Each of
    ``n_boot`` trials draws one index multiset and applies it to *both* systems, so the
    pairing (a sentence both systems found easy/hard) is preserved. ``win_rate`` is the
    fraction of trials where the system's resampled mean exceeds the baseline's; the
    two-sided p-value is ``2 * min(win_rate, 1 - win_rate)`` clipped to [0, 1].
    """
    if len(system) != len(baseline):
        raise ValueError("system and baseline must be sentence-aligned (equal length).")
    if not system:
        raise ValueError("paired_bootstrap needs at least one sentence.")
    rng = random.Random(seed)
    n = len(system)
    observed_delta = _mean(system) - _mean(baseline)
    wins = losses = ties = 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d = _resample_mean(system, idx) - _resample_mean(baseline, idx)
        if d > 0:
            wins += 1
        elif d < 0:
            losses += 1
        else:
            ties += 1
    # One-sided p = fraction of resamples that contradict the observed direction
    # (ties never support either side). No observed difference => p = 1 (not significant).
    if observed_delta > 0:
        p_one = (losses + ties) / n_boot
    elif observed_delta < 0:
        p_one = (wins + ties) / n_boot
    else:
        p_one = 1.0
    return PairedResult(
        delta=observed_delta,
        p_value=min(1.0, 2.0 * p_one),
        win_rate=wins / n_boot,
        n_boot=n_boot,
    )


def significance_marker(result: PairedResult) -> str:
    """Compact symbol for tables: ``▲``/``▼`` significant up/down, ``≈`` not significant."""
    if not result.significant_05:
        return "≈"
    return "▲" if result.delta > 0 else "▼"
