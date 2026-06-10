"""Quality filters for the dialect *classifier* dataset.

Why this exists
---------------
The classifier is trained to be a **GRPO reward model**: the style reward is
``p(target_do) - p(standard)`` (see ``rewards/style.py``). Its job is to tell a
genuine dialect translation (*True Attempt*) apart from an output that merely copies
the standard source (*False Success*) — the central distinction in DIA-REFINE
(Park et al., LREC 2026, arXiv:2511.06680).

A dialect sentence that differs from its standard form by a single morpheme is
*almost* standard text wearing a dialect label. Training on it teaches the reward
model that near-standard strings deserve dialect credit — which is exactly the
reward-hacking signal we must avoid. The paper's fix, reproduced here, is to admit a
dialect sample only when it diverges enough at the surface level:

    normalized Levenshtein distance(standard, dialect) >= 0.1

We add two more guards specific to the AI-Hub corpus:
  * **standard-class de-pollution** — some AI-Hub "standard" transcriptions retain
    high-precision dialect endings (e.g. ``-카노``, ``-드래요``). Emitting those under
    label 0 blurs the standard boundary the reward depends on.
  * **collision removal** — a string that appears under more than one label is an
    unlearnable contradiction; drop it.
"""
from __future__ import annotations

import re


def norm_levenshtein(a: str, b: str) -> float:
    """Character-level Levenshtein distance normalized to ``[0, 1]``.

    ``0`` = identical, ``1`` = maximally different. Pure-Python DP so the data
    pipeline has no hard dependency on ``python-Levenshtein``; the strings here are
    short utterances, so the O(len_a*len_b) cost is negligible. Uses the bundled fast
    C implementation when available.
    """
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    try:  # fast path when python-Levenshtein is installed
        import Levenshtein

        return Levenshtein.distance(a, b) / max(len(a), len(b))
    except ImportError:
        pass
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ai != b[j - 1]),
            )
        prev = cur
    return prev[lb] / max(la, lb)


# High-precision (low false-positive) dialect sentence-ending markers. These almost
# never appear in genuine standard Korean, so a "standard"-labelled sentence carrying
# one is very likely a mislabeled dialect transcription. Kept deliberately small and
# conservative — the goal is to remove obvious label noise, not to re-classify.
GYEONGSANG_MARKERS: tuple[str, ...] = (
    "카노",
    "카나",
    "카니",
    "캤",
    "했능교",
    "능교",
    "하이소",
    "마이소",
    "이라예",
    "거든예",
    "랍니더",
    "심더",
    "니더",
    "아입니",
    "아이가",
)
# Gangwon endings overlap heavily with standard Korean, so we keep only the few that
# are unambiguous (see the paper's note that Gangwon is close to standard).
GANGWON_MARKERS: tuple[str, ...] = ("드래요", "드래여")

_MARKER_RE = re.compile("|".join(re.escape(m) for m in GYEONGSANG_MARKERS + GANGWON_MARKERS))


def carries_dialect_marker(text: str) -> bool:
    """True if ``text`` contains a high-precision dialect ending (likely mislabeled)."""
    return bool(_MARKER_RE.search(text))
