from __future__ import annotations

from sacrebleu.metrics import CHRF

from ._utils import extract_texts

_chrf = CHRF()


def r_content(prompts, completions, standard, dialect, direction=None, **kw):
    """chrF(gen, gold) — Thank-you-BART content reward, no copy-baseline.

    The copy-baseline term chrF(src, gold) is a per-group constant in GRPO
    and cancels out during advantage normalisation, so it is intentionally omitted.
    """
    texts = extract_texts(completions)
    rewards = []
    for i, (gen, std, dia) in enumerate(zip(texts, standard, dialect)):
        dir_i = direction[i] if direction is not None else "std2dia"
        gold = dia if dir_i == "std2dia" else std
        score = _chrf.sentence_score(gen, [gold]).score / 100.0
        rewards.append(float(score))
    return rewards


def r_copy_margin(prompts, completions, standard, dialect, direction=None, **kw):
    """Copy-debiased content reward: chrF(gen, gold) − chrF(gen, source).

    Plain chrF-vs-gold over-rewards *copying the source* whenever ``gold ≈ source``
    — the Gangwon case, where many sentences are (near-)identical to standard. A
    model that just echoes the input then scores high on content while doing zero
    style transfer. The copy *margin* subtracts the source-similarity baseline, so
    an echo (chrF(gen,gold) ≈ chrF(gen,source)) collapses to ~0 and only genuine
    movement *toward the gold and away from the source* is rewarded.

    Unlike the per-group-constant baseline noted in :func:`r_content`, ``chrF(gen,
    source)`` depends on the *generation*, so it does NOT cancel under GRPO group
    normalisation — it is a real, gen-dependent penalty on copying.

    Mapping: the raw margin ∈ [-100, 100] (chrF points) is affine-mapped to [0, 1]
    via ``(margin + 100) / 200`` and clamped, so a pure copy (margin 0) → 0.5, a
    full move to gold → up to 1.0, and copying when it hurts → below 0.5. This keeps
    the reward on the same unit basis as the others (no separate range needed).

    ``gold`` / ``source`` follow ``direction``: for std2dia, source=standard,
    gold=dialect; flipped for dia2std.

    Cite: Fu et al. 2018, "Style Transfer in Text: Exploration and Evaluation"
    (G2/H2 content-preservation vs. copy baseline); Hallinan et al. 2025, "Mind the
    Style Gap" (arXiv:2502.15022), on copy-bias in style-transfer metrics.
    """
    texts = extract_texts(completions)
    rewards = []
    for i, (gen, std, dia) in enumerate(zip(texts, standard, dialect)):
        dir_i = direction[i] if direction is not None else "std2dia"
        if dir_i == "std2dia":
            gold, source = dia, std
        else:
            gold, source = std, dia
        chrf_gold = _chrf.sentence_score(gen, [gold]).score
        chrf_src = _chrf.sentence_score(gen, [source]).score
        margin = chrf_gold - chrf_src  # ∈ [-100, 100]
        unit = (margin + 100.0) / 200.0
        rewards.append(min(1.0, max(0.0, unit)))
    return rewards
