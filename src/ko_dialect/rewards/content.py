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


def r_overcorrection(prompts, completions, standard, dialect, direction=None, **kw):
    """Overcorrection guard: penalize drifting *farther from the source than the gold does*.

    Observed failure mode (Arm1 qualitative gate, edit-distance ≤ 1 Gangwon rows where
    gold ≈ source): the policy invents dialect forms that appear in *neither* source nor
    gold (나오고→나온구나, 요새→오새), driven by the unbounded ``r_style`` "move away from
    standard" pull. ``r_copy_margin`` rewards *approaching* the gold but places no cap on
    *overshooting past it* — once gen has moved to the gold it can keep drifting and
    ``r_style`` keeps paying out. This term supplies the missing upper rail.

    The gold defines the *legitimate edit budget*: how far the reference itself moves from
    the source. Measuring distance as ``100 − chrF(·, source)`` (chrF points):

        d_gold = 100 − chrF(gold, source)          # the reference's own drift
        d_gen  = 100 − chrF(gen,  source)           # the generation's drift
        over   = max(0, d_gen − d_gold)
               = max(0, chrF(gold, source) − chrF(gen, source))
        reward = 1 − over / 100   ∈ [0, 1]

    A faithful conversion stays within the gold's budget (``d_gen ≤ d_gold`` ⇒ ``over=0`` ⇒
    1.0); only *excess* drift past the reference is penalized, so legitimate dialect
    transformation is never punished. The guard is deliberately **one-sided**: a pure copy
    (``d_gen < d_gold``) also scores 1.0 here — under-conversion is already penalized by
    ``r_copy_margin`` (→0.5) and ``r_edit_recall``, so this axis must not double-charge it.
    Pairing the two gives a two-sided anchor: *approach the gold (copy_margin) but do not
    overshoot it (overcorrection)*.

    ``gold`` / ``source`` follow ``direction`` exactly as in :func:`r_copy_margin`.

    Cite: Pauli, Augenstein & Assent 2025, "Mind the Style Gap" (arXiv:2502.15022), on over-stylization /
    copy-bias in TST metrics; Fu et al. 2018 content-preservation budget; plan
    .omc/plans/ralplan-grpo-rewards.md §3 A3 (copy-margin) + Arm1 qualitative gate follow-up.
    """
    texts = extract_texts(completions)
    rewards = []
    for i, (gen, std, dia) in enumerate(zip(texts, standard, dialect)):
        dir_i = direction[i] if direction is not None else "std2dia"
        if dir_i == "std2dia":
            gold, source = dia, std
        else:
            gold, source = std, dia
        chrf_gold_src = _chrf.sentence_score(gold, [source]).score
        chrf_gen_src = _chrf.sentence_score(gen, [source]).score
        over = max(0.0, chrf_gold_src - chrf_gen_src)  # ∈ [0, 100]
        rewards.append(1.0 - over / 100.0)
    return rewards
