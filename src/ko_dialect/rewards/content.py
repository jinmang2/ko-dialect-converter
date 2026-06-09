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
