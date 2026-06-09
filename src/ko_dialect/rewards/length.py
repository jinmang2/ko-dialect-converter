from __future__ import annotations

from ._utils import extract_texts


def make_length_reward(max_ratio: float = 1.5, tolerance: float = 0.2):
    """Factory: a DAPO-style *soft overlong penalty* reward, relative to the gold length.

    Why: GRPO/PPO maximise reward and tend to pad generations — verbosity bias.
    Here it is especially exploitable: a longer output has more chances for a gold
    dialect word to incidentally appear (``r_edit`` p_hit) and for standard words to
    be "absent" (inflated p_removed). For a translation task the output length should
    track the reference, so we anchor the penalty to the gold target length.

    Shape (per sample), with ``free = 1 + tolerance``::

        ratio = len(gen) / len(gold)
        ratio <= free        -> 1.0                       (no penalty band)
        free < ratio < max   -> (max_ratio - ratio) / (max_ratio - free)   (linear decay)
        ratio >= max_ratio   -> 0.0                       (fully penalised)

    Only over-length is penalised; under-length is already discouraged by
    content/chrF rewards. Output is in [0, 1] so it mixes cleanly with the other
    unit-scaled rewards. Add via config: ``- {name: length, weight: 0.2}``.
    """
    free = 1.0 + tolerance
    if max_ratio <= free:
        raise ValueError(
            f"length reward needs max_ratio ({max_ratio}) > 1 + tolerance ({free})."
        )
    decay_span = max_ratio - free

    def r_length(prompts, completions, standard, dialect, direction=None, **kw):
        texts = extract_texts(completions)
        rewards = []
        for i, (gen, std, dia) in enumerate(zip(texts, standard, dialect)):
            dir_i = direction[i] if direction is not None else "std2dia"
            gold = dia if dir_i == "std2dia" else std
            ref_len = len(gold.strip())
            gen_len = len(gen.strip())
            if ref_len == 0:
                # No reference to compare against: reward only an (also-)empty output.
                rewards.append(1.0 if gen_len == 0 else 0.0)
                continue
            ratio = gen_len / ref_len
            if ratio <= free:
                rewards.append(1.0)
            elif ratio >= max_ratio:
                rewards.append(0.0)
            else:
                rewards.append(float((max_ratio - ratio) / decay_span))
        return rewards

    r_length.__name__ = "r_length"
    return r_length
