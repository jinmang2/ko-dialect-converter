from __future__ import annotations

from ._utils import extract_texts, normalize_eojeol, tokenize_words


def r_edit(prompts, completions, standard, dialect, dialect_eojeol_map, direction=None, **kw):
    """Edit-detection reward using dialect_eojeol_map.

    Does NOT require gold matching — rewards:
      (0.7) fraction of source-form words that were removed from the output
      (0.3) fraction of gold dialect words that appeared in the output (soft bonus)

    This avoids punishing many-valid answers while still pushing the model
    to transform the right eojeols. Enable once r_style+r_content are stable.

    Reward-hacking guard: both the generated tokens and the gold eojeols are
    punctuation-normalized (:func:`tokenize_words` / :func:`normalize_eojeol`) before
    matching. A naive ``set(gen.split())`` would treat "갔어," and "갔어" as different,
    which (a) miscounts a still-present standard word as "removed" — inflating the
    reward — and (b) unfairly misses a gold dialect word that carries trailing
    punctuation. Normalizing both sides removes that surface-form exploit.

    Note: a future, stricter alternative is token-index alignment of the eojeol_map
    (MRC-style span matching) built at data-prep time; normalization covers the
    common punctuation case without a data-pipeline change.
    """
    texts = extract_texts(completions)
    rewards = []
    for i, (gen, std, dia, eojeol_map) in enumerate(
        zip(texts, standard, dialect, dialect_eojeol_map)
    ):
        eojeol_map = list(eojeol_map) if eojeol_map else []
        dir_i = direction[i] if direction is not None else "std2dia"
        src = std if dir_i == "std2dia" else dia

        if not eojeol_map:
            # is_identical row: reward preservation
            rewards.append(1.0 if gen.strip() == src.strip() else 0.2)
            continue

        gen_word_set = tokenize_words(gen)

        std_words = [
            n for e in eojeol_map if (n := normalize_eojeol(e.get("standard", "")))
        ]
        dia_words = [
            n for e in eojeol_map if (n := normalize_eojeol(e.get("dialect", "")))
        ]

        if not std_words:
            rewards.append(0.5)
            continue

        # Primary: did the model remove the standard-form words?
        removed = sum(1 for w in std_words if w not in gen_word_set)
        p_removed = removed / len(std_words)

        # Bonus: did the model use gold dialect words? (not mandatory)
        p_hit = (
            sum(1 for w in dia_words if w in gen_word_set) / len(dia_words) if dia_words else 0.0
        )

        rewards.append(0.7 * p_removed + 0.3 * p_hit)

    return rewards
