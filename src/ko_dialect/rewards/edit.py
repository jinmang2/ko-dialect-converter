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

        std_words = [n for e in eojeol_map if (n := normalize_eojeol(e.get("standard", "")))]
        dia_words = [n for e in eojeol_map if (n := normalize_eojeol(e.get("dialect", "")))]

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


def r_edit_precision(
    prompts, completions, standard, dialect, dialect_eojeol_map, direction=None, **kw
):
    """Non-target-eojeol *preservation* precision (collateral-damage guard).

    Complements :func:`r_edit`'s recall-style signal, which only measures whether
    target eojeols were transformed and can be hacked by mangling *non-target*
    eojeols (recall goes up while content is destroyed). This term measures the
    opposite: of the eojeols that should be *kept unchanged*, how many survive.

        nontarget = tokenize_words(src) − {normalize_eojeol(e["standard"]) for e in map}
        precision = |nontarget ∩ tokenize_words(gen)| / max(|nontarget|, 1)

    where ``src = standard`` for std2dia else ``dialect`` (the input the model saw).
    Subtracting the gold *standard* forms leaves exactly the words that must be
    copied through verbatim; rewarding their presence directly penalises the
    non-target damage that ``r_edit`` alone is blind to (plan §4-C / §11-M2).

    Note: matching is set-membership, so a non-target surface form that legitimately
    repeats in ``src`` is counted at most once (multiplicity cap). Repeated identical
    eojeols can therefore be mis-counted; MRC-style index alignment of the eojeol_map
    is the stricter follow-up (data-prep, plan R1) and is intentionally deferred here.

    Cite: Luo et al. 2019, "A Dual Reinforcement Learning Framework for Unsupervised
    Text Style Transfer" (edit-targeted / content-preservation signal); plan
    .omc/plans/ralplan-grpo-rewards.md §4-C, §11-M2.
    """
    texts = extract_texts(completions)
    rewards = []
    for i, (gen, std, dia, eojeol_map) in enumerate(
        zip(texts, standard, dialect, dialect_eojeol_map)
    ):
        eojeol_map = list(eojeol_map) if eojeol_map else []
        dir_i = direction[i] if direction is not None else "std2dia"
        src = std if dir_i == "std2dia" else dia

        src_words = tokenize_words(src)
        target_std = {n for e in eojeol_map if (n := normalize_eojeol(e.get("standard", "")))}
        # Words that must be carried through unchanged (everything but the targets).
        nontarget = src_words - target_std

        if not nontarget:
            # Nothing to preserve (whole sentence is a target, or empty src):
            # precision is vacuously satisfied.
            rewards.append(1.0)
            continue

        gen_word_set = tokenize_words(gen)
        kept = sum(1 for w in nontarget if w in gen_word_set)
        rewards.append(kept / len(nontarget))

    return rewards


def r_edit_recall(
    prompts, completions, standard, dialect, dialect_eojeol_map, direction=None, **kw
):
    """Target-eojeol transformation recall (did the right gold dialect forms appear).

    Of the eojeols flagged for change, how many produced their gold dialect form
    in the output::

        recall = |{normalize(e["dialect"])} ∩ tokenize_words(gen)| / num_target_eojeols

    Unlike :func:`r_edit`'s ``p_removed`` (which only checks the standard form was
    *removed* and so rewards deletion/garbling), this requires the *correct* dialect
    form to be present, so it cannot be satisfied by simply dropping the word. Paired
    with :func:`r_edit_precision` it forms a recall/precision pair (combine via F1 or
    as separate floor-aggregation axes, plan §4-C).

    Note: set-membership matching applies the same multiplicity cap as
    :func:`r_edit_precision` (a gold form repeated in the map is credited at most once).

    Cite: Luo et al. 2019, "A Dual Reinforcement Learning Framework for Unsupervised
    Text Style Transfer"; plan .omc/plans/ralplan-grpo-rewards.md §4-C, §11-M2.
    """
    texts = extract_texts(completions)
    rewards = []
    for i, (gen, eojeol_map) in enumerate(zip(texts, dialect_eojeol_map)):
        eojeol_map = list(eojeol_map) if eojeol_map else []
        dia_words = {n for e in eojeol_map if (n := normalize_eojeol(e.get("dialect", "")))}
        if not dia_words:
            # is_identical row (no targets): nothing to recall — vacuously satisfied.
            rewards.append(1.0)
            continue
        gen_word_set = tokenize_words(gen)
        hit = sum(1 for w in dia_words if w in gen_word_set)
        rewards.append(hit / len(dia_words))

    return rewards
