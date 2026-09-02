from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from ko_dialect.data.labels import DO_TO_LABEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eval framework note (plan §3, ADR): why checkpoint selection uses proxy-INDEPENDENT
# signals only (reconstruction_bleu + qualitative gate), never J-score/TDR/DFS.
#
# J-score below is MONITORING ONLY — never a selection objective. It aggregates axes that
# overlap with the training rewards, so selecting by it creates a reward-proxy circularity
# that re-introduces over-optimisation. The grounding for this rule is OURS, from three
# converging sources — NOT a prescription of any single paper:
#   (a) MEASURED here: the 500-step GRPO run drove the classifier reward up (TDR past the
#       gold's own TDR) while chrF/BLEU vs gold *dropped ~9–11 pts* — reward↑, fidelity↓.
#       (memory grpo-reward-over-optimization; the over-optimised checkpoints are on disk.)
#   (b) DOMAIN: for Gangwon, gold ≈ standard, so any classifier/n-gram proxy is especially
#       easy to game (a near-copy already scores well) — the copy-bias is acute in-domain.
#   (c) LITERATURE: DIA-REFINE (arXiv:2511.06680) shows n-gram/classifier metrics reward
#       source-copying ("False Success"); MO-GRPO (arXiv:2509.22047) shows multi-objective
#       reward circularity exists. These support the PREMISE; the selection RULE is our call.
#
# Proxy-independent selection signals:
#   1. reconstruction_bleu  — dialect→standard reverse generation vs source
#      (no classifier, no eojeol_map; Dual-RL / Luo et al. 2019)
#   2. qualitative gate     — stratified human/LLM-judge samples (AC4)
#
# copy_margin = chrF(gen, gold) − chrF(gen, source) strips the copy-bias that
# makes raw chrF-vs-gold unreliable for Gangwon where gold ≈ standard
# (plan §3, A3; Mind the Style Gap arXiv:2502.15022 §4).
# ---------------------------------------------------------------------------

# Label ids come from the single source of truth (ko_dialect.data.labels) so the
# classifier, rewards, and these metrics can never drift out of sync — see top imports.


@torch.no_grad()
def compute_tdr(
    outputs: list[str],
    target_do: str,
    classifier,
    tokenizer,
    max_length: int = 128,
) -> float:
    """Target Dialect Rate: fraction of outputs the classifier tags as *target_do*.

    From Park et al. (arXiv:2511.06680): measures how well the model adopts the target dialect.
    """
    if not outputs:
        return 0.0

    target_label = DO_TO_LABEL[target_do]
    device = next(classifier.parameters()).device

    encoded = tokenizer(
        outputs,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    logits = classifier(encoded["input_ids"].to(device)).logits
    preds = logits.argmax(dim=-1)
    return float((preds == target_label).float().mean().item())


def compute_dfs(
    outputs: list[str],
    dialect_refs: list[str],
    standard_refs: list[str],
    embed_fn: Callable[[list[str]], torch.Tensor],
    eps: float = 1e-6,
) -> float:
    """Dialect Fidelity Score: avg[log(1+cos(out,dia)+ε) - log(1+cos(out,std)+ε)].

    From Park et al. (arXiv:2511.06680). ``embed_fn`` maps a list of strings to a (N, D) tensor.
    """
    if not outputs:
        return 0.0

    out_emb = embed_fn(outputs)
    dia_emb = embed_fn(dialect_refs)
    std_emb = embed_fn(standard_refs)

    sim_dia = F.cosine_similarity(out_emb, dia_emb, dim=-1)
    sim_std = F.cosine_similarity(out_emb, std_emb, dim=-1)
    dfs = (torch.log(1 + sim_dia + eps) - torch.log(1 + sim_std + eps)).mean().item()
    return float(dfs)


def compute_eojeol_accuracy(
    outputs: list[str],
    dialect_eojeol_maps: list[list[dict[str, Any]]],
) -> float:
    """Eojeol-level dialect accuracy using ``dialect_eojeol_map`` from the dataset.

    Novel metric not in DIA-REFINE: checks what fraction of expected dialect eojeols
    appear verbatim in the generated output. Rewards precise lexical transformation
    rather than style alone.
    """
    if not outputs:
        return 0.0

    hit, total = 0, 0
    for out_text, eojeol_map in zip(outputs, dialect_eojeol_maps):
        out_words = set(out_text.split())
        for entry in eojeol_map:
            dialect_word = (entry.get("dialect") or "").strip()
            if not dialect_word:
                continue
            total += 1
            if dialect_word in out_words:
                hit += 1

    return hit / total if total > 0 else 0.0


def compute_bleu(outputs: list[str], refs: list[str]) -> float:
    """Corpus-level BLEU (sacrebleu) against dialect references."""
    if not outputs:
        return 0.0
    from sacrebleu.metrics import BLEU

    return BLEU(effective_order=True).corpus_score(outputs, [refs]).score


def compute_chrf(outputs: list[str], refs: list[str]) -> float:
    """Corpus-level chrF (sacrebleu) against dialect references."""
    if not outputs:
        return 0.0
    from sacrebleu.metrics import CHRF

    return CHRF().corpus_score(outputs, [refs]).score


def compute_chrf_source(outputs: list[str], source_refs: list[str]) -> float:
    """Corpus-level chrF of generated outputs against the *source* (standard Korean).

    Used together with compute_chrf (vs gold dialect) to derive copy_margin.
    A generation that merely copies the source will score ~100 here → copy_margin ≈ 0.
    """
    if not outputs:
        return 0.0
    from sacrebleu.metrics import CHRF

    return CHRF().corpus_score(outputs, [source_refs]).score


def compute_copy_baseline(source_refs: list[str], gold_refs: list[str]) -> float:
    """chrF the do-nothing model scores: emit the input unchanged.

    A property of the *region*, not of any model. Measured dia2std on held-out data it
    is 56.7 for Gyeongsang, 76.8 Gangwon, 79.9 Chungcheong, 18.9 Jeju — which is why raw
    chrF cannot be compared across regions (docs/CORPUS_ANALYSIS_5REGION.md §6).
    """
    if not source_refs:
        return 0.0
    return compute_chrf(source_refs, gold_refs)


def compute_gain_over_copy(
    outputs: list[str], source_refs: list[str], gold_refs: list[str]
) -> float:
    """chrF(gen, gold) − chrF(source, gold): value added over emitting the input.

    Negative means the model is worse than doing nothing. Distinct from
    :func:`compute_copy_margin`, which penalises a generation for *resembling* the
    source; this compares against the region's floor instead. Lives here rather than
    being inlined by each caller so the definition cannot drift between the leaderboard
    and ``scripts/eval_dia2std.py``.

    The *principle* — score relative to the unchanged input, negative meaning the system
    made things worse — is the one behind the I-measure (Felice & Briscoe, NAACL 2015,
    "Towards a standard evaluation method for grammatical error detection and
    correction"), which exists because F-score cannot tell a do-nothing system from one
    that only makes wrong corrections. This is **not** an implementation of that metric:
    the I-measure is a token-level three-way alignment, this is a corpus-chrF difference.
    See docs/REFERENCES.md A13.
    """
    if not outputs:
        return 0.0
    return compute_chrf(outputs, gold_refs) - compute_copy_baseline(source_refs, gold_refs)


def compute_copy_margin(outputs: list[str], gold_refs: list[str], source_refs: list[str]) -> float:
    """copy_margin = chrF(gen, gold) − chrF(gen, source).

    Removes copy-bias for dialects (e.g. Gangwon) where gold ≈ source:
    a model that simply echoes the input scores near 0 instead of near 100.

    Reference: plan §3 A3; Mind the Style Gap arXiv:2502.15022 §4.
    """
    return compute_chrf(outputs, gold_refs) - compute_chrf_source(outputs, source_refs)


def _remap_unit(value: float, lo: float = -1.0, hi: float = 1.0, eps: float = 1e-6) -> float:
    """Linearly remap *value* from [lo, hi] to (0, 1], clamped, with ε floor.

    Used to ensure every axis fed into J-score lives in (0, 1] so that
    log/geometric operations are well-defined.
    """
    span = hi - lo
    remapped = (value - lo) / span if span > 0 else 0.5
    return max(eps, min(1.0, remapped))


def compute_jscore(
    tdr: float,
    content: float,
    edit: float,
    fluency: float,
    *,
    content_is_copy_margin: bool = True,
    tdr_range: tuple[float, float] = (0.0, 1.0),
    content_range: tuple[float, float] = (-100.0, 100.0),
    edit_range: tuple[float, float] = (0.0, 1.0),
    fluency_range: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Geometric mean of four (0,1]-remapped evaluation axes.

    MONITORING ONLY — must NOT be used as checkpoint-selection objective.
    Circularity: J-score aggregates axes that overlap with training rewards;
    using it for selection recreates the reward-proxy loop (plan §3 A4;
    MO-GRPO arXiv:2509.22047).

    Selection uses proxy-independent signals:
      reconstruction_bleu + qualitative gate (AC4).

    J-score is useful for:
      - visualising the learning curve (all four axes in one number)
      - detecting over-optimisation: a J-score peak followed by decline
        while TDR keeps rising signals style/content trade-off collapse (AC5)

    Parameters
    ----------
    tdr:
        Target Dialect Rate ∈ [0, 1].
    content:
        Either copy_margin (recommended) or raw chrF-vs-gold.
        Set ``content_is_copy_margin=True`` when passing copy_margin so the
        appropriate range [-100, 100] is applied.
    edit:
        Eojeol-level edit accuracy ∈ [0, 1].
    fluency:
        Fluency proxy (e.g. 1 − normalised PPL) ∈ [0, 1].
    content_is_copy_margin:
        When True, ``content_range`` defaults to (−100, 100); otherwise (0, 100).
    """
    if not content_is_copy_margin:
        content_range = (0.0, 100.0)

    t = _remap_unit(tdr, *tdr_range)
    c = _remap_unit(content, *content_range)
    e = _remap_unit(edit, *edit_range)
    f = _remap_unit(fluency, *fluency_range)

    # Geometric mean — all axes already in (0,1] so product is well-defined
    return float((t * c * e * f) ** 0.25)


def reconstruction_bleu(reverse_outputs: list[str], source_refs: list[str]) -> float:
    """BLEU of reverse-generated (dialect→standard) outputs vs original standard source.

    This is the proxy-independent content-preservation signal used for
    checkpoint selection (plan §3, AC1; Dual-RL framework: Luo et al. 2019).

    The reverse generation pass is NOT performed here — the caller (eval script)
    must produce ``reverse_outputs`` by re-prompting the model with a
    dialect→standard instruction on the model's own std→dialect outputs.

    Independence: does not use the classifier, eojeol_map, or any reward proxy,
    so it cannot be gamed by reward over-optimisation.
    """
    if not reverse_outputs:
        return 0.0
    from sacrebleu.metrics import BLEU

    return BLEU(effective_order=True).corpus_score(reverse_outputs, [source_refs]).score


def evaluate_all(
    outputs: list[str],
    dialect_refs: list[str],
    standard_refs: list[str],
    dialect_eojeol_maps: list[list[dict[str, Any]]],
    target_do: str,
    classifier,
    cls_tokenizer,
    embed_fn: Callable[[list[str]], torch.Tensor] | None = None,
    *,
    reverse_outputs: list[str] | None = None,
) -> dict[str, float]:
    """Compute TDR, DFS (if embed_fn provided), eojeol accuracy, chrF, BLEU.

    Extended metrics (plan §3 / ralplan-grpo-rewards.md):
    - ``chrf_source``: chrF(gen, source) — needed to derive copy_margin.
    - ``copy_margin``: chrF(gen, gold) − chrF(gen, source).  Strips copy-bias
      for dialects where gold ≈ source (Gangwon).  Replaces raw chrF as the
      content-preservation axis (A3; Mind the Style Gap arXiv:2502.15022).
    - ``copy_baseline``: chrF(source, gold) — what emitting the input unchanged
      already scores.  It is a property of the *region*, not of the model, and it
      varies enormously: measured dia2std on held-out data it is 56.7 for Gyeongsang
      but 79.9 for Chungcheong and 18.9 for Jeju.  Without it a raw chrF of 76.9
      (Gangwon) reads as better than 83.9 is good, when in fact the first is a
      +0.07 gain over doing nothing and the second is +27.
    - ``gain_over_copy``: chrF(gen, gold) − copy_baseline.  Value added over the
      do-nothing model; **negative means the model is worse than emitting its input**.
      This is the axis to compare *across regions* — raw chrF is not comparable
      because each region has a different floor.  Distinct from copy_margin, which
      is a per-generation penalty for resembling the source rather than a
      per-region floor (docs/CORPUS_ANALYSIS_5REGION.md §6).
    - ``reconstruction_bleu``: BLEU of dialect→standard reverse generation vs
      original source — proxy-independent selection signal (A4; Luo et al. 2019).
      Only computed when ``reverse_outputs`` is supplied.
    - ``jscore``: geometric mean of (0,1]-remapped {tdr, copy_margin, eojeol,
      fluency=0.5 placeholder}.  MONITORING ONLY — not a selection objective.

    All pre-v2 keys are preserved for back-compat.
    """
    results: dict[str, float] = {}
    results["tdr"] = compute_tdr(outputs, target_do, classifier, cls_tokenizer)
    if embed_fn is not None:
        results["dfs"] = compute_dfs(outputs, dialect_refs, standard_refs, embed_fn)
    results["eojeol_accuracy"] = compute_eojeol_accuracy(outputs, dialect_eojeol_maps)
    results["bleu"] = compute_bleu(outputs, dialect_refs)
    results["chrf"] = compute_chrf(outputs, dialect_refs)

    # --- extended metrics ---
    results["chrf_source"] = compute_chrf_source(outputs, standard_refs)
    results["copy_margin"] = results["chrf"] - results["chrf_source"]
    # This function assumes std2dia throughout (gold = dialect, source = standard), like
    # the rest of the leaderboard. scripts/eval_dia2std.py passes the pair the other way
    # round for the deployed direction — hence the explicit (source, gold) argument order.
    results["copy_baseline"] = compute_copy_baseline(standard_refs, dialect_refs)
    results["gain_over_copy"] = compute_gain_over_copy(outputs, standard_refs, dialect_refs)

    if reverse_outputs is not None:
        results["reconstruction_bleu"] = reconstruction_bleu(reverse_outputs, standard_refs)

    # J-score: MONITORING ONLY (plan §3 A4 — not selection objective)
    results["jscore"] = compute_jscore(
        tdr=results["tdr"],
        content=results["copy_margin"],
        edit=results["eojeol_accuracy"],
        fluency=0.5,  # placeholder: caller may override with 1 - norm_ppl
        content_is_copy_margin=True,
    )

    return results
