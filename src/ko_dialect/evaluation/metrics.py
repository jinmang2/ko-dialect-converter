from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)

# Must match TextCNNConfig
LABEL_STANDARD = 0
DO_TO_LABEL: dict[str, int] = {"gangwondo": 1, "gyeongsangdo": 2}


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


def evaluate_all(
    outputs: list[str],
    dialect_refs: list[str],
    standard_refs: list[str],
    dialect_eojeol_maps: list[list[dict[str, Any]]],
    target_do: str,
    classifier,
    cls_tokenizer,
    embed_fn: Callable[[list[str]], torch.Tensor] | None = None,
) -> dict[str, float]:
    """Compute TDR, DFS (if embed_fn provided), and eojeol accuracy."""
    results: dict[str, float] = {}
    results["tdr"] = compute_tdr(outputs, target_do, classifier, cls_tokenizer)
    if embed_fn is not None:
        results["dfs"] = compute_dfs(outputs, dialect_refs, standard_refs, embed_fn)
    results["eojeol_accuracy"] = compute_eojeol_accuracy(outputs, dialect_eojeol_maps)
    results["bleu"] = compute_bleu(outputs, dialect_refs)
    results["chrf"] = compute_chrf(outputs, dialect_refs)
    return results
