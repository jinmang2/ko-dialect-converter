#!/usr/bin/env python3
"""Sweep GRPO checkpoints to find the over-optimization point and pick the best one.

GRPO maximises the classifier reward (TDR), but the classifier is a hackable proxy:
reward can keep rising while the *real* translation fidelity (chrF/BLEU vs gold dialect)
degrades. This evaluates the SFT base and every grpo checkpoint-* on a fixed valid
subset and prints TDR / chrF / copy_margin / reconstruction-BLEU / eojeol-accuracy per
step, so you can:
  - see the over-optimisation curve (chrF peaks early, then drops as TDR keeps rising)
  - select the best checkpoint by a *proxy-independent* metric (reconstruction-BLEU or
    copy_margin; NOT J-score — see note below)

Checkpoint-selection note (plan §3 A4; MO-GRPO arXiv:2509.22047):
  J-score and TDR overlap with training rewards — using them as selection criteria
  creates a reward-proxy circularity.  Use proxy-independent signals only:
    1. reconstruction_bleu  (dialect→standard reverse generation vs source)
    2. qualitative gate     (stratified human/LLM-judge samples, AC4)

Stratified buckets (plan §3 / compare_sft_grpo.py):
  Valid set is split by edit-distance bucket (small / med / large) × dialect region
  (gangwon / gyeongsang) so per-stratum metrics reveal where copy-bias lives.

References:
  - MO-GRPO            arXiv:2509.22047  (per-objective normalisation; selection note)
  - Mind the Style Gap arXiv:2502.15022  (copy_margin / J-score limits)
  - Dual-RL Luo 2019                     (reconstruction / cycle reward)

Usage:
    python scripts/eval_grpo_checkpoints.py --grpo_dir outputs/grpo_500 --n 150
"""

from __future__ import annotations

import logging
from collections import defaultdict

import fire
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from ko_dialect.evaluation import evaluate_all, generate_batched, grpo_runs
from ko_dialect.models import TextCNNForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sweep")

BASE = "outputs/sft_merged"
CLS = "outputs/classifier_clean"
CLS_TOK = CLS

# Edit-distance thresholds for small / med / large buckets (character-level Levenshtein).
BUCKET_SMALL = 3
BUCKET_MED = 8


def _lev_distance(a: str, b: str) -> int:
    """Simple character-level Levenshtein distance."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[lb]


def _edit_bucket(standard: str, dialect: str) -> str:
    d = _lev_distance(standard, dialect)
    if d <= BUCKET_SMALL:
        return "small"
    if d <= BUCKET_MED:
        return "med"
    return "large"


def _dialect_region(do: str) -> str:
    if "gangwon" in do.lower():
        return "gangwon"
    if "gyeongsang" in do.lower():
        return "gyeongsang"
    return "other"


def _build_dia2std_prompt(dialect_text: str) -> str:
    """Reverse prompt: dialect → standard Korean (for reconstruction-BLEU pass)."""
    return f"다음 방언 문장을 표준어로 바꿔줘.\n방언: {dialect_text}\n표준어: "


def generate(model, tok, prompts, device, batch_size=16, max_new_tokens=64):
    return generate_batched(
        model,
        tok,
        prompts,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_length=448,
    )


def eval_one(
    base,
    adapter,
    prompts,
    gold,
    src,
    emaps,
    target_do,
    classifier,
    cls_tok,
    model_tok,
):
    """Load model (+ optional adapter), generate, compute all metrics including
    reconstruction-BLEU via a second reverse-generation pass."""
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    device = next(model.parameters()).device
    classifier.to(device).eval()

    # Forward pass: standard → dialect
    outs = generate(model, model_tok, prompts, device)

    # Reverse pass: model's output dialect → standard (reconstruction-BLEU)
    # Dual-RL / Luo et al. 2019: cycle back through the model with a reversed prompt.
    rev_prompts = [_build_dia2std_prompt(o) for o in outs]
    rev_outs = generate(model, model_tok, rev_prompts, device)

    res = evaluate_all(
        outputs=outs,
        dialect_refs=gold,
        standard_refs=src,
        dialect_eojeol_maps=emaps,
        target_do=target_do,
        classifier=classifier,
        cls_tokenizer=cls_tok,
        reverse_outputs=rev_outs,
    )
    del model
    torch.cuda.empty_cache()
    return res, outs


def eval_buckets(
    outs: list[str],
    gold: list[str],
    src: list[str],
    emaps: list,
    target_do: str,
    classifier,
    cls_tok,
    bucket_keys: list[str],
):
    """Return per-bucket aggregate metrics dict keyed by bucket label."""
    # Group indices by bucket
    idx_by_bucket: dict[str, list[int]] = defaultdict(list)
    for i, bk in enumerate(bucket_keys):
        idx_by_bucket[bk].append(i)

    bucket_results: dict[str, dict] = {}
    for bk, idxs in sorted(idx_by_bucket.items()):
        b_outs = [outs[i] for i in idxs]
        b_gold = [gold[i] for i in idxs]
        b_src = [src[i] for i in idxs]
        b_emaps = [emaps[i] for i in idxs]
        b_res = evaluate_all(
            outputs=b_outs,
            dialect_refs=b_gold,
            standard_refs=b_src,
            dialect_eojeol_maps=b_emaps,
            target_do=target_do,
            classifier=classifier,
            cls_tokenizer=cls_tok,
        )
        bucket_results[bk] = b_res
    return bucket_results


def main(
    grpo_dir: str = "outputs/grpo_500",
    target_do: str = "gangwondo",
    n: int = 150,
    select_by: str = "reconstruction_bleu",
    base: str = BASE,
    classifier_path: str = CLS,
    cls_tokenizer_path: str = CLS_TOK,
    dataset_path: str = "outputs/datasets/grpo",
):
    """Sweep checkpoints.  Default selection metric is reconstruction_bleu (proxy-independent).

    Checkpoint selection note (plan §3 A4; MO-GRPO arXiv:2509.22047):
      J-score overlaps with training rewards → circularity.  Use reconstruction_bleu
      (+ qualitative gate AC4) for selection; J-score is monitoring/visualisation only.
    """
    ds = load_from_disk(dataset_path)["valid"]
    ds = ds.filter(
        lambda x: (
            x["do"] == target_do and x["direction"] == "std2dia" and x["standard"] != x["dialect"]
        )
    )
    ds = ds.select(range(grpo_runs.clamp_select_count(len(ds), n)))

    prompts = list(ds["prompt"])
    gold = list(ds["dialect"])
    src = list(ds["standard"])
    emaps = [m or [] for m in ds["dialect_eojeol_map"]]

    # Build stratified bucket labels: edit-distance bucket × dialect region
    bucket_keys = [
        f"{_dialect_region(row['do'])}_{_edit_bucket(row['standard'], row['dialect'])}"
        for row in ds
    ]

    cls_tok = AutoTokenizer.from_pretrained(cls_tokenizer_path, local_files_only=True)
    if cls_tok.pad_token is None:
        cls_tok.pad_token = cls_tok.eos_token
    classifier = TextCNNForSequenceClassification.from_pretrained(classifier_path)

    model_tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if model_tok.pad_token is None:
        model_tok.pad_token = model_tok.eos_token

    stages = grpo_runs.discover_grpo_stages(grpo_dir)
    if len(stages) == 1:
        logger.warning("No GRPO adapter checkpoints found under %s; evaluating SFT only.", grpo_dir)

    print(f"\nSweep: {target_do} std2dia, n={len(ds)}, select_by={select_by}")
    hdr = f"{'stage':10s} {'tdr':>8s} {'chrf':>8s} {'copy_mg':>9s} {'recon_b':>9s} {'bleu':>8s} {'eojeol':>8s} {'jscore':>8s}"
    print(hdr)
    rows = []
    all_outs: dict[str, list[str]] = {}
    for stage in stages:
        r, outs = eval_one(
            base,
            stage.adapter,
            prompts,
            gold,
            src,
            emaps,
            target_do,
            classifier,
            cls_tok,
            model_tok,
        )
        rows.append((stage.tag, r))
        all_outs[stage.tag] = outs
        recon = r.get("reconstruction_bleu", float("nan"))
        print(
            f"{stage.tag:10s} {r['tdr']:>8.4f} {r['chrf']:>8.3f} {r['copy_margin']:>+9.3f}"
            f" {recon:>9.3f} {r['bleu']:>8.3f} {r['eojeol_accuracy']:>8.4f}"
            f" {r.get('jscore', float('nan')):>8.4f}"
        )

    # --- per-bucket breakdown for the best checkpoint ---
    best = grpo_runs.best_by_metric(rows, select_by)
    print(
        f"\nBest by {select_by}: {best[0]}  ({select_by}={best[1].get(select_by, float('nan')):.3f}, tdr={best[1]['tdr']:.3f})"
    )
    print(
        "NOTE: J-score is MONITORING ONLY — not a selection criterion (plan §3 A4).\n"
        "      Use reconstruction_bleu + qualitative gate (AC4) for checkpoint selection."
    )

    print(f"\n--- Per-bucket breakdown: {best[0]} ---")
    b_results = eval_buckets(
        all_outs[best[0]], gold, src, emaps, target_do, classifier, cls_tok, bucket_keys
    )
    print(f"{'bucket':28s} {'n':>4s} {'tdr':>8s} {'chrf':>8s} {'copy_mg':>9s} {'eojeol':>8s}")
    for bk, br in sorted(b_results.items()):
        n_bk = bucket_keys.count(bk)
        print(
            f"{bk:28s} {n_bk:>4d} {br['tdr']:>8.4f} {br['chrf']:>8.3f}"
            f" {br['copy_margin']:>+9.3f} {br['eojeol_accuracy']:>8.4f}"
        )

    # Also print SFT buckets for comparison
    print("\n--- Per-bucket breakdown: SFT(0) ---")
    sft_b = eval_buckets(
        all_outs["SFT(0)"],
        gold,
        src,
        emaps,
        target_do,
        classifier,
        cls_tok,
        bucket_keys,
    )
    print(f"{'bucket':28s} {'n':>4s} {'tdr':>8s} {'chrf':>8s} {'copy_mg':>9s} {'eojeol':>8s}")
    for bk, br in sorted(sft_b.items()):
        n_bk = bucket_keys.count(bk)
        print(
            f"{bk:28s} {n_bk:>4d} {br['tdr']:>8.4f} {br['chrf']:>8.3f}"
            f" {br['copy_margin']:>+9.3f} {br['eojeol_accuracy']:>8.4f}"
        )

    print(
        "\nNOTE: copy_margin = chrF(gen,gold) − chrF(gen,source).  "
        "A value near 0 for Gangwon indicates copy-bias (gold ≈ source); "
        "positive margin = genuine dialect conversion."
    )


if __name__ == "__main__":
    fire.Fire(main)
