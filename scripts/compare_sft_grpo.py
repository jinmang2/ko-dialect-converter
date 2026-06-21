#!/usr/bin/env python3
"""Did GRPO actually improve dialect conversion over the SFT base?

Generates greedily from the SFT base and from SFT+GRPO adapter on the *same* valid
prompts, then reports independent metrics (TDR via the classifier, chrF/BLEU vs gold
dialect, copy_margin, reconstruction-BLEU) plus side-by-side qualitative samples.

This is the over-optimisation check: GRPO reward (classifier prob) going up only
matters if these independent metrics move too.

Stratified qualitative protocol (plan §3, AC4):
  Samples are split by edit-distance bucket (small / med / large) × dialect region
  (gangwon / gyeongsang).  N samples per bucket are printed in 3-column format
  (SFT / GRPO / gold) so per-stratum qualitative judgment is possible.

Checkpoint-selection note (plan §3 A4; MO-GRPO arXiv:2509.22047):
  J-score aggregates axes that overlap with training rewards → circularity.
  Selection uses proxy-independent signals only:
    1. reconstruction_bleu  — dialect→standard reverse generation vs source
       (Dual-RL framework: Luo et al. 2019)
    2. qualitative gate     — stratified human/LLM-judge samples (AC4;
       Mind the Style Gap arXiv:2502.15022: content metrics must be style-aware, and
       same-size LLM autoraters underperform them — so treat LLM-judge as a weak proxy)

copy_margin = chrF(gen, gold) − chrF(gen, source):
  Strips copy-bias for Gangwon where gold ≈ source (plan §3 A3).

Usage:
    python scripts/compare_sft_grpo.py --target_do gangwondo --n 200
"""

from __future__ import annotations

import logging
from collections import defaultdict

import fire
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from ko_dialect.evaluation import evaluate_all, generate_batched
from ko_dialect.evaluation.grpo_runs import clamp_select_count
from ko_dialect.models import TextCNNForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare")

BASE = "outputs/sft_merged"
GRPO_ADAPTER = "outputs/grpo_500"
CLS = "outputs/classifier_clean"
# Load the classifier tokenizer from the LOCAL classifier dir, not the Hub id.
# Newer transformers makes a model_info() Hub call during tokenizer init (the
# _patch_mistral_regex / is_base_mistral path); on a flaky link that raises
# httpx.RemoteProtocolError and kills the whole comparison. The classifier_clean
# dir ships the exact tokenizer the classifier was trained with, so this is both
# network-free and more correct.
CLS_TOK = CLS

BUCKET_SMALL = 3
BUCKET_MED = 8


def _lev_distance(a: str, b: str) -> int:
    """Character-level Levenshtein distance."""
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


def load(base: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tok


def _bucket_metrics(
    outs: list[str],
    gold: list[str],
    src: list[str],
    emaps: list,
    target_do: str,
    classifier,
    cls_tok,
    bucket_keys: list[str],
) -> dict[str, dict]:
    """Compute per-bucket aggregate metrics."""
    idx_by_bucket: dict[str, list[int]] = defaultdict(list)
    for i, bk in enumerate(bucket_keys):
        idx_by_bucket[bk].append(i)

    bucket_results: dict[str, dict] = {}
    for bk, idxs in sorted(idx_by_bucket.items()):
        b_res = evaluate_all(
            outputs=[outs[i] for i in idxs],
            dialect_refs=[gold[i] for i in idxs],
            standard_refs=[src[i] for i in idxs],
            dialect_eojeol_maps=[emaps[i] for i in idxs],
            target_do=target_do,
            classifier=classifier,
            cls_tokenizer=cls_tok,
        )
        bucket_results[bk] = b_res
    return bucket_results


def main(
    target_do: str = "gangwondo",
    n: int = 200,
    max_new_tokens: int = 64,
    n_show: int = 4,
    grpo_dir: str = GRPO_ADAPTER,
    base: str = BASE,
    classifier_path: str = CLS,
    cls_tokenizer_path: str = CLS_TOK,
    dataset_path: str = "outputs/datasets/grpo",
):
    """Compare SFT vs GRPO with stratified bucket breakdown and qualitative samples.

    Parameters
    ----------
    n_show:
        Number of qualitative samples to print *per bucket*.  The plan (§3 AC4)
        requires qualitative judgment per edit-distance × dialect stratum.
    """
    ds = load_from_disk(dataset_path)["valid"]
    ds = ds.filter(
        lambda x: (
            x["do"] == target_do and x["direction"] == "std2dia" and x["standard"] != x["dialect"]
        )
    )
    ds = ds.select(range(clamp_select_count(len(ds), n)))
    prompts = list(ds["prompt"])
    gold = list(ds["dialect"])
    src = list(ds["standard"])
    emaps = [m or [] for m in ds["dialect_eojeol_map"]]

    # Stratified bucket labels: dialect_region × edit_distance_bucket
    bucket_keys = [
        f"{_dialect_region(row['do'])}_{_edit_bucket(row['standard'], row['dialect'])}"
        for row in ds
    ]

    logger.info("Comparing on %d %s std2dia samples", len(ds), target_do)

    cls_tok = AutoTokenizer.from_pretrained(cls_tokenizer_path, local_files_only=True)
    if cls_tok.pad_token is None:
        cls_tok.pad_token = cls_tok.eos_token
    classifier = TextCNNForSequenceClassification.from_pretrained(classifier_path)

    results: dict[str, dict] = {}
    gens: dict[str, list[str]] = {}
    rev_outs: dict[str, list[str]] = {}

    for tag, adapter in [("SFT", None), ("GRPO", grpo_dir)]:
        logger.info("=== %s: loading + generating ===", tag)
        model, tok = load(base, adapter)
        device = next(model.parameters()).device
        classifier.to(device).eval()

        # Forward: standard → dialect
        outs = generate(model, tok, prompts, device, max_new_tokens=max_new_tokens)
        gens[tag] = outs

        # Reverse: model's dialect output → standard (reconstruction-BLEU)
        # Dual-RL / Luo et al. 2019: cycle consistency as content-preservation signal.
        rev_prompts = [_build_dia2std_prompt(o) for o in outs]
        rev = generate(model, tok, rev_prompts, device, max_new_tokens=max_new_tokens)
        rev_outs[tag] = rev

        results[tag] = evaluate_all(
            outputs=outs,
            dialect_refs=gold,
            standard_refs=src,
            dialect_eojeol_maps=emaps,
            target_do=target_do,
            classifier=classifier,
            cls_tokenizer=cls_tok,
            reverse_outputs=rev,
        )
        del model
        torch.cuda.empty_cache()

    # Gold upper-bound (no reverse pass needed — gold is already dialect)
    gold_scores = evaluate_all(
        outputs=gold,
        dialect_refs=gold,
        standard_refs=src,
        dialect_eojeol_maps=emaps,
        target_do=target_do,
        classifier=classifier,
        cls_tokenizer=cls_tok,
    )

    # -----------------------------------------------------------------------
    # Aggregate metrics table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 84)
    print(f"METRICS — {target_do}, std2dia, n={len(ds)}  (higher = better)")
    print("=" * 84)
    keys = [
        "tdr",
        "eojeol_accuracy",
        "chrf",
        "copy_margin",
        "reconstruction_bleu",
        "bleu",
        "jscore",
    ]
    print(f"{'metric':22s} {'SFT':>10s} {'GRPO':>10s} {'Δ(GRPO-SFT)':>14s} {'GOLD':>10s}")
    for k in keys:
        s = results["SFT"].get(k, float("nan"))
        g = results["GRPO"].get(k, float("nan"))
        gd = gold_scores.get(k, float("nan"))
        delta = g - s if not (s != s or g != g) else float("nan")  # nan check
        print(f"{k:22s} {s:>10.4f} {g:>10.4f} {delta:>+14.4f} {gd:>10.4f}")

    print(
        "\nNOTE: copy_margin = chrF(gen,gold) − chrF(gen,source).  "
        "Strips copy-bias for Gangwon (gold ≈ source)."
    )
    print(
        "NOTE: reconstruction_bleu = BLEU of dialect→std reverse generation vs source.  "
        "Proxy-independent selection signal (Dual-RL / Luo et al. 2019)."
    )
    print(
        "NOTE: jscore is MONITORING ONLY — not a selection criterion.  "
        "Plan §3 A4 / MO-GRPO arXiv:2509.22047: jscore overlaps with training rewards "
        "→ circularity.  Use reconstruction_bleu + qualitative gate (AC4)."
    )

    # -----------------------------------------------------------------------
    # Per-bucket aggregate breakdown
    # -----------------------------------------------------------------------
    print("\n" + "=" * 84)
    print("PER-BUCKET METRICS  (edit-distance bucket × dialect region)")
    print("=" * 84)
    for tag in ("SFT", "GRPO"):
        b_res = _bucket_metrics(
            gens[tag], gold, src, emaps, target_do, classifier, cls_tok, bucket_keys
        )
        print(f"\n  [{tag}]")
        print(f"  {'bucket':28s} {'n':>4s} {'tdr':>8s} {'chrf':>8s} {'copy_mg':>9s} {'eojeol':>8s}")
        for bk, br in sorted(b_res.items()):
            n_bk = bucket_keys.count(bk)
            print(
                f"  {bk:28s} {n_bk:>4d} {br['tdr']:>8.4f} {br['chrf']:>8.3f}"
                f" {br['copy_margin']:>+9.3f} {br['eojeol_accuracy']:>8.4f}"
            )

    # -----------------------------------------------------------------------
    # Qualitative samples — stratified by bucket (plan §3 AC4 protocol)
    # -----------------------------------------------------------------------
    # Group indices by bucket for stratified sampling
    idx_by_bucket: dict[str, list[int]] = defaultdict(list)
    for i, bk in enumerate(bucket_keys):
        idx_by_bucket[bk].append(i)

    print("\n" + "=" * 84)
    print(
        f"QUALITATIVE SAMPLES  (입력 표준어 → SFT / GRPO / 정답 방언)\n"
        f"  Stratified by edit-distance bucket × dialect region, {n_show} samples per bucket.\n"
        f"  Plan §3 AC4 qualitative gate: human/LLM-judge rubric —\n"
        f"    방언성 / 의미보존 / 자연스러움 / 과교정 (Mind the Style Gap arXiv:2502.15022)\n"
        f"  승률 ≥ 60 %% (GRPO vs SFT) required to pass the gate."
    )
    for bk in sorted(idx_by_bucket):
        idxs = idx_by_bucket[bk]
        print(f"\n--- bucket: {bk}  (n={len(idxs)}) ---")
        for rank, i in enumerate(idxs[:n_show], 1):
            ed = _lev_distance(src[i], gold[i])
            print(f"\n  [{rank}/{min(n_show, len(idxs))}] edit_dist={ed}")
            print(f"  표준어 : {src[i]}")
            print(f"  SFT   : {gens['SFT'][i]}")
            print(f"  GRPO  : {gens['GRPO'][i]}")
            print(f"  정답  : {gold[i]}")


if __name__ == "__main__":
    fire.Fire(main)
