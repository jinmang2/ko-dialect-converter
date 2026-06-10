#!/usr/bin/env python3
"""Did GRPO actually improve dialect conversion over the SFT base?

Generates greedily from the SFT base and from SFT+GRPO adapter on the *same* valid
prompts, then reports independent metrics (TDR via the classifier, chrF/BLEU vs gold
dialect, eojeol-level dialect accuracy) plus side-by-side qualitative samples. This is
the over-optimization check: GRPO reward (classifier prob) going up only matters if
these independent metrics move too.

    python scripts/compare_sft_grpo.py --target_do gangwondo --n 200
"""
from __future__ import annotations

import logging

import fire
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from ko_dialect.evaluation import evaluate_all
from ko_dialect.models import TextCNNForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare")

BASE = "outputs/sft_merged"
GRPO_ADAPTER = "outputs/grpo_500"
CLS = "outputs/classifier_clean"
CLS_TOK = "Qwen/Qwen2.5-0.5B-Instruct"


@torch.no_grad()
def generate(model, tok, prompts, device, batch_size=16, max_new_tokens=64):
    prev = tok.padding_side
    tok.padding_side = "left"
    out = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=448).to(device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1] :]
            out.extend(t.strip() for t in tok.batch_decode(new, skip_special_tokens=True))
    finally:
        tok.padding_side = prev
    return out


def load(adapter: str | None):
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
                                                 device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tok


def main(target_do: str = "gangwondo", n: int = 200, max_new_tokens: int = 64,
         n_show: int = 12):
    ds = load_from_disk("outputs/datasets/grpo")["valid"]
    ds = ds.filter(lambda x: x["do"] == target_do and x["direction"] == "std2dia"
                   and x["standard"] != x["dialect"])
    ds = ds.select(range(min(n, len(ds))))
    prompts = list(ds["prompt"])
    gold = list(ds["dialect"])
    src = list(ds["standard"])
    emaps = [m or [] for m in ds["dialect_eojeol_map"]]
    logger.info("Comparing on %d %s std2dia samples", len(ds), target_do)

    cls_tok = AutoTokenizer.from_pretrained(CLS_TOK)
    if cls_tok.pad_token is None:
        cls_tok.pad_token = cls_tok.eos_token
    classifier = TextCNNForSequenceClassification.from_pretrained(CLS)

    results = {}
    gens = {}
    for tag, adapter in [("SFT", None), ("GRPO", GRPO_ADAPTER)]:
        logger.info("=== %s: loading + generating ===", tag)
        model, tok = load(adapter)
        device = next(model.parameters()).device
        classifier.to(device).eval()
        outs = generate(model, tok, prompts, device, max_new_tokens=max_new_tokens)
        gens[tag] = outs
        results[tag] = evaluate_all(
            outputs=outs, dialect_refs=gold, standard_refs=src,
            dialect_eojeol_maps=emaps, target_do=target_do,
            classifier=classifier, cls_tokenizer=cls_tok,
        )
        # also score the GOLD dialect as an upper bound (once)
        del model
        torch.cuda.empty_cache()

    gold_scores = evaluate_all(
        outputs=gold, dialect_refs=gold, standard_refs=src,
        dialect_eojeol_maps=emaps, target_do=target_do,
        classifier=classifier, cls_tokenizer=cls_tok,
    )

    print("\n" + "=" * 72)
    print(f"METRICS — {target_do}, std2dia, n={len(ds)}  (higher = better)")
    print("=" * 72)
    keys = ["tdr", "eojeol_accuracy", "chrf", "bleu"]
    print(f"{'metric':18s} {'SFT':>10s} {'GRPO':>10s} {'Δ(GRPO-SFT)':>14s} {'GOLD':>10s}")
    for k in keys:
        s, g, gd = results["SFT"].get(k, 0), results["GRPO"].get(k, 0), gold_scores.get(k, 0)
        print(f"{k:18s} {s:>10.4f} {g:>10.4f} {g - s:>+14.4f} {gd:>10.4f}")

    print("\n" + "=" * 72)
    print("QUALITATIVE SAMPLES  (입력 표준어 → SFT / GRPO / 정답 방언)")
    print("=" * 72)
    for i in range(min(n_show, len(ds))):
        print(f"\n[{i}] 표준어 : {src[i]}")
        print(f"    SFT  : {gens['SFT'][i]}")
        print(f"    GRPO : {gens['GRPO'][i]}")
        print(f"    정답 : {gold[i]}")


if __name__ == "__main__":
    fire.Fire(main)
