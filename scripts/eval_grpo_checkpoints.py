#!/usr/bin/env python3
"""Sweep GRPO checkpoints to find the over-optimization point and pick the best one.

GRPO maximizes the classifier reward (TDR), but the classifier is a hackable proxy:
reward can keep rising while the *real* translation fidelity (chrF/BLEU vs gold dialect)
degrades. This evaluates the SFT base and every grpo checkpoint-* on a fixed valid
subset and prints TDR / chrF / BLEU / eojeol-accuracy per step, so you can:
  - see the over-optimization curve (chrF peaks early, then drops as TDR keeps rising)
  - select the best checkpoint by an *independent* metric (default: chrf)

    python scripts/eval_grpo_checkpoints.py --grpo_dir outputs/grpo_500 --n 150
"""
from __future__ import annotations

import glob
import logging
import os
import re

import fire
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from ko_dialect.evaluation import evaluate_all
from ko_dialect.models import TextCNNForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sweep")

BASE = "outputs/sft_merged"
CLS = "outputs/classifier_clean"
CLS_TOK = "Qwen/Qwen2.5-0.5B-Instruct"


@torch.no_grad()
def generate(model, tok, prompts, device, batch_size=16, max_new_tokens=64):
    prev = tok.padding_side
    tok.padding_side = "left"
    out = []
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=448).to(device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1] :]
            out.extend(t.strip() for t in tok.batch_decode(new, skip_special_tokens=True))
    finally:
        tok.padding_side = prev
    return out


def eval_one(adapter, prompts, gold, src, emaps, target_do, classifier, cls_tok):
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16,
                                                 device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    device = next(model.parameters()).device
    classifier.to(device).eval()
    outs = generate(model, tok, prompts, device)
    res = evaluate_all(outputs=outs, dialect_refs=gold, standard_refs=src,
                       dialect_eojeol_maps=emaps, target_do=target_do,
                       classifier=classifier, cls_tokenizer=cls_tok)
    del model
    torch.cuda.empty_cache()
    return res


def main(grpo_dir: str = "outputs/grpo_500", target_do: str = "gangwondo",
         n: int = 150, select_by: str = "chrf"):
    ds = load_from_disk("outputs/datasets/grpo")["valid"]
    ds = ds.filter(lambda x: x["do"] == target_do and x["direction"] == "std2dia"
                   and x["standard"] != x["dialect"]).select(range(n))
    prompts, gold = list(ds["prompt"]), list(ds["dialect"])
    src, emaps = list(ds["standard"]), [m or [] for m in ds["dialect_eojeol_map"]]

    cls_tok = AutoTokenizer.from_pretrained(CLS_TOK)
    if cls_tok.pad_token is None:
        cls_tok.pad_token = cls_tok.eos_token
    classifier = TextCNNForSequenceClassification.from_pretrained(CLS)

    ckpts = sorted(glob.glob(os.path.join(grpo_dir, "checkpoint-*")),
                   key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    def _step(p):
        return re.search(r"checkpoint-(\d+)", p).group(1)

    stages = [("SFT(0)", None)] + [("step-" + _step(c), c) for c in ckpts]
    # also the final adapter at the root, if present
    if os.path.exists(os.path.join(grpo_dir, "adapter_config.json")):
        stages.append(("final", grpo_dir))

    print(f"\nSweep: {target_do} std2dia, n={len(ds)}, select_by={select_by}")
    print(f"{'stage':10s} {'tdr':>8s} {'chrf':>8s} {'bleu':>8s} {'eojeol':>8s}")
    rows = []
    for tag, adapter in stages:
        r = eval_one(adapter, prompts, gold, src, emaps, target_do, classifier, cls_tok)
        rows.append((tag, r))
        print(f"{tag:10s} {r['tdr']:>8.4f} {r['chrf']:>8.3f} {r['bleu']:>8.3f} {r['eojeol_accuracy']:>8.4f}")

    best = max(rows, key=lambda tr: tr[1].get(select_by, 0))
    print(f"\nBest by {select_by}: {best[0]}  ({select_by}={best[1][select_by]:.3f}, "
          f"tdr={best[1]['tdr']:.3f})")
    print("NOTE: if chrf peaks before the last step while tdr keeps rising, that gap is "
          "the over-optimization — select the chrf-best checkpoint, not the final one.")


if __name__ == "__main__":
    fire.Fire(main)
