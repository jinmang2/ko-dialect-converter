#!/usr/bin/env python3
"""A/B: does prosody supervision improve dialect conversion vs a matched control?

Both models are full SFT models trained on identical rows; the only difference is whether
the dialect side carried prosody markers (<UP>/<DOWN>/<KEEP>) during training. The prosody
model emits those markers, so we **strip them before scoring** — the comparison is about
dialect content, not the marker tokens. Metrics reuse the leaderboard machinery
(reconstruction_bleu↑, copy_margin↑, tdr↑, eojeol↑) plus paired-bootstrap significance
(Koehn 2004) on per-sentence reconstruction BLEU.

Usage:
    python scripts/eval_prosody_ab.py \
        --control outputs/sft_control --prosody outputs/sft_prosody \
        --target_do gangwondo --n 150
"""

from __future__ import annotations

import logging

import fire
import torch
from datasets import load_from_disk
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer

from ko_dialect.data.prosody import strip_markers
from ko_dialect.evaluation import grpo_runs
from ko_dialect.evaluation.generation import generate_batched
from ko_dialect.evaluation.leaderboard import RunSpec, evaluate_run
from ko_dialect.evaluation.metric_registry import header_label
from ko_dialect.evaluation.significance import PairedResult, paired_bootstrap, significance_marker
from ko_dialect.models import TextCNNForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prosody_ab")

CLS = "outputs/classifier_clean"
SHOW = ("reconstruction_bleu", "copy_margin", "tdr", "dfs", "chrf", "eojeol_accuracy")


def _make_generate_fn(device, *, strip: bool, batch_size=16, max_new_tokens=64):
    """Greedy generation; when ``strip`` remove prosody markers from each output."""

    def generate_fn(model, tok, prompts):
        return generate_batched(
            model,
            tok,
            prompts,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            max_length=448,
            post=strip_markers if strip else None,
        )

    return generate_fn


def main(
    control: str = "outputs/sft_control",
    prosody: str = "outputs/sft_prosody",
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    target_do: str = "gangwondo",
    n: int = 150,
    classifier_path: str = CLS,
    dataset_path: str = "outputs/datasets/grpo",
    max_new_tokens: int = 64,
):
    """Compare the control and prosody SFT models on a held-out region slice."""
    ds = load_from_disk(dataset_path)["valid"]
    ds = ds.filter(
        lambda x: (
            x["do"] == target_do and x["direction"] == "std2dia" and x["standard"] != x["dialect"]
        )
    )
    ds = ds.select(range(grpo_runs.clamp_select_count(len(ds), n)))
    prompts, gold = list(ds["prompt"]), list(ds["dialect"])
    src = list(ds["standard"])
    emaps = [m or [] for m in ds["dialect_eojeol_map"]]
    logger.info("A/B on %d %s std2dia samples", len(ds), target_do)

    cls_tok = AutoTokenizer.from_pretrained(classifier_path, local_files_only=True)
    if cls_tok.pad_token is None:
        cls_tok.pad_token = cls_tok.eos_token
    classifier = TextCNNForSequenceClassification.from_pretrained(classifier_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sent_bleu = BLEU(effective_order=True)

    rows: dict[str, dict] = {}
    recon: dict[str, list[float]] = {}
    for tag, adapter, strip in (("control", control, True), ("prosody", prosody, True)):
        # Adapter dirs (LoRA on the fp16 base) — evaluate_run merges in memory, so no
        # save_merged step is needed. Tokenizer comes from the adapter dir (trainer saved it).
        model_tok = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        if model_tok.pad_token is None:
            model_tok.pad_token = model_tok.eos_token
        gen_fn = _make_generate_fn(device, strip=strip, max_new_tokens=max_new_tokens)
        metrics, _outs, rev = evaluate_run(
            RunSpec(tag, base_model, adapter),
            prompts=prompts,
            gold=gold,
            src=src,
            emaps=emaps,
            target_do=target_do,
            classifier=classifier,
            cls_tokenizer=cls_tok,
            model_tokenizer=model_tok,
            generate_fn=gen_fn,
        )
        rows[tag] = metrics
        recon[tag] = [sent_bleu.sentence_score(r, [s]).score for r, s in zip(rev, src)]

    print("\n" + "=" * 76)
    print(f"PROSODY A/B — {target_do} std2dia, n={len(ds)}  (markers stripped before scoring)")
    print("=" * 76)
    hdr = f"{'metric':24s} {'control':>10s} {'prosody':>10s} {'Δ(pros-ctrl)':>14s}"
    print(hdr)
    for k in SHOW:
        c, p = rows["control"].get(k, float("nan")), rows["prosody"].get(k, float("nan"))
        print(f"{header_label(k):24s} {c:>10.3f} {p:>10.3f} {p - c:>+14.3f}")

    res = paired_bootstrap(recon["prosody"], recon["control"])
    marker = significance_marker(PairedResult(res.delta, res.p_value, res.win_rate, res.n_boot))
    print(
        f"\nPaired bootstrap (Koehn 2004), per-sentence recon BLEU, prosody vs control:\n"
        f"  Δ={res.delta:+.2f}  p={res.p_value:.3f}  {marker} "
        + ("(significant)" if res.significant_05 else "(n.s.)")
    )
    print(
        "\nVerdict guide: prosody helps if copy_margin↑ / recon_bleu↑ move positive and "
        "the recon delta is significant; a wash means markers add no dialect-content gain."
    )


if __name__ == "__main__":
    fire.Fire(main)
