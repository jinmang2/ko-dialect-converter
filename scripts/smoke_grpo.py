#!/usr/bin/env python3
"""Fast GRPO smoke test — surfaces runtime errors without a full run.

Runs a couple of optimizer steps on a tiny dataset slice with the 6GB-safe config
(Unsloth 16-bit LoRA, no vLLM). Use this to iterate on env/API/reward-signature
issues before launching the real stage3_grpo.py.

    python scripts/smoke_grpo.py
"""

from __future__ import annotations

# Unsloth must be imported before transformers/trl so its kernel patches apply.
import unsloth  # noqa: F401  isort:skip

import logging
import os

from datasets import load_from_disk
from transformers import AutoTokenizer

from ko_dialect.models import TextCNNForSequenceClassification
from ko_dialect.training import GRPOConfig, build_reward_fns, train_grpo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke_grpo")

GRPO_DATASET = "outputs/datasets/grpo"
CLASSIFIER = "outputs/classifier_clean"
BASE_MODEL = "outputs/sft_merged"
CLS_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
N_SAMPLES = int(os.environ.get("SMOKE_N", "16"))
LOAD_4BIT = os.environ.get("SMOKE_4BIT", "0") == "1"
MAX_STEPS = int(os.environ.get("SMOKE_STEPS", "2"))


def main() -> None:
    cfg = GRPOConfig(
        base_model_path=BASE_MODEL,
        backend="unsloth",
        dtype="fp16",
        load_in_4bit=LOAD_4BIT,
        apply_lora=True,
        use_vllm=False,
        output_dir="outputs/grpo_smoke",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,
        beta=0.1,
        epsilon_high=0.28,
        mask_truncated_completions=True,
        max_new_tokens=64,
        max_seq_length=512,  # prompt+completion budget; prompts have a tail to ~520 toks
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_steps=10_000,  # don't checkpoint during smoke
        report_to=[],
        aggregation=os.environ.get("SMOKE_AGGREGATION", "mo_grpo"),
        rewards=[
            {"name": "style", "weight": 0.5},
            {"name": "copy_margin", "weight": 1.0},
            {"name": "edit_precision", "weight": 0.5},
            {"name": "edit_recall", "weight": 0.5},
            {"name": "overcorrection", "weight": 0.5},
        ],
        normalize_rewards=True,
    )

    logger.info("Loading dataset slice (%d rows) from %s", N_SAMPLES, GRPO_DATASET)
    ds = load_from_disk(GRPO_DATASET)
    train_ds = ds["train"].select(range(min(N_SAMPLES, len(ds["train"]))))

    logger.info("Loading classifier from %s", CLASSIFIER)
    cls_tok = AutoTokenizer.from_pretrained(CLS_TOKENIZER)
    if cls_tok.pad_token is None:
        cls_tok.pad_token = cls_tok.eos_token
    classifier = TextCNNForSequenceClassification.from_pretrained(CLASSIFIER)
    classifier.eval()

    reward_fns, reward_weights = build_reward_fns(cfg, classifier, cls_tok)
    logger.info("Built %d reward fns, weights=%s", len(reward_fns), reward_weights)

    train_grpo(cfg, train_ds, reward_fns, reward_weights=reward_weights)
    logger.info("SMOKE OK")


if __name__ == "__main__":
    main()
