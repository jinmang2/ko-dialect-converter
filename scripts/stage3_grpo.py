#!/usr/bin/env python3
"""Stage 3: GRPO fine-tuning using TextCNN classifier as reward signal."""
from __future__ import annotations

# Unsloth must be imported before transformers/trl/peft so its mixed-precision and
# kernel patches apply. With the unsloth backend, importing it late leaves GRPO grads
# in bf16 and the fp16 grad-scaler then dies on Turing with:
#   NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
#                        not implemented for 'BFloat16'
# The import is wrapped so non-unsloth backends (or boxes without unsloth) still run.
try:  # noqa: SIM105
    import unsloth  # noqa: F401  isort:skip
except Exception:  # pragma: no cover - unsloth optional for hf/bnb backends
    pass

import logging

import hydra
from datasets import load_from_disk
from omegaconf import DictConfig
from transformers import AutoTokenizer

from ko_dialect.config_utils import from_omegaconf
from ko_dialect.models import TextCNNForSequenceClassification
from ko_dialect.tracking import setup_tracking
from ko_dialect.training import GRPOConfig, build_reward_fns, train_grpo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def filter_long_prompts(dataset, tokenizer, max_prompt_tokens: int):
    """Drop rows whose tokenized prompt exceeds ``max_prompt_tokens``.

    trl 1.5.1 removed ``GRPOConfig.max_prompt_length``, so prompts are no longer
    truncated. A prompt longer than ``max_seq_length - max_new_tokens`` overflows the
    model's positional buffer and unsloth crashes with a 4-D attention-mask size
    mismatch (``tensor a (max_seq_length) must match tensor b (actual_len)``). The
    dialect prompts are short (median ~73 tok) with a thin tail to ~520, so filtering
    the outliers costs ~0.1% of data and keeps every sequence within budget.
    """
    if dataset is None:
        return None
    before = len(dataset)
    lengths = dataset.map(
        lambda b: {"_plen": [len(x) for x in tokenizer(b["prompt"], add_special_tokens=False)["input_ids"]]},
        batched=True,
        desc="measuring prompt lengths",
    )
    kept = lengths.filter(lambda r: r["_plen"] <= max_prompt_tokens).remove_columns("_plen")
    logger.info(
        "Prompt filter (<= %d toks): kept %d / %d rows (dropped %d)",
        max_prompt_tokens, len(kept), before, before - len(kept),
    )
    return kept


@hydra.main(config_path="../configs", config_name="grpo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    grpo_cfg = from_omegaconf(
        GRPOConfig,
        cfg.training,
        base_model_path=cfg.data.base_model_path,
        report_to=list(cfg.logger.report_to),
        remove_unused_columns=False,
    )

    logger.info("Loading reward-model tokenizer %s", cfg.data.cls_tokenizer_name)
    cls_tokenizer = AutoTokenizer.from_pretrained(cfg.data.cls_tokenizer_name)
    if cls_tokenizer.pad_token is None:
        cls_tokenizer.pad_token = cls_tokenizer.eos_token

    logger.info("Loading GRPO dataset from %s", cfg.data.grpo_dataset_path)
    ds = load_from_disk(cfg.data.grpo_dataset_path)
    # Keep prompt + generated completion within the positional budget (see helper docstring).
    margin = 8
    max_prompt_tokens = grpo_cfg.max_seq_length - grpo_cfg.max_new_tokens - margin
    train_ds = filter_long_prompts(ds["train"], cls_tokenizer, max_prompt_tokens)
    eval_ds = filter_long_prompts(ds.get("valid"), cls_tokenizer, max_prompt_tokens)

    logger.info("Loading classifier (reward model) from %s", cfg.data.classifier_path)
    classifier = TextCNNForSequenceClassification.from_pretrained(cfg.data.classifier_path)
    classifier.eval()

    reward_fns, reward_weights = build_reward_fns(grpo_cfg, classifier, cls_tokenizer)
    with setup_tracking(cfg.logger, cfg.experiment):
        logger.info("GRPOConfig: %s", grpo_cfg)
        train_grpo(
            grpo_cfg, train_ds, reward_fns, reward_weights=reward_weights, eval_dataset=eval_ds
        )


if __name__ == "__main__":
    main()
