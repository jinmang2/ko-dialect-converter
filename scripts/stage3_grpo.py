#!/usr/bin/env python3
"""Stage 3: GRPO fine-tuning using TextCNN classifier as reward signal."""
from __future__ import annotations

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


@hydra.main(config_path="../configs", config_name="grpo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    grpo_cfg = from_omegaconf(
        GRPOConfig,
        cfg.training,
        base_model_path=cfg.data.base_model_path,
        report_to=list(cfg.logger.report_to),
        remove_unused_columns=False,
    )

    logger.info("Loading GRPO dataset from %s", cfg.data.grpo_dataset_path)
    ds = load_from_disk(cfg.data.grpo_dataset_path)
    train_ds = ds["train"]
    eval_ds = ds.get("valid")

    logger.info("Loading classifier from %s", cfg.data.classifier_path)
    cls_tokenizer = AutoTokenizer.from_pretrained(cfg.data.cls_tokenizer_name)
    if cls_tokenizer.pad_token is None:
        cls_tokenizer.pad_token = cls_tokenizer.eos_token

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
