#!/usr/bin/env python3
"""Stage 2: Train TextCNN dialect classifier (GRPO style reward model).

Class count is data-driven (``num_labels_for``): 3 for a gangwon/gyeongsang dataset,
6 for the full 5-region old_dialect dataset.
"""

from __future__ import annotations

import logging

import hydra
from datasets import load_from_disk
from omegaconf import DictConfig
from transformers import AutoTokenizer

from ko_dialect.data import ClassifierCollator
from ko_dialect.data.labels import num_labels_for
from ko_dialect.models import TextCNNConfig, TextCNNForSequenceClassification
from ko_dialect.tracking import setup_tracking
from ko_dialect.training import (
    ClassifierTrainerConfig,
    compute_class_weights,
    train_classifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="classifier", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cls_cfg = ClassifierTrainerConfig(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        early_stopping_patience=cfg.training.get("early_stopping_patience", 5),
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        seed=cfg.training.seed,
        report_to=list(cfg.logger.report_to),
    )

    logger.info("Loading classifier dataset from %s", cfg.data.cls_dataset_path)
    ds = load_from_disk(cfg.data.cls_dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve class weights from the train split per cfg.model.class_weighting.
    model_cfg_node = cfg.get("model", {})
    weighting = str(model_cfg_node.get("class_weighting", "none"))
    # Data-driven: a gangwon/gyeongsang-only dataset stays 3-class, while a full 5-region
    # (old_dialect) dataset becomes 6-class — no global constant to bump, and existing
    # 3-class checkpoints keep working when rebuilt from the same 2-region data.
    num_labels = num_labels_for(ds["train"]["label"])
    logger.info("Resolved num_labels=%d from training labels.", num_labels)
    class_weights = None
    if weighting == "manual":
        manual = model_cfg_node.get("class_weights")
        if manual is None:
            raise ValueError("class_weighting=manual requires model.class_weights to be set.")
        class_weights = list(manual)
    elif weighting in ("balanced", "inverse"):
        class_weights = compute_class_weights(ds["train"]["label"], num_labels, scheme=weighting)
    elif weighting != "none":
        raise ValueError(f"Unknown class_weighting={weighting!r}.")
    if class_weights is not None:
        logger.info("Class weights (%s): %s", weighting, class_weights)

    # NOTE: use len(tokenizer), not tokenizer.vocab_size — the latter excludes
    # the 22 added special tokens (e.g. pad/eos ids 151643+), so sizing the
    # embedding to vocab_size makes pad lookups go out of bounds → CUDA assert.
    model_cfg = TextCNNConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        num_labels=num_labels,
        class_weights=class_weights,
    )
    model = TextCNNForSequenceClassification(model_cfg)
    logger.info(
        "TextCNN params: %s",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    collator = ClassifierCollator(tokenizer=tokenizer, max_length=cfg.data.max_length)
    with setup_tracking(cfg.logger, cfg.experiment):
        train_classifier(
            cls_cfg,
            model,
            train_dataset=ds["train"],
            eval_dataset=ds.get("valid"),
            data_collator=collator,
        )


if __name__ == "__main__":
    main()
