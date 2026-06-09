from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class ClassifierTrainerConfig:
    output_dir: str = "outputs/classifier"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    fp16: bool = True
    bf16: bool = False
    seed: int = 42
    report_to: list[str] = field(default_factory=list)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def train(
    cfg: ClassifierTrainerConfig,
    model,
    train_dataset,
    eval_dataset=None,
    data_collator=None,
) -> None:
    # load_best_model_at_end requires the save and eval strategies (and, for
    # "steps", their intervals) to match. Force save to follow eval so toggling
    # eval_strategy between "epoch"/"steps" can't raise a strategy-mismatch error.
    save_strategy = cfg.eval_strategy if cfg.load_best_model_at_end else cfg.save_strategy
    save_steps = cfg.eval_steps if save_strategy == "steps" else cfg.save_steps

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        save_steps=save_steps,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        seed=cfg.seed,
        remove_unused_columns=False,
        report_to=cfg.report_to if cfg.report_to else "none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    logger.info("Classifier saved to %s", cfg.output_dir)
