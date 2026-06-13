from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

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
    # Select on macro-F1, not accuracy: this model is a GRPO reward, so balanced
    # per-class separation matters more than overall accuracy (which the majority
    # `standard` class would dominate). Mirrors the DIA-REFINE paper's headline metric.
    metric_for_best_model: str = "f1_macro"
    greater_is_better: bool = True
    # Stop when the selection metric stops improving for this many evals. Makes runs
    # self-terminating (the previous run was killed at step 1800 / epoch 0.27 by hand)
    # and guards against overtraining. None disables.
    early_stopping_patience: int | None = 5
    fp16: bool = True
    bf16: bool = False
    seed: int = 42
    report_to: list[str] = field(default_factory=list)


def compute_class_weights(
    labels,
    num_labels: int,
    scheme: str = "balanced",
) -> list[float]:
    """Derive per-class CrossEntropyLoss weights from a label sequence.

    - ``"balanced"`` (sklearn convention): ``n_samples / (num_labels * count[c])``
      — mean weight ≈ 1, rare classes weighted up.
    - ``"inverse"``: ``1 / count[c]``, renormalized so the weights sum to
      ``num_labels`` (same scale as ``balanced`` but sharper on rare classes).

    Empty classes are treated as count 1 to avoid division by zero.
    """
    counts = np.bincount(np.asarray(labels), minlength=num_labels).astype(np.float64)
    counts[counts == 0] = 1.0
    n = counts.sum()
    if scheme == "balanced":
        w = n / (num_labels * counts)
    elif scheme == "inverse":
        w = 1.0 / counts
        w = w * num_labels / w.sum()
    else:
        raise ValueError(f"scheme must be 'balanced' or 'inverse', got {scheme!r}.")
    return w.tolist()


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

    callbacks = []
    if cfg.early_stopping_patience is not None and eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    logger.info("Classifier saved to %s", cfg.output_dir)
