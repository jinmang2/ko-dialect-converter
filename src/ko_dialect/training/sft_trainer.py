from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from ko_dialect.data.template import get_template
from ko_dialect.models.loading import (
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_MODEL,
    BackendConfig,
    load_backbone,
    save_merged_16bit,
)

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    model_name: str = DEFAULT_MODEL
    max_seq_length: int = 256
    # Backend / precision (RTX 2060-safe defaults; flip for Colab)
    backend: str = "unsloth"  # unsloth | bnb | hf | loftq | awq | gptq | qat
    dtype: str = "fp16"  # fp16 | bf16 | fp32
    load_in_4bit: bool = True
    quantized_model_path: str | None = None  # awq/gptq backends
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES)
    )
    # Data / loss shaping
    template_name: str = "default"
    output_mode: str = "text"  # "text" (pre-rendered) | "structured" (render at train time)
    loss_on: str = "all"  # "all" | "completion" | "assistant" (structured mode only)
    packing: bool = True
    # Trainer
    output_dir: str = "outputs/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    # Eval has no backward pass, so it can use a larger batch than training.
    # Was unset (HF default 8) and ran slow; expose it so it can be tuned.
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    save_steps: int = 500
    logging_steps: int = 50
    fp16: bool = True
    bf16: bool = False  # RTX 2060 (Turing SM 7.5) has no native BF16
    dataloader_num_workers: int = 0
    seed: int = 42
    num_train_samples: int | None = None
    num_eval_samples: int | None = None
    report_to: list[str] = field(default_factory=list)
    eval_strategy: str = "epoch"  # "no" | "steps" | "epoch"
    eval_steps: int = 500
    save_strategy: str = "steps"  # "no" | "steps" | "epoch"
    # --- early stopping (opt-in; off by default to preserve existing runs) ---
    # When early_stopping_patience is set, training stops after that many evals without
    # improvement in metric_for_best_model, and (with load_best_model_at_end) the best
    # checkpoint is restored before saving. Needs an eval_dataset.
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int | None = None
    save_merged: bool = True  # also save full 16-bit weights so GRPO can load directly
    # --- training-optimization knobs ---
    # optim: "adamw_torch" (default) or "paged_adamw_8bit" to shrink optimizer state on 6GB.
    optim: str = "adamw_torch"
    # gradient_checkpointing trades recompute for activation memory (unsloth manages its own).
    gradient_checkpointing: bool = False
    # deepspeed: path to a ZeRO config json. Use with a non-unsloth backend (bnb/hf) — unsloth
    # patches the model and does not compose with the DeepSpeed engine. Benchmark vs unsloth.
    deepspeed: str | None = None

    def to_backend_config(self) -> BackendConfig:
        return BackendConfig(
            backend=self.backend,
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            quantized_model_path=self.quantized_model_path,
            apply_lora=True,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_target_modules=self.lora_target_modules,
            seed=self.seed,
        )


def load_model_and_tokenizer(cfg: SFTConfig):
    """Load the SFT backbone via the pluggable backend factory."""
    return load_backbone(cfg.to_backend_config())


def _preprocess_logits_for_metrics(logits, labels):
    """Reduce (B, T, vocab) → (B, T, 2) to avoid OOM during eval.

    Channel 0: argmax token id (float)
    Channel 1: per-position Shannon entropy
    """
    probs = torch.softmax(logits.float(), dim=-1)
    argmax = logits.argmax(dim=-1).float()
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    return torch.stack([argmax, entropy], dim=-1)


def _compute_metrics(eval_pred):
    """Token-level accuracy and mean entropy on supervised positions (labels != -100).

    With ``loss_on="all"`` every non-pad position is supervised; with completion/
    assistant masking the prompt positions are -100 and excluded here.
    """
    import numpy as np

    preds, labels = eval_pred  # preds: (N, T, 2), labels: (N, T)
    # logits[i]는 token[i+1]을 예측 → 정렬 맞추기
    argmax = np.round(preds[:, :-1, 0]).astype(np.int64)
    entropy = preds[:, :-1, 1]
    labels = labels[:, 1:]
    mask = labels != -100
    accuracy = float((argmax[mask] == labels[mask]).mean())
    mean_entropy = float(entropy[mask].mean())
    return {"token_accuracy": accuracy, "mean_entropy": mean_entropy}


def _render_structured(dataset, tokenizer, template, loss_on: str):
    """Materialise structured rows (source/target/do/direction) into the columns TRL
    expects for the requested loss-masking strategy.

    - "all"        → ``text`` (full rendered chat, supervised everywhere)
    - "completion" → ``prompt`` + ``completion`` (TRL masks the prompt automatically)
    - "assistant"  → ``messages`` (paired with ``assistant_only_loss=True``)
    """
    cols = dataset.column_names

    if loss_on == "assistant":

        def _to(ex):
            return {
                "messages": template.format_messages(
                    ex["source"], ex["target"], ex["do"], ex["direction"]
                )
            }

        return dataset.map(_to, remove_columns=cols)

    if loss_on == "completion":

        def _to(ex):
            return {
                "prompt": template.build_prompt(tokenizer, ex["source"], ex["do"], ex["direction"]),
                "completion": ex["target"],
            }

        return dataset.map(_to, remove_columns=cols)

    # loss_on == "all"
    def _to(ex):
        return {
            "text": template.apply(tokenizer, ex["source"], ex["target"], ex["do"], ex["direction"])
        }

    return dataset.map(_to, remove_columns=cols)


def train(cfg: SFTConfig, train_dataset, eval_dataset=None) -> None:
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    # Workaround for a TRL bug: base_trainer.create_model_card references `wandb` by
    # name without importing it, raising NameError on checkpoint save even when
    # report_to="none". Inject the module into TRL's namespace if wandb is installed.
    try:
        import trl.trainer.base_trainer as _trl_base

        if not hasattr(_trl_base, "wandb"):
            import wandb as _wandb

            _trl_base.wandb = _wandb
    except ImportError:
        pass

    model, tokenizer = load_model_and_tokenizer(cfg)

    # Cast any stray bf16 tensors to fp16 (Turing has no native BF16).
    if cfg.fp16 and not cfg.bf16:
        for param in model.parameters():
            if param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float16)

    # Render structured datasets into the columns TRL needs for the chosen loss mode.
    packing = cfg.packing
    assistant_only_loss = False
    sft_kwargs: dict = {}
    if cfg.output_mode == "structured":
        template = get_template(cfg.template_name)
        train_dataset = _render_structured(train_dataset, tokenizer, template, cfg.loss_on)
        if eval_dataset is not None:
            eval_dataset = _render_structured(eval_dataset, tokenizer, template, cfg.loss_on)
        if cfg.loss_on == "all":
            sft_kwargs["dataset_text_field"] = "text"
        elif cfg.loss_on == "assistant":
            assistant_only_loss = True
            packing = False  # masking is simplest/portable without packing
        else:  # completion
            packing = False  # completion-only loss is incompatible with packing
    else:
        if cfg.loss_on != "all":
            logger.warning(
                "loss_on=%s requires output_mode=structured; ignoring for text mode.",
                cfg.loss_on,
            )
        sft_kwargs["dataset_text_field"] = "text"

    # Early stopping / best-checkpoint restore needs an eval set; reconcile save↔eval
    # strategies the way the classifier trainer does (HF requires them to match for
    # load_best_model_at_end, and save_steps to be a multiple of eval_steps).
    use_best = cfg.load_best_model_at_end and eval_dataset is not None
    eval_strategy = cfg.eval_strategy if eval_dataset is not None else "no"
    save_strategy = eval_strategy if use_best else cfg.save_strategy
    save_steps = cfg.eval_steps if save_strategy == "steps" else cfg.save_steps

    # Sanity-warn on a likely metric/direction mismatch: a "lower is better" metric (loss/
    # perplexity) with greater_is_better=True, or a "higher is better" one (accuracy/f1/
    # bleu/chrf) with greater_is_better=False would pick the *worst* checkpoint.
    if cfg.early_stopping_patience is not None or use_best:
        lower = any(k in cfg.metric_for_best_model.lower() for k in ("loss", "perplex", "ppl"))
        higher = any(
            k in cfg.metric_for_best_model.lower() for k in ("acc", "f1", "bleu", "chrf", "score")
        )
        if (lower and cfg.greater_is_better) or (higher and not cfg.greater_is_better):
            logger.warning(
                "metric_for_best_model=%r with greater_is_better=%s looks inverted — "
                "this may select the worst checkpoint. Check the direction.",
                cfg.metric_for_best_model,
                cfg.greater_is_better,
            )

    training_args = TRLSFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        save_steps=save_steps,
        save_strategy=save_strategy,
        logging_steps=cfg.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=use_best,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        optim=cfg.optim,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
        max_length=cfg.max_seq_length,
        packing=packing,
        packing_strategy="bfd",  # Best Fit Decreasing
        assistant_only_loss=assistant_only_loss,
        seed=cfg.seed,
        report_to=cfg.report_to if cfg.report_to else "none",
        # DeepSpeed only when a config is provided (else keep the plain single-GPU path).
        **({"deepspeed": cfg.deepspeed} if cfg.deepspeed else {}),
        **sft_kwargs,
    )

    callbacks = []
    if cfg.early_stopping_patience is not None and eval_dataset is not None:
        from transformers import EarlyStoppingCallback

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))
        logger.info("Early stopping enabled (patience=%d)", cfg.early_stopping_patience)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=callbacks or None,
        compute_metrics=_compute_metrics if eval_dataset is not None else None,
        preprocess_logits_for_metrics=(
            _preprocess_logits_for_metrics if eval_dataset is not None else None
        ),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("SFT adapter saved to %s", cfg.output_dir)

    # Write standalone 16-bit weights so stage3 GRPO loads the *trained* model, not the
    # raw base (the historical adapter-dir loading bug).
    if cfg.save_merged:
        try:
            save_merged_16bit(model, tokenizer, cfg.output_dir)
        except Exception as exc:  # pragma: no cover - best-effort, non-fatal
            logger.warning("Could not save merged 16-bit weights: %s", exc)
