from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from ko_dialect.models.loading import (
    DEFAULT_LORA_TARGET_MODULES,
    BackendConfig,
    load_backbone,
)

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    # stage1 SFT writes merged 16-bit weights here (see SFTConfig.save_merged), so this
    # is a real standalone model dir, not a bare adapter — the factory can load it directly.
    base_model_path: str = "outputs/sft"
    max_seq_length: int = 256
    # Backend / precision. Defaults stay 2060-safe (hf fp16, no vLLM); flip for Colab.
    backend: str = "hf"  # hf | unsloth | bnb | loftq | awq | gptq
    dtype: str = "fp16"
    load_in_4bit: bool = False  # set True with backend=unsloth/bnb to fit bigger runs
    use_vllm: bool = False  # unsloth fast_inference rollouts (Ampere+/Colab)
    gpu_memory_utilization: float = 0.5
    # LoRA — only applied for trainable-backbone backends (unsloth/bnb/loftq/awq/gptq).
    apply_lora: bool = False  # hf path defaults to full fine-tune (historical behavior)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES)
    )
    quantized_model_path: str | None = None
    # Trainer
    output_dir: str = "outputs/grpo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-6
    num_generations: int = 4
    beta: float = 0.04
    max_new_tokens: int = 128
    max_prompt_length: int = 256
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 20
    save_steps: int = 200
    seed: int = 42
    report_to: list[str] = field(default_factory=list)
    # Reward spec list: [{name: style, weight: 1.0}, {name: content, weight: 0.5}, ...].
    rewards: list[dict[str, Any]] | None = None
    use_edit_reward: bool = False  # back-compat shim: appends {edit, 0.3} if rewards unset
    # Must be False so extra dataset columns (do, standard, dialect, ...) reach reward_fns
    remove_unused_columns: bool = False

    def to_backend_config(self) -> BackendConfig:
        return BackendConfig(
            backend=self.backend,
            model_name=self.base_model_path,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            use_vllm=self.use_vllm,
            gpu_memory_utilization=self.gpu_memory_utilization,
            apply_lora=self.apply_lora,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_target_modules=self.lora_target_modules,
            quantized_model_path=self.quantized_model_path,
            seed=self.seed,
        )

    def reward_specs(self) -> list[dict[str, Any]] | None:
        """Resolve the reward spec list, honoring the legacy ``use_edit_reward`` flag."""
        if self.rewards:
            return self.rewards
        if self.use_edit_reward:
            from ko_dialect.rewards import DEFAULT_REWARDS

            return [*DEFAULT_REWARDS, {"name": "edit", "weight": 0.3}]
        return None  # registry falls back to DEFAULT_REWARDS (style + content)


def build_reward_fns(
    cfg: GRPOConfig, classifier, cls_tokenizer
) -> tuple[list[Callable], list[float]]:
    """Resolve the configured reward specs into ``(reward_funcs, reward_weights)``."""
    from ko_dialect.rewards import build_reward_fns as _build

    return _build(
        cfg.reward_specs(),
        classifier=classifier,
        cls_tokenizer=cls_tokenizer,
        max_length=cfg.max_seq_length,
    )


def train(
    cfg: GRPOConfig,
    train_dataset,
    reward_funcs: list[Callable] | Callable,
    reward_weights: list[float] | None = None,
    eval_dataset=None,
) -> None:
    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    # Load via the shared factory so GRPO gets the same quant/LoRA/vLLM options as SFT.
    # base_model_path now points at stage1's *merged 16-bit* output, so the trained SFT
    # weights are actually used (the old AutoModelForCausalLM(adapter_dir) bug is gone).
    model, tokenizer = load_backbone(cfg.to_backend_config())

    # Cast any stray bf16 tensors to fp16 (Turing has no native BF16).
    if cfg.fp16 and not cfg.bf16:
        for param in model.parameters():
            if param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float16)

    # Weights only apply to a multi-reward list; a single callable carries no weighting.
    grpo_reward_weights = reward_weights if isinstance(reward_funcs, list) else None

    grpo_args = TRLGRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_generations=cfg.num_generations,
        beta=cfg.beta,
        max_completion_length=cfg.max_new_tokens,
        max_prompt_length=cfg.max_prompt_length,
        use_vllm=cfg.use_vllm,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        seed=cfg.seed,
        report_to=cfg.report_to if cfg.report_to else "none",
        remove_unused_columns=cfg.remove_unused_columns,
        reward_weights=grpo_reward_weights,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    logger.info("GRPO model saved to %s", cfg.output_dir)
