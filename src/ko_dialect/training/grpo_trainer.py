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
    # -1 = run full epochs; set a positive value to cap optimizer steps (smoke tests,
    # quick reward-hacking probes). Mirrors HF TrainingArguments.max_steps semantics.
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-6
    num_generations: int = 4
    beta: float = 0.04  # KL-to-reference weight; raise to anchor harder to the SFT policy
    # DAPO knobs (trl 1.5.1 GRPOTrainer already defaults loss_type=dapo, token-level IS).
    # Clip-Higher: set epsilon_high > epsilon to let the policy raise good-token prob more
    # (prevents entropy collapse). mask_truncated: drop completions cut off at the token
    # cap so length-truncation noise doesn't pollute the gradient.
    epsilon: float = 0.2
    epsilon_high: float | None = None
    mask_truncated_completions: bool = False
    max_new_tokens: int = 128
    # max_prompt_length: int = 256  # TypeError: GRPOConfig.__init__() got an unexpected keyword argument 'max_prompt_length'
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 20
    save_steps: int = 200
    seed: int = 42
    report_to: list[str] = field(default_factory=list)
    # How multiple reward functions become the scalar advantage:
    #   "weighted"  — Arm 0 control. TRL's default sum-then-normalize (weighted sum of
    #                 rewards, then one group-normalization). Uses GRPOTrainer.
    #   "mo_grpo"   — Arm 1. MO-GRPO (arXiv:2509.22047): per-objective z-normalize within
    #                 the prompt group, THEN sum. Uses MOGRPOTrainer. Removes high-variance-
    #                 axis domination (the style-axis reward-hacking we saw).
    #   "hm"        — fallback. Collapse the axes into ONE joint weighted-harmonic-mean
    #                 reward (floor aggregation: a dead axis sinks the whole score) and
    #                 pass it to a plain GRPOTrainer as a single reward_func.
    aggregation: str = "weighted"
    # Reward spec list: [{name: style, weight: 1.0}, {name: content, weight: 0.5}, ...].
    rewards: list[dict[str, Any]] | None = None
    use_edit_reward: bool = False  # back-compat shim: appends {edit, 0.3} if rewards unset
    # Rescale every reward to a common [0, 1] basis before weighting, so weights (not
    # each reward's native scale, e.g. r_style ∈ [-1, 1]) govern the mixture.
    normalize_rewards: bool = False
    # `length` reward (DAPO-style soft overlong penalty) tuning, relative to gold length.
    length_max_ratio: float = 1.5
    length_tolerance: float = 0.2
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
    """Resolve the configured reward specs into ``(reward_funcs, reward_weights)``.

    Reward scaling & weighting: r_style is natively [-1, 1] while r_content / r_edit /
    r_length are [0, 1]. Since GRPO sums the *weighted* rewards before group-normalizing
    the advantage, a wider-range reward silently dominates the mix. Set
    ``cfg.normalize_rewards=True`` to rescale every reward onto a common [0, 1] basis
    (via ``REWARD_OUTPUT_RANGES``) so the configured weights become the only mixing knob.
    """
    from ko_dialect.rewards import build_reward_fns as _build

    return _build(
        cfg.reward_specs(),
        classifier=classifier,
        cls_tokenizer=cls_tokenizer,
        max_length=cfg.max_seq_length,
        normalize=cfg.normalize_rewards,
        length_max_ratio=cfg.length_max_ratio,
        length_tolerance=cfg.length_tolerance,
    )


def build_joint_hm_reward(
    reward_funcs: list[Callable],
    reward_weights: list[float],
    *,
    eps: float = 0.1,
) -> Callable:
    """Collapse per-axis rewards into ONE weighted-harmonic-mean ("floor") reward.

    Fallback for when MO-GRPO is unavailable. Per plan §11-M1, the joint reward is

        r = (Σ wᵢ) / Σ( wᵢ / (xᵢ + ε) ),   ε = 0.1,

    over axes xᵢ that must all live on a (0, 1] basis (the caller is responsible for
    rescaling, e.g. ``normalize_rewards=True`` maps style's [-1, 1] to [0, 1]). Unlike a
    weighted *sum*, the harmonic mean is dominated by the smallest axis: if any single
    objective collapses toward 0 the whole reward collapses, so the policy cannot trade
    one axis off against another (the over-optimization failure mode). ε sets how hard
    the floor bites (ε→0 ⇒ pure HM/min, ε large ⇒ ~weighted mean); it is the first lever
    for AC6 group-gradient health (plan §11-M3).

    The per-axis funcs share the TRL reward signature, so this returns a single
    TRL-compatible ``fn(prompts, completions, **columns) -> list[float]`` that evaluates
    each axis and combines row-wise. ``reward_weights`` align with ``reward_funcs``.
    """
    w_sum = float(sum(reward_weights))

    def joint_hm(prompts, completions, **columns) -> list[float]:
        per_axis = [fn(prompts=prompts, completions=completions, **columns) for fn in reward_funcs]
        out: list[float] = []
        for row in zip(*per_axis):
            denom = sum(w / (max(0.0, min(1.0, x)) + eps) for w, x in zip(reward_weights, row))
            out.append(w_sum / denom if denom > 0 else 0.0)
        return out

    joint_hm.__name__ = "joint_hm"
    return joint_hm


def train(
    cfg: GRPOConfig,
    train_dataset,
    reward_funcs: list[Callable] | Callable,
    reward_weights: list[float] | None = None,
    eval_dataset=None,
) -> None:
    """Run GRPO training with the resolved reward functions.

    Verbosity-bias guard: GRPO/PPO maximise reward and tend to pad generations,
    which is exploitable here (a longer output incidentally hits more gold dialect
    words in ``r_edit``). Mitigate by adding the DAPO-style ``length`` reward to the
    config (``- {name: length, weight: 0.2}``); it applies a soft overlong penalty
    anchored to the gold target length. Tune via ``cfg.length_max_ratio`` /
    ``cfg.length_tolerance``.
    """

    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    from ko_dialect.training.mo_grpo_trainer import MOGRPOTrainer

    # "hm" fallback: collapse the per-axis rewards into a single weighted-harmonic-mean
    # reward BEFORE constructing the trainer. TRL sums multiple reward_funcs, which is the
    # opposite of a floor aggregation, so the HM must be a single registered reward_func
    # (plan §11-M1). Requires the per-axis rewards on a (0, 1] basis (normalize_rewards).
    if cfg.aggregation == "hm":
        if not isinstance(reward_funcs, list):
            raise ValueError(
                "aggregation='hm' needs the list of per-axis reward_funcs to combine; "
                "got a single callable."
            )
        # Harmonic mean is only meaningful on (0, 1] inputs; enforce normalization.
        if not cfg.normalize_rewards:
            raise ValueError(
                "aggregation='hm' requires normalize_rewards=True so each reward axis is "
                "on a (0, 1] basis before the harmonic mean is computed. Without it, "
                "axes with different native ranges (e.g. r_style ∈ [-1,1]) produce a "
                "meaningless result."
            )
        weights = reward_weights or [1.0] * len(reward_funcs)
        reward_funcs = build_joint_hm_reward(reward_funcs, weights)
        reward_weights = None  # single joint reward carries no per-axis weighting

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

    # DAPO Clip-Higher invariant: epsilon_high must be >= epsilon.
    if cfg.epsilon_high is not None and cfg.epsilon_high < cfg.epsilon:
        raise ValueError(
            f"DAPO Clip-Higher requires epsilon_high >= epsilon, but got "
            f"epsilon_high={cfg.epsilon_high} < epsilon={cfg.epsilon}."
        )

    grpo_args = TRLGRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_generations=cfg.num_generations,
        beta=cfg.beta,
        epsilon=cfg.epsilon,
        epsilon_high=cfg.epsilon_high,
        mask_truncated_completions=cfg.mask_truncated_completions,
        max_completion_length=cfg.max_new_tokens,
        # max_prompt_length=cfg.max_prompt_length,
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

    # MO-GRPO (Arm 1): per-objective z-normalize within the group, then sum. TRL realizes
    # this with multi_objective_aggregation="normalize_then_sum" + scale_rewards="none";
    # MOGRPOTrainer enforces both at construction. "weighted"/"hm" use the default
    # sum-then-normalize path on a plain GRPOTrainer.
    if cfg.aggregation == "mo_grpo":
        grpo_args.multi_objective_aggregation = "normalize_then_sum"
        grpo_args.scale_rewards = "none"
        trainer_cls = MOGRPOTrainer
    else:
        trainer_cls = GRPOTrainer

    trainer = trainer_cls(
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
