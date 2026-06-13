"""Pluggable model-loading backends for SFT / GRPO.

A single entry point, :func:`load_backbone`, returns ``(model, tokenizer)`` for a
:class:`BackendConfig`. Both the SFT and GRPO trainers go through here so the
quantization / precision / LoRA story is configured in exactly one place.

Defaults are **RTX 2060 (Turing, SM 7.5) safe**: ``dtype="fp16"`` and ``use_vllm=False``.
Flip flags (``bf16``, ``use_vllm``, a different ``backend``) for Ampere+/Colab.

Backend menu and where each one runs
------------------------------------
Repo-level (only transformers + peft + torchao — no Unsloth fork needed):
    ``hf``     plain full-precision transformers (+ optional LoRA) — the legacy GRPO path
    ``bnb``    bitsandbytes 4-bit NF4 QLoRA
    ``loftq``  PEFT LoftQ-initialised low-bit LoRA (better init than vanilla QLoRA)
    ``awq`` / ``gptq``  load a *pre-quantized* checkpoint, then attach a PEFT LoRA
    ``qat``    torchao quantization-aware-training + LoRA (fake-quant during training)

Unsloth turnkey entry points we simply *call* (still no internal edits):
    ``unsloth``  ``FastLanguageModel`` 4-bit/16-bit, gradient-checkpointing="unsloth",
                 optional vLLM ``fast_inference`` for fast GRPO rollouts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

VALID_BACKENDS = {"unsloth", "bnb", "hf", "loftq", "awq", "gptq", "qat"}


@dataclass
class BackendConfig:
    """Hardware/quantization-agnostic backbone-loading spec.

    Defaults reproduce the project's historical SFT behaviour (unsloth 4-bit + LoRA,
    fp16). Every richer path is opt-in via a flag.
    """

    backend: str = "unsloth"
    model_name: str = DEFAULT_MODEL
    max_seq_length: int = 256
    dtype: str = "fp16"  # fp16 | bf16 | fp32
    load_in_4bit: bool = True
    attn_implementation: str | None = None
    use_gradient_checkpointing: bool | str = "unsloth"
    use_vllm: bool = False  # unsloth fast_inference (GRPO rollout acceleration)
    gpu_memory_utilization: float = 0.5  # only used when use_vllm=True

    # LoRA / adapter
    apply_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES)
    )
    lora_bias: str = "none"
    seed: int = 42

    # PTQ backends (awq / gptq): path to an already-quantized checkpoint.
    quantized_model_path: str | None = None
    # QAT backend (torchao): fake-quant scheme, e.g. "int4-weight-only".
    qat_scheme: str = "int4-weight-only"

    def __post_init__(self) -> None:
        if self.backend not in VALID_BACKENDS:
            raise ValueError(
                f"Unknown backend {self.backend!r}. Choose one of {sorted(VALID_BACKENDS)}."
            )


def resolve_dtype(cfg: BackendConfig):
    """Map ``cfg.dtype`` to a torch dtype, guarding bf16 on pre-Ampere GPUs.

    RTX 2060 is Turing (SM 7.5) with no native BF16; requesting bf16 there silently
    degrades, so we downgrade to fp16 with a loud warning instead.
    """
    import torch

    if cfg.dtype == "bf16":
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                logger.warning(
                    "bf16 requested on SM %d.x (<8.0, e.g. Turing/RTX20xx) which lacks "
                    "native BF16 — forcing fp16. Use dtype=fp16 on this GPU.",
                    major,
                )
                return torch.float16
        return torch.bfloat16
    if cfg.dtype == "fp32":
        return torch.float32
    return torch.float16


def load_backbone(cfg: BackendConfig):
    """Dispatch to the configured backend; returns ``(model, tokenizer)``."""
    loader = {
        "unsloth": _load_unsloth,
        "bnb": _load_bnb,
        "hf": _load_hf,
        "loftq": _load_loftq,
        "awq": _load_ptq,
        "gptq": _load_ptq,
        "qat": _load_qat,
    }[cfg.backend]
    logger.info(
        "Loading backbone via backend=%s (dtype=%s, 4bit=%s, lora=%s)",
        cfg.backend,
        cfg.dtype,
        cfg.load_in_4bit,
        cfg.apply_lora,
    )
    return loader(cfg)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_unsloth(cfg: BackendConfig):
    from unsloth import FastLanguageModel

    kwargs = dict(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=resolve_dtype(cfg),
        load_in_4bit=cfg.load_in_4bit,
    )
    if cfg.use_vllm:
        # Unsloth's native vLLM fast inference (≈20x rollout throughput). GRPO-only.
        kwargs.update(fast_inference=True, gpu_memory_utilization=cfg.gpu_memory_utilization)
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)

    if cfg.apply_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias=cfg.lora_bias,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
            random_state=cfg.seed,
        )
    logger.info("Loaded via unsloth FastLanguageModel (vllm=%s)", cfg.use_vllm)
    return model, _ensure_pad_token(tokenizer)


def _load_bnb(cfg: BackendConfig):
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = resolve_dtype(cfg)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = _ensure_pad_token(AutoTokenizer.from_pretrained(cfg.model_name))
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_cfg,
        torch_dtype=compute_dtype,
        device_map="auto",
        attn_implementation=cfg.attn_implementation,
    )
    if cfg.apply_lora:
        model.enable_input_require_grads()  # gradient flow through frozen 4-bit backbone
        model = get_peft_model(model, _lora_config(cfg))
    logger.info("Loaded via bitsandbytes QLoRA")
    return model, tokenizer


def _load_hf(cfg: BackendConfig):
    """Plain full-precision transformers (the legacy GRPO path), optional LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = _ensure_pad_token(AutoTokenizer.from_pretrained(cfg.model_name))
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=resolve_dtype(cfg),
        device_map="auto",
        attn_implementation=cfg.attn_implementation,
    )
    if cfg.apply_lora:
        from peft import get_peft_model

        model.enable_input_require_grads()
        model = get_peft_model(model, _lora_config(cfg))
    logger.info("Loaded via plain transformers (hf)")
    return model, tokenizer


def _load_loftq(cfg: BackendConfig):
    """LoftQ-initialised LoRA — repo-level, PEFT-native, better low-bit init than QLoRA.

    Ref: PEFT docs `developer_guides/quantization` (LoftQConfig).
    """
    from peft import LoftQConfig, LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = _ensure_pad_token(AutoTokenizer.from_pretrained(cfg.model_name))
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=resolve_dtype(cfg))
    loftq_config = LoftQConfig(loftq_bits=4)
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias=cfg.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights="loftq",
        loftq_config=loftq_config,
    )
    model = get_peft_model(model, lora_cfg)
    logger.info("Loaded via LoftQ-initialised LoRA")
    return model, tokenizer


def _load_ptq(cfg: BackendConfig):
    """PTQ (AWQ/GPTQ) base + PEFT LoRA. Repo-level.

    Pre-quantize the base once with ``scripts/ptq_quantize.py`` (AutoAWQ / AutoGPTQ),
    point ``quantized_model_path`` at it, then train a LoRA on top. AWQ generally gives
    lower perplexity than GPTQ for adapter training.
    """
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = cfg.quantized_model_path
    if not path:
        raise ValueError(
            f"backend={cfg.backend!r} needs `quantized_model_path` pointing at a "
            "pre-quantized checkpoint. Build one with scripts/ptq_quantize.py."
        )
    tokenizer = _ensure_pad_token(AutoTokenizer.from_pretrained(path))
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=resolve_dtype(cfg), device_map="auto"
    )
    if cfg.apply_lora:
        model.enable_input_require_grads()
        model = get_peft_model(model, _lora_config(cfg))
    logger.info("Loaded pre-quantized %s model + LoRA from %s", cfg.backend, path)
    return model, tokenizer


def _load_qat(cfg: BackendConfig):
    """QAT + LoRA via torchao fake-quantization.

    Why this backend exists: the export target is GGUF Q4_K_M. QAT fake-quantizes
    weights during training so the eventual 4-bit model degrades far less than a
    naive PTQ. torchao's QAT composes with LoRA at the *repo level* (no Unsloth fork);
    Unsloth also ships a turnkey QAT path if you prefer its kernels.

    Implementation is intentionally a thin, explicit scaffold so the technique can be
    filled in without touching the rest of the pipeline.
    """
    try:
        from torchao.quantization import qat  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "backend='qat' needs torchao (`pip install torchao`). Apply a QAT quantizer "
            "(e.g. Int4WeightOnlyQATQuantizer) to the base model, then attach LoRA. "
            "See torchao docs: workflows/qat."
        ) from exc
    raise NotImplementedError(
        "QAT backend scaffold: insert torchao QAT quantizer + LoRA here. The seam "
        "(BackendConfig.qat_scheme, save_merged_16bit) is ready; choose torchao-native "
        "or Unsloth's QAT entry point per hardware."
    )


def _lora_config(cfg: BackendConfig):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias=cfg.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )


# --------------------------------------------------------------------------- #
# Saving — make SFT output directly loadable by GRPO
# --------------------------------------------------------------------------- #
def save_merged_16bit(model, tokenizer, output_dir: str) -> None:
    """Save full 16-bit weights (LoRA merged into the base) into ``output_dir``.

    This is what fixes the GRPO-loads-an-adapter-dir bug: after SFT we write a real,
    standalone model so ``AutoModelForCausalLM.from_pretrained(output_dir)`` (and the
    factory's ``hf``/``unsloth`` backends) load the *trained* weights, not the raw base.
    """
    # Unsloth models expose a merged-save helper that handles 4-bit bases.
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
        logger.info("Saved unsloth merged_16bit model to %s", output_dir)
        return

    # PEFT path: merge adapter into the (de-quantized) base and save.
    merged = model
    if hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved merged 16-bit model to %s", output_dir)
