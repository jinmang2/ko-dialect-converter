"""Canonical eval-time generation harness.

Every eval/serve script (``evaluate.py``, ``eval_leaderboard.py``, ``eval_prosody_ab.py``,
``compare_sft_grpo.py``, ``eval_grpo_checkpoints.py``) used to carry its own byte-identical
copy of "load a model or adapter, then greedy-decode prompts with left padding". That
duplication is the easiest place for the decoder-only left-padding rule to silently rot in
one copy. This module is the single source of truth; the scripts import from here.

Following the ``serving.py`` convention, the pure / filesystem logic (``resolve_eval_model``)
is unit-tested on CPU; the GPU path (``load_generation_model``) stays thin.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .checkpoints import resolve_best_checkpoint

logger = logging.getLogger(__name__)

# fp16 base for RTX 2060 (Turing, no BF16). Adapters record an Unsloth 4-bit base, but the
# LoRA weights are architecture-compatible with the plain fp16 checkpoint we eval against.
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class _Generator(Protocol):
    def generate(self, **kwargs: Any) -> Any: ...


def resolve_eval_model(model_path: str, base_model: str | None = None) -> tuple[str, bool]:
    """Resolve an eval target to ``(path, is_adapter)``.

    A Trainer output dir holding only ``checkpoint-*`` subdirs is resolved to its latest
    checkpoint. ``is_adapter`` is True when the resolved dir carries an
    ``adapter_config.json`` (a LoRA adapter to merge onto ``base_model``) rather than a
    full model. Pure filesystem logic — unit-tested without a GPU.
    """
    resolved = resolve_best_checkpoint(model_path)
    is_adapter = (Path(resolved) / "adapter_config.json").exists()
    return resolved, is_adapter


def load_generation_model(
    model_path: str,
    base_model: str | None = None,
    *,
    dtype: Any = None,
):
    """Load a full model, or a LoRA adapter merged onto its fp16 base, plus tokenizer.

    Imports torch/transformers lazily so importing this module (and the eval scripts) stays
    cheap on a CPU-only CI box.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.float16

    resolved, is_adapter = resolve_eval_model(model_path, base_model)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    if is_adapter:
        from peft import PeftModel

        base = base_model or DEFAULT_BASE_MODEL
        logger.info("Detected LoRA adapter at %s; base=%s", resolved, base)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, device_map="auto")
        model = PeftModel.from_pretrained(model, resolved).merge_and_unload()
    else:
        logger.info("Loading full model from %s", resolved)
        model = AutoModelForCausalLM.from_pretrained(resolved, torch_dtype=dtype, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_batched(
    model: _Generator,
    tokenizer: Any,
    prompts: list[str],
    *,
    device: Any = None,
    batch_size: int = 16,
    max_new_tokens: int = 64,
    max_length: int | None = None,
    post: Callable[[str], str] | None = None,
) -> list[str]:
    """Greedy-decode ``prompts`` in batches with **left padding**.

    Decoder-only models require left padding for correct batched generation: right padding
    would push pad tokens between the prompt and the first generated token, corrupting the
    output. With left padding every row shares the same input length, so a single slice
    recovers the continuation for the whole batch.

    ``post`` is an optional per-output transform applied after decoding (e.g. stripping
    prosody markers before scoring). ``device`` defaults to the model's device.
    """
    import torch

    if device is None:
        device = next(model.parameters()).device  # type: ignore[attr-defined]

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                chunk = prompts[start : start + batch_size]
                enc = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                gen_ids = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                new_ids = gen_ids[:, enc["input_ids"].shape[1] :]
                decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
                for text in decoded:
                    text = text.strip()
                    outputs.append(post(text) if post else text)
                logger.info("Generated %d / %d", len(outputs), len(prompts))
    finally:
        tokenizer.padding_side = prev_side
    return outputs
