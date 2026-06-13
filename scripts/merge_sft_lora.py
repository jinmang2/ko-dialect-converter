#!/usr/bin/env python3
"""Merge the stage-1 SFT LoRA adapter onto an fp16 base -> a standalone model.

vLLM serves a plain merged checkpoint most reliably (LoRA hot-loading works too, but
merging avoids dtype/format surprises on Turing). The adapter records an Unsloth
4-bit base, but the LoRA deltas are architecture-compatible with the plain fp16
``Qwen/Qwen2.5-0.5B-Instruct`` — which is what we want for inference on the RTX 2060.

Usage:
    python scripts/merge_sft_lora.py \
        --adapter_path outputs/sft \
        --out outputs/sft_merged
"""

from __future__ import annotations

import logging

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ko_dialect.evaluation import resolve_best_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main(
    adapter_path: str = "outputs/sft",
    out: str = "outputs/sft_merged",
    base_model: str = DEFAULT_BASE_MODEL,
) -> None:
    from peft import PeftModel

    resolved = resolve_best_checkpoint(adapter_path)
    logger.info("Merging adapter %s onto base %s", resolved, base_model)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, resolved)
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    logger.info("Merged fp16 model written to %s", out)


if __name__ == "__main__":
    fire.Fire(main)
