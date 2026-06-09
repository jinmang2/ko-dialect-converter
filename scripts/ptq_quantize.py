#!/usr/bin/env python3
"""Post-training-quantize the base model to AWQ or GPTQ, for the awq/gptq LoRA backends.

This is a *repo-level* step — no Unsloth fork. Run it once to produce a quantized
checkpoint, then point ``training.backend=awq`` (or ``gptq``) and
``training.quantized_model_path=<out>`` at it to train a LoRA on the quantized base.

    python scripts/ptq_quantize.py --method awq \
        --model_name Qwen/Qwen2.5-0.5B-Instruct --output_dir outputs/qwen-awq

AWQ generally yields lower perplexity than GPTQ for subsequent adapter training, but
needs a small calibration set. Both require their own optional dependency
(``autoawq`` / ``auto-gptq`` or ``gptqmodel``), kept out of the core install.
"""

from __future__ import annotations

import logging

import fire

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CALIB = [
    "다음 문장을 강원도 사투리로 바꿔줘.",
    "오늘 날씨가 참 좋네요.",
    "경상도 사투리를 표준어로 바꿔주세요.",
    "밥은 먹고 다니나?",
]


def _quantize_awq(model_name: str, output_dir: str, bits: int) -> None:
    try:
        from awq import AutoAWQForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("AWQ quantization needs `pip install autoawq`.") from exc
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAWQForCausalLM.from_pretrained(model_name)
    quant_config = {"w_bit": bits, "q_group_size": 128, "zero_point": True, "version": "GEMM"}
    model.quantize(tokenizer, quant_config=quant_config, calib_data=DEFAULT_CALIB)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("AWQ %d-bit checkpoint written to %s", bits, output_dir)


def _quantize_gptq(model_name: str, output_dir: str, bits: int) -> None:
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "GPTQ quantization needs `pip install gptqmodel` (or auto-gptq)."
        ) from exc

    quant_config = QuantizeConfig(bits=bits, group_size=128)
    model = GPTQModel.load(model_name, quant_config)
    model.quantize(DEFAULT_CALIB)
    model.save(output_dir)
    logger.info("GPTQ %d-bit checkpoint written to %s", bits, output_dir)


def main(
    method: str = "awq",
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "outputs/quantized",
    bits: int = 4,
) -> None:
    """Quantize ``model_name`` with ``method`` ("awq" | "gptq") into ``output_dir``."""
    if method == "awq":
        _quantize_awq(model_name, output_dir, bits)
    elif method == "gptq":
        _quantize_gptq(model_name, output_dir, bits)
    else:
        raise ValueError(f"method must be 'awq' or 'gptq', got {method!r}.")


if __name__ == "__main__":
    fire.Fire(main)
