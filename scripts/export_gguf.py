#!/usr/bin/env python3
"""Export: merge LoRA → convert to GGUF → quantize → serve with llama-server.

Usage examples:
    python scripts/export_gguf.py merge  --base outputs/sft --lora outputs/sft --out outputs/merged
    python scripts/export_gguf.py convert --model outputs/merged --out outputs/gguf/model_f16.gguf
    python scripts/export_gguf.py quantize --input outputs/gguf/model_f16.gguf
    python scripts/export_gguf.py serve   --gguf outputs/gguf/model_Q4_K_M.gguf
"""
from __future__ import annotations

import logging

import fire

from ko_dialect.export import (
    convert_to_gguf,
    merge_lora_and_save,
    quantize_gguf,
    start_llama_server,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def merge(
    base: str,
    lora: str,
    out: str = "outputs/merged",
) -> None:
    """Merge LoRA adapter into the base model (fp16 safetensors)."""
    merge_lora_and_save(base, lora, out)


def convert(
    model: str,
    out: str = "outputs/gguf/model_f16.gguf",
    llama_cpp_dir: str = "llama.cpp",
    outtype: str = "f16",
) -> None:
    """Convert merged HF model to GGUF f16."""
    convert_to_gguf(model, out, llama_cpp_dir, outtype)


def quantize(
    input: str,
    output: str = "outputs/gguf/model_Q4_K_M.gguf",
    llama_cpp_dir: str = "llama.cpp",
    quant_type: str = "Q4_K_M",
) -> None:
    """Quantize an f16 GGUF to Q4_K_M (or other quant type)."""
    quantize_gguf(input, output, llama_cpp_dir, quant_type)


def serve(
    gguf: str,
    llama_cpp_dir: str = "llama.cpp",
    host: str = "127.0.0.1",
    port: int = 8080,
    n_ctx: int = 2048,
    n_gpu_layers: int = 35,
) -> None:
    """Start llama-server. Ctrl-C to stop."""
    proc = start_llama_server(gguf, llama_cpp_dir, host, port, n_ctx, n_gpu_layers)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("Server stopped.")


if __name__ == "__main__":
    fire.Fire({"merge": merge, "convert": convert, "quantize": quantize, "serve": serve})
