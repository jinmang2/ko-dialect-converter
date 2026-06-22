from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def merge_lora_and_save(
    base_model_path: str,
    lora_adapter_path: str,
    merged_output_path: str,
) -> None:
    """Merge LoRA adapter into the base model and save full fp16 weights."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model from %s", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model = model.merge_and_unload()

    out = Path(merged_output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("Merged model saved to %s", out)


def convert_to_gguf(
    model_path: str,
    output_path: str,
    llama_cpp_dir: str,
    outtype: str = "f16",
) -> Path:
    """Convert a HuggingFace model directory to GGUF using llama.cpp's convert script."""
    convert_script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"llama.cpp convert script not found: {convert_script}\n"
            "Clone and build llama.cpp first: https://github.com/ggerganov/llama.cpp"
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(convert_script),
        model_path,
        "--outfile",
        str(out),
        "--outtype",
        outtype,
    ]
    logger.info("Converting to GGUF: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("GGUF saved to %s", out)
    return out


def quantize_gguf(
    input_gguf: str,
    output_gguf: str,
    llama_cpp_dir: str,
    quant_type: str = "Q4_K_M",
) -> Path:
    """Quantize a GGUF file with llama-quantize."""
    llama_cpp = Path(llama_cpp_dir)
    quantize_bin = llama_cpp / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        raise FileNotFoundError(
            f"llama-quantize not found under {llama_cpp_dir}. "
            "Build llama.cpp with: cmake -B build && cmake --build build -j"
        )

    out = Path(output_gguf)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(quantize_bin), input_gguf, str(out), quant_type]
    logger.info("Quantizing %s → %s (%s)", input_gguf, out, quant_type)
    subprocess.run(cmd, check=True)
    logger.info("Quantized GGUF saved to %s", out)
    return out


def start_llama_server(
    gguf_path: str,
    llama_cpp_dir: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,  # -1 = offload all layers; Qwen2.5-0.5B has 24 layers (not 35)
    chat_template: str | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Start llama-server in a subprocess and return its Popen handle.

    Stop the server with: proc.terminate()
    """
    llama_cpp = Path(llama_cpp_dir)
    server_bin = llama_cpp / "llama-server"
    if not server_bin.exists():
        server_bin = llama_cpp / "build" / "bin" / "llama-server"
    if not server_bin.exists():
        raise FileNotFoundError(
            f"llama-server not found under {llama_cpp_dir}. Build llama.cpp first."
        )

    cmd = [
        str(server_bin),
        "--model",
        gguf_path,
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(n_ctx),
        "--n-gpu-layers",
        str(n_gpu_layers),
    ]
    if chat_template:
        cmd += ["--chat-template", chat_template]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Starting llama-server (PID will log below): %s", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    logger.info("llama-server PID=%d  →  http://%s:%d", proc.pid, host, port)
    return proc
