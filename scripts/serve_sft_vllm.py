#!/usr/bin/env python3
"""Qualitatively check the SFT model with vLLM (offline, batched) on the RTX 2060.

Turing (SM 7.5) + 6 GB VRAM safety (see "Will 0.5B OOM?" below):
  * ``dtype="float16"``       — Turing has NO bf16; Qwen2.5's config dtype is bf16,
                                so we MUST override or vLLM errors/degrades.
  * ``enforce_eager=True``    — skip CUDA-graph capture (saves ~0.5-1 GB, avoids the
                                profiling spike that OOMs a 6 GB card).
  * ``gpu_memory_utilization``— fraction of *total* VRAM vLLM may claim. 0.55 leaves
                                headroom on a card already holding ~1 GB of desktop.
  * ``max_model_len`` / ``max_num_seqs`` kept small — KV cache scales with both.

Memory math (0.5B, fp16): weights ≈ 1.0 GB (measured: vLLM loads it in 0.93 GiB).
KV cache per token ≈ 24 layers × 2 × 2 KV-heads × 64 dim × 2 B ≈ 12 KB; 1024 tokens ×
16 seqs ≈ 0.2 GB. The model fits comfortably in <2.5 GB — **OOM is not the failure
mode on this card.** A bare ``LLM("...0.5B")`` with defaults (gpu_mem_util=0.9, cuda
graphs, max_len 32k) could spike during profiling, which the flags above prevent.

VLLM ON THIS BOX (Turing + pip-CUDA): vLLM auto-selects the FlashInfer backend (FA2
needs SM>=8.0). FlashInfer JIT-compiles kernels with ``nvcc`` at startup, which needs:
  (a) ``nvcc`` on PATH — handled by ``_ensure_nvcc`` (points CUDA_HOME at the bundled
      ``nvidia/cu*`` wheel toolkit when ``/usr/local/cuda`` lacks nvcc), AND
  (b) a FlashInfer build whose bundled cccl headers match the toolkit CUDA version.
On this env (FlashInfer 0.6.11 + CUDA 13.3) (b) fails with a header-incompatibility
error. Fix by pinning a CUDA-13-matched FlashInfer, or just use ``--backend hf`` (the
default here) — same greedy outputs via transformers, no JIT toolchain needed.

Usage:
    # 1) merge first (vLLM serves a standalone model):
    python scripts/merge_sft_lora.py --out outputs/sft_merged
    # 2) translate eval samples, std -> dialect, side by side with the reference:
    python scripts/serve_sft_vllm.py \
        --model_path outputs/sft_merged \
        --raw_dataset_path outputs/dialect_raw_new \
        --target_do gangwondo --direction std2dia --n 20         # backend=hf (default)
    python scripts/serve_sft_vllm.py --backend vllm ...          # once FlashInfer is fixed
"""

from __future__ import annotations

import glob
import logging
import os
import shutil

import fire
from datasets import load_from_disk
from transformers import AutoTokenizer

from ko_dialect.data import ChatTemplate
from ko_dialect.evaluation import resolve_best_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ensure_nvcc() -> None:
    """Point CUDA_HOME at a toolkit that actually has ``nvcc``.

    vLLM on Turing falls back to FlashInfer, which JIT-compiles its sampling/prefill
    kernels with ``nvcc``. If ``CUDA_HOME`` (default ``/usr/local/cuda``) has no nvcc,
    that compile fails with ``nvcc: not found`` — which looks like a vLLM/OOM error but
    is just a missing compiler. The ``nvidia-cuda-*`` pip wheels ship a usable nvcc, so
    prefer that when the system toolkit is absent. Must run before importing vllm.
    """
    if shutil.which("nvcc"):
        return
    home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if os.path.exists(os.path.join(home, "bin", "nvcc")):
        os.environ.setdefault("PATH", "")
        os.environ["PATH"] = f"{home}/bin:{os.environ['PATH']}"
        return
    matches = glob.glob(
        os.path.join(
            os.path.dirname(os.__file__),
            "site-packages",
            "nvidia",
            "cu*",
            "bin",
            "nvcc",
        )
    )
    if matches:
        toolkit = os.path.dirname(os.path.dirname(matches[0]))
        os.environ["CUDA_HOME"] = toolkit
        os.environ["PATH"] = f"{toolkit}/bin:{os.environ.get('PATH', '')}"


def _load_eval_samples(raw_dataset_path, split, target_do, direction, n):
    ds = load_from_disk(raw_dataset_path)[split]
    ds = ds.filter(lambda x: x["do"] == target_do and not x["is_identical"])
    if n and n < len(ds):
        ds = ds.select(range(n))
    template = ChatTemplate()
    rows = []
    for s in ds:
        source = s["standard"] if direction == "std2dia" else s["dialect"]
        reference = s["dialect"] if direction == "std2dia" else s["standard"]
        rows.append({"source": source, "reference": reference})
    return rows, template


def _gen_vllm(model_path, prompts, max_new_tokens, gpu_mem, max_model_len, max_num_seqs):
    _ensure_nvcc()  # FlashInfer JIT needs nvcc; fix CUDA_HOME before importing vllm
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="float16",  # Turing: no bf16
        enforce_eager=True,  # no CUDA graphs -> lower peak VRAM
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outs = llm.generate(prompts, sp)
    # preserve input order
    return [o.outputs[0].text.strip() for o in outs]


def _gen_hf(model_path, prompts, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), 16):
            chunk = prompts[i : i + 16]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True).to(dev)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1] :]
            out.extend(t.strip() for t in tok.batch_decode(new, skip_special_tokens=True))
    return out


def main(
    model_path: str = "outputs/sft_merged",
    raw_dataset_path: str = "outputs/dialect_raw_new",
    split: str = "valid",
    target_do: str = "gangwondo",
    direction: str = "std2dia",
    n: int = 20,
    max_new_tokens: int = 128,
    backend: str = "hf",
    gpu_memory_utilization: float = 0.55,
    max_model_len: int = 1024,
    max_num_seqs: int = 16,
) -> None:
    resolved = resolve_best_checkpoint(model_path)
    rows, template = _load_eval_samples(raw_dataset_path, split, target_do, direction, n)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    prompts = [template.build_prompt(tokenizer, r["source"], target_do, direction) for r in rows]

    logger.info(
        "Generating %d samples via backend=%s (%s, %s)", len(rows), backend, target_do, direction
    )
    if backend == "vllm":
        gens = _gen_vllm(
            resolved, prompts, max_new_tokens, gpu_memory_utilization, max_model_len, max_num_seqs
        )
    else:
        gens = _gen_hf(resolved, prompts, max_new_tokens)

    exact = 0
    print("\n" + "=" * 78)
    print(f"SFT QUALITATIVE CHECK  {target_do} / {direction}  (backend={backend})")
    print("=" * 78)
    for r, g in zip(rows, gens):
        exact += int(g.strip() == r["reference"].strip())
        print(f"  SRC : {r['source']}")
        print(f"  GEN : {g}")
        print(f"  REF : {r['reference']}")
        print("  " + "-" * 74)
    print(
        f"\nexact match vs reference: {exact}/{len(rows)} "
        f"({100 * exact / max(len(rows), 1):.1f}%)  "
        f"[low is normal — dialect conversion is many-valid-outputs]"
    )


if __name__ == "__main__":
    fire.Fire(main)
