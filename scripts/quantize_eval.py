#!/usr/bin/env python3
"""Quantization trade-off bench: quality × size × latency for PTQ variants.

EXPERIMENTS §3: the quantization backends are scaffolded but unmeasured. This quantifies
the trade-off for a *finished merged model* — how much chrF / copy_margin you give up, and
how much size / latency you save, by serving it 4-bit (bitsandbytes NF4) instead of fp16.

    python scripts/quantize_eval.py --model_path outputs/grpo_500_merged --target_do gangwondo

Reuses the shared generation harness (ko_dialect.evaluation.generate_batched) and the
CPU-tested trade-off summary (ko_dialect.evaluation.serving.quantization_tradeoff). GGUF
(Q4_K_M) is measured separately via scripts/export_gguf.py + bench_serving.py. PTQ first;
escalate to QAT (training.backend=qat) only if 4-bit PTQ degrades quality too much.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import fire

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_fp16(model_path: str):
    from ko_dialect.evaluation import load_generation_model

    return load_generation_model(model_path)


def _load_bnb4(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # RTX 2060 (Turing): fp16, never bf16
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    return model, tok


def _measure(model, tok, prompts, refs, srcs, *, max_new_tokens, batch_size) -> dict:
    from sacrebleu.metrics import CHRF

    from ko_dialect.evaluation import generate_batched

    device = next(model.parameters()).device
    t0 = time.perf_counter()
    gens = generate_batched(
        model, tok, prompts, device=device, batch_size=batch_size, max_new_tokens=max_new_tokens
    )
    elapsed = time.perf_counter() - t0
    chrf = CHRF()
    chrf_ref = chrf.corpus_score(gens, [refs]).score
    chrf_src = chrf.corpus_score(gens, [srcs]).score
    return {
        "chrf": round(chrf_ref, 3),
        "copy_margin": round(chrf_ref - chrf_src, 3),
        "latency_ms_p50": round(1000 * elapsed / max(len(prompts), 1), 2),
    }


def main(
    model_path: str,
    target_do: str = "gangwondo",
    n_samples: int = 100,
    max_new_tokens: int = 64,
    batch_size: int = 16,
    config: str | None = None,
    output_file: str | None = None,
    include_bnb4: bool = True,
) -> None:
    """Bench fp16 vs 4-bit for ``model_path`` and print the trade-off table."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "quantize_eval needs a CUDA GPU: it reports MEASURED peak VRAM + 4-bit latency, "
            "which are meaningless on CPU (bnb 4-bit won't load; size_mb would fall back to "
            "disk size, mixing units into a misleading table). Run on the GPU box."
        )
    from datasets import load_from_disk

    from ko_dialect.data import ChatTemplate
    from ko_dialect.evaluation import EvalConfig
    from ko_dialect.evaluation.serving import (
        directory_size_mb,
        format_tradeoff_table,
        quantization_tradeoff,
    )

    cfg = EvalConfig.load(
        config, n_samples=n_samples, max_new_tokens=max_new_tokens, batch_size=batch_size
    )
    ds = load_from_disk(cfg.raw_dataset_path)[cfg.split]
    ds = ds.filter(lambda x: x["do"] == target_do and not x["is_identical"])
    if cfg.n_samples:
        ds = ds.select(range(min(cfg.n_samples, len(ds))))

    template = ChatTemplate()
    disk_mb = directory_size_mb(model_path)  # fp16 on-disk reference (logged, not the footprint)
    variants = []
    plan = [("fp16", _load_fp16)] + ([("bnb4", _load_bnb4)] if include_bnb4 else [])
    for name, loader in plan:
        logger.info("Benching variant: %s", name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model, tok = loader(model_path)
        prompts = [template.build_prompt(tok, s["standard"], target_do, "std2dia") for s in ds]
        refs = [s["dialect"] for s in ds]
        srcs = [s["standard"] for s in ds]
        metrics = _measure(
            model,
            tok,
            prompts,
            refs,
            srcs,
            max_new_tokens=cfg.max_new_tokens,
            batch_size=cfg.batch_size,
        )
        # size_mb = MEASURED peak VRAM (the real footprint that decides what fits a 6GB GPU),
        # not a disk-size guess. disk_mb (fp16 on-disk) is logged for reference only.
        peak_mb = (
            round(torch.cuda.max_memory_allocated() / (1024**2), 1)
            if torch.cuda.is_available()
            else disk_mb
        )
        metrics.update(name=name, size_mb=peak_mb, disk_mb=disk_mb if name == "fp16" else None)
        logger.info("[%s] peak VRAM=%.1f MB", name, peak_mb)
        variants.append(metrics)
        del model
        torch.cuda.empty_cache()

    summary = quantization_tradeoff(variants, baseline="fp16")
    # Identify the run so report.py can distinguish (and label) quantization benches on
    # different models/regions instead of collapsing them by their shared variant names.
    summary["model"] = model_path
    summary["target_do"] = target_do
    summary["n_samples"] = cfg.n_samples
    print(format_tradeoff_table(summary))
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        logger.info("Wrote %s", output_file)


if __name__ == "__main__":
    fire.Fire(main)
