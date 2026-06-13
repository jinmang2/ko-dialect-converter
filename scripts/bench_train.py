#!/usr/bin/env python3
"""Honest training-throughput bench: unsloth vs DeepSpeed ZeRO-offload (and plain hf).

Times a short SFT run (default 20 optimizer steps) and reports steady-state tokens/sec +
peak VRAM, so the "does DeepSpeed actually beat unsloth on a single 6GB RTX 2060?" question
is answered with numbers rather than assumption. On a single small GPU the CPU-offload
transfer overhead usually loses to unsloth's fused kernels — this measures it.

    # baseline (fused kernels)
    python scripts/bench_train.py --backend unsloth
    # DeepSpeed ZeRO-2 + optimizer offload (must use a non-unsloth backend)
    deepspeed scripts/bench_train.py --backend bnb \
        --deepspeed configs/training/deepspeed_zero2_offload.json

Pure throughput maths lives in ko_dialect.training.bench (unit-tested). Feed the printed
tokens/sec of each backend into compare_throughput for the verdict.
"""

from __future__ import annotations

import logging
import time

import fire

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(
    backend: str = "unsloth",
    deepspeed: str | None = None,
    max_steps: int = 20,
    sft_dataset_path: str = "outputs/datasets/sft",
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    max_seq_length: int = 512,
) -> None:
    """Run a short SFT and print steady-state tokens/sec + peak VRAM for ``backend``."""
    import torch
    from datasets import load_from_disk
    from transformers import TrainerCallback

    from ko_dialect.training.bench import summarize_train_throughput
    from ko_dialect.training.sft_trainer import SFTConfig, load_model_and_tokenizer

    cfg = SFTConfig(
        backend=backend,
        deepspeed=deepspeed,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
        num_train_epochs=1,
        gradient_checkpointing=backend != "unsloth",
        save_merged=False,
        eval_strategy="no",
    )
    model, tokenizer = load_model_and_tokenizer(cfg)
    # Cast stray bf16 tensors to fp16 (Turing has no native BF16) — mirrors sft_trainer.train
    # so non-unsloth backends don't trip the fp16 grad-scaler.
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    train_ds = load_from_disk(sft_dataset_path)["train"].select(
        range(max_steps * per_device_train_batch_size * gradient_accumulation_steps + 8)
    )

    step_times: list[float] = []

    class _Timer(TrainerCallback):
        def on_step_begin(self, *a, **k):
            self._t = time.perf_counter()

        def on_step_end(self, args, state, control, **k):
            step_times.append(time.perf_counter() - self._t)
            if state.global_step >= max_steps:
                control.should_training_stop = True

    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    args = TRLSFTConfig(
        output_dir="outputs/_bench_train",
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=True,
        bf16=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        max_length=max_seq_length,
        packing=True,
        packing_strategy="bfd",  # match the real SFT path; avoids padding_free max_length error
        logging_steps=1,
        report_to="none",
        **({"deepspeed": deepspeed} if deepspeed else {}),
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train_ds, args=args)
    trainer.add_callback(_Timer())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None

    tokens_per_step = per_device_train_batch_size * gradient_accumulation_steps * max_seq_length
    summary = summarize_train_throughput(step_times, tokens_per_step, peak_vram_mb=peak)
    logger.info("backend=%s deepspeed=%s -> %s", backend, bool(deepspeed), summary)
    print(summary)


if __name__ == "__main__":
    fire.Fire(main)
