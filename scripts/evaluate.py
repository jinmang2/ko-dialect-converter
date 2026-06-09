#!/usr/bin/env python3
"""Evaluation: compute TDR / DFS / eojeol accuracy for a trained model."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import fire
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from ko_dialect.data import ChatTemplate
from ko_dialect.evaluation import evaluate_all
from ko_dialect.models import TextCNNForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_batched(
    model,
    tokenizer,
    prompts: list[str],
    device,
    batch_size: int = 16,
    max_new_tokens: int = 128,
) -> list[str]:
    """Greedy-decode ``prompts`` in batches.

    Decoder-only models require **left padding** for correct batched generation:
    right padding would push pad tokens between the prompt and the first generated
    token, corrupting the output. With left padding every row shares the same input
    length, so a single slice recovers the generated continuation for the whole batch.
    """
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_ids = gen_ids[:, enc["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            outputs.extend(text.strip() for text in decoded)
            logger.info("Generated %d / %d", len(outputs), len(prompts))
    finally:
        tokenizer.padding_side = prev_side
    return outputs


def main(
    model_path: str,
    raw_dataset_path: str,
    classifier_path: str,
    cls_tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    target_do: str = "gangwondo",
    split: str = "valid",
    n_samples: int = 500,
    max_new_tokens: int = 128,
    batch_size: int = 16,
    output_file: str | None = None,
) -> None:
    """
    Args:
        model_path: Path to the trained causal LM (SFT or GRPO output).
        raw_dataset_path: Path to original dialect Arrow dataset (for references).
        classifier_path: Path to TextCNN classifier (output of stage2).
        cls_tokenizer_name: Tokenizer matching the classifier.
        target_do: Dialect to evaluate (gangwondo | gyeongsangdo).
        split: Dataset split to evaluate on.
        n_samples: Number of samples to evaluate (0 = all).
        max_new_tokens: Max tokens to generate per sample.
        batch_size: Prompts decoded per forward pass. Larger = faster until VRAM
            saturates; on an 8 GB RTX 2060, 16-32 is a good starting point.
        output_file: Optional JSON file to write metric results.
    """
    logger.info("Loading generation model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    cls_tokenizer = AutoTokenizer.from_pretrained(cls_tokenizer_name)
    if cls_tokenizer.pad_token is None:
        cls_tokenizer.pad_token = cls_tokenizer.eos_token

    classifier = TextCNNForSequenceClassification.from_pretrained(classifier_path)
    classifier.eval().to(device)

    logger.info("Loading eval dataset from %s [%s]", raw_dataset_path, split)
    ds = load_from_disk(raw_dataset_path)
    eval_ds = ds[split].filter(
        lambda x: x["do"] == target_do and not x["is_identical"]
    )
    if n_samples and n_samples < len(eval_ds):
        eval_ds = eval_ds.select(range(n_samples))
    logger.info("Evaluating on %d samples (batch_size=%d)", len(eval_ds), batch_size)

    template = ChatTemplate()
    prompts = [
        template.build_prompt(tokenizer, sample["standard"], target_do, "std2dia")
        for sample in eval_ds
    ]
    dialect_refs = [s["dialect"] for s in eval_ds]
    standard_refs = [s["standard"] for s in eval_ds]
    eojeol_maps = [s.get("dialect_eojeol_map") or [] for s in eval_ds]

    t0 = time.perf_counter()
    outputs_text = generate_batched(
        model,
        tokenizer,
        prompts,
        device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "Generation done in %.1fs (%.3fs/sample)",
        elapsed,
        elapsed / max(len(prompts), 1),
    )

    results = evaluate_all(
        outputs=outputs_text,
        dialect_refs=dialect_refs,
        standard_refs=standard_refs,
        dialect_eojeol_maps=eojeol_maps,
        target_do=target_do,
        classifier=classifier,
        cls_tokenizer=cls_tokenizer,
    )

    logger.info("Results for %s: %s", target_do, results)
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Results written to %s", output_file)


if __name__ == "__main__":
    fire.Fire(main)
