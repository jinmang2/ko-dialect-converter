#!/usr/bin/env python3
"""Evaluation: compute TDR / DFS / eojeol accuracy for a trained model."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import fire
from datasets import load_from_disk
from transformers import AutoTokenizer

from ko_dialect.data import ChatTemplate
from ko_dialect.evaluation import (
    EvalConfig,
    evaluate_all,
    generate_batched,
    load_generation_model,
)
from ko_dialect.models import TextCNNForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(
    model_path: str,
    raw_dataset_path: str | None = None,
    classifier_path: str | None = None,
    cls_tokenizer_name: str | None = None,
    base_model: str | None = None,
    target_do: str = "gangwondo",
    split: str | None = None,
    n_samples: int | None = None,
    max_new_tokens: int | None = None,
    batch_size: int | None = None,
    output_file: str | None = None,
    config: str | None = None,
) -> None:
    """Evaluate a single model: TDR / DFS / eojeol accuracy + surface metrics.

    Defaults come from ``configs/eval/default.yaml`` (override ``config`` to point
    elsewhere); any explicit flag wins over the config value.

    Args:
        model_path: Trained causal LM (SFT/GRPO output dir or a specific checkpoint).
            A Trainer dir holding only checkpoint-* subdirs resolves to its latest.
        raw_dataset_path: Original dialect Arrow dataset (gold references).
        classifier_path: TextCNN classifier (output of stage2) for tdr/dfs.
        cls_tokenizer_name: Tokenizer matching the classifier.
        base_model: Base for LoRA adapters (fp16); ignored for full/merged models.
        target_do: Dialect to evaluate (gangwondo | gyeongsangdo | ...).
        split: Dataset split to evaluate on.
        n_samples: Samples to evaluate (0 = all).
        max_new_tokens: Max tokens generated per sample.
        batch_size: Prompts per forward pass (16-32 fits a 6GB RTX 2060).
        output_file: Optional JSON file to write metric results.
        config: Path to an eval YAML (defaults to configs/eval/default.yaml).
    """
    cfg = EvalConfig.load(
        config,
        raw_dataset_path=raw_dataset_path,
        classifier_path=classifier_path,
        cls_tokenizer_name=cls_tokenizer_name,
        split=split,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    logger.info("Loading generation model from %s", model_path)
    model, tokenizer = load_generation_model(model_path, base_model)
    device = next(model.parameters()).device

    cls_tokenizer = AutoTokenizer.from_pretrained(cfg.cls_tokenizer_name)
    if cls_tokenizer.pad_token is None:
        cls_tokenizer.pad_token = cls_tokenizer.eos_token

    classifier = TextCNNForSequenceClassification.from_pretrained(cfg.classifier_path)
    classifier.eval().to(device)

    logger.info("Loading eval dataset from %s [%s]", cfg.raw_dataset_path, cfg.split)
    ds = load_from_disk(cfg.raw_dataset_path)
    eval_ds = ds[cfg.split].filter(lambda x: x["do"] == target_do and not x["is_identical"])
    if cfg.n_samples and cfg.n_samples < len(eval_ds):
        eval_ds = eval_ds.select(range(cfg.n_samples))
    logger.info("Evaluating on %d samples (batch_size=%d)", len(eval_ds), cfg.batch_size)

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
        device=device,
        batch_size=cfg.batch_size,
        max_new_tokens=cfg.max_new_tokens,
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
