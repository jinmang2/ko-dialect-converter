#!/usr/bin/env python3
"""Evaluation: compute TDR / DFS / eojeol accuracy for a trained model."""
from __future__ import annotations

import json
import logging
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


def main(
    model_path: str,
    raw_dataset_path: str,
    classifier_path: str,
    cls_tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    target_do: str = "gangwondo",
    split: str = "valid",
    n_samples: int = 500,
    max_new_tokens: int = 128,
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
    logger.info("Evaluating on %d samples", len(eval_ds))

    template = ChatTemplate()
    outputs_text, dialect_refs, standard_refs, eojeol_maps = [], [], [], []

    for sample in eval_ds:
        prompt = template.build_prompt(tokenizer, sample["standard"], target_do, "std2dia")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        outputs_text.append(generated.strip())
        dialect_refs.append(sample["dialect"])
        standard_refs.append(sample["standard"])
        eojeol_maps.append(sample.get("dialect_eojeol_map") or [])

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
