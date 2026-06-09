#!/usr/bin/env python3
"""Stage 0: Build SFT and classifier datasets from a raw Arrow dialect dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import fire
from transformers import AutoTokenizer

from ko_dialect.data import (
    build_classification_dataset,
    build_grpo_dataset,
    build_sft_dataset,
    get_template,
    load_dialect_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(
    raw_dataset_path: str,
    output_dir: str = "outputs/datasets",
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    both_directions: bool = True,
    template_name: str = "default",
    output_mode: str = "text",
    cls_filter_identical: bool = True,
    cls_max_per_label: int | None = None,
    cls_standard_cap_ratio: float | None = None,
    seed: int = 42,
) -> None:
    """
    Args:
        raw_dataset_path: Path to dialect Arrow dataset saved by scripts/prepare_data.py.
        output_dir: Where to write the three output datasets (sft / grpo / classifier).
        model_name: Tokenizer to use for chat-template formatting.
        both_directions: Whether to generate both std→dia and dia→std SFT examples.
        template_name: Registered chat template (see data/template.py).
        output_mode: "text" (pre-render chat string) or "structured" (store raw fields so
            the template/loss-mask can be swapped at train time without rebuilding).
        cls_filter_identical: Drop the dialect copy when standard == dialect (no markers).
            Leave True — disabling reintroduces ~20% contradictory labels.
        cls_max_per_label: Hard cap on rows per classifier label (None = no cap).
        cls_standard_cap_ratio: Cap the standard class to this multiple of the largest
            dialect class, e.g. 2.0 (None = no cap). Train split only.
        seed: RNG seed for classifier downsampling.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw dataset from %s", raw_dataset_path)
    dataset = load_dialect_dataset(raw_dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template = get_template(template_name)

    logger.info("Building SFT dataset (mode=%s, template=%s) ...", output_mode, template_name)
    sft_ds = build_sft_dataset(
        dataset,
        tokenizer,
        template,
        both_directions=both_directions,
        output_mode=output_mode,
    )
    sft_ds.save_to_disk(str(out / "sft"))
    logger.info("SFT saved: %s", sft_ds)

    logger.info("Building GRPO prompt dataset ...")
    grpo_ds = build_grpo_dataset(dataset, tokenizer, template, direction="std2dia")
    grpo_ds.save_to_disk(str(out / "grpo"))
    logger.info("GRPO saved: %s", grpo_ds)

    logger.info("Building classifier dataset ...")
    cls_ds = build_classification_dataset(
        dataset,
        filter_identical=cls_filter_identical,
        max_per_label=cls_max_per_label,
        standard_cap_ratio=cls_standard_cap_ratio,
        seed=seed,
    )
    cls_ds.save_to_disk(str(out / "classifier"))
    logger.info("Classifier saved: %s", cls_ds)


if __name__ == "__main__":
    fire.Fire(main)
