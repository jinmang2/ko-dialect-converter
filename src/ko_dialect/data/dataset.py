from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

from .template import ChatTemplate, Direction

logger = logging.getLogger(__name__)

DIALECT_LABELS: dict[str, int] = {
    "standard": 0,
    "gangwondo": 1,
    "gyeongsangdo": 2,
}

SUPPORTED_DO: set[str] = {"gangwondo", "gyeongsangdo"}


def load_dialect_dataset(path: str | Path) -> DatasetDict:
    return load_from_disk(str(path))


def build_sft_dataset(
    dataset: DatasetDict | Dataset,
    tokenizer,
    template: ChatTemplate | None = None,
    both_directions: bool = True,
    filter_identical: bool = True,
    output_mode: str = "text",
) -> DatasetDict:
    """Build SFT dataset from raw dialect DatasetDict.

    Each raw row expands to 1 (dia2std only) or 2 (``both_directions``) examples.

    ``output_mode`` controls how each example is stored:

    - ``"text"`` (default, no regression): the chat template is rendered at build time
      into a ``text`` column. Fast, but changing the template means rebuilding.
    - ``"structured"``: store raw ``source,target,do,direction`` and apply the template
      *at train time*. Lets you swap templates / loss-masking without rebuilding data.
    """
    if output_mode not in ("text", "structured"):
        raise ValueError(f"output_mode must be 'text' or 'structured', got {output_mode!r}.")
    if template is None:
        template = ChatTemplate()
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    result: dict[str, Dataset] = {}
    for split_name, split_ds in dataset.items():
        if filter_identical:
            split_ds = split_ds.filter(lambda x: not x["is_identical"])
        split_ds = split_ds.filter(lambda x: x["do"] in SUPPORTED_DO)

        rows: list[dict[str, Any]] = []
        for sample in split_ds:
            do = sample["do"]
            pairs: list[tuple[str, str, Direction]] = [
                (sample["standard"], sample["dialect"], "std2dia"),
            ]
            if both_directions:
                pairs.append((sample["dialect"], sample["standard"], "dia2std"))

            for source, target, direction in pairs:
                if output_mode == "structured":
                    rows.append(
                        {
                            "source": source,
                            "target": target,
                            "do": do,
                            "direction": direction,
                        }
                    )
                else:
                    text = template.apply(tokenizer, source, target, do, direction)
                    rows.append({"text": text, "do": do, "direction": direction})

        result[split_name] = Dataset.from_list(rows)
        logger.info("[%s] %d SFT examples (output_mode=%s)", split_name, len(rows), output_mode)

    return DatasetDict(result)


def build_grpo_dataset(
    dataset: DatasetDict | Dataset,
    tokenizer,
    template: ChatTemplate | None = None,
    direction: Direction = "std2dia",
    filter_identical: bool = True,
) -> DatasetDict:
    """Build prompt-only dataset for GRPO training.

    Columns: ``prompt``, ``do``, ``standard``, ``dialect``, ``dialect_eojeol_map``.
    """
    if template is None:
        template = ChatTemplate()
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    result: dict[str, Dataset] = {}
    for split_name, split_ds in dataset.items():
        if filter_identical:
            split_ds = split_ds.filter(lambda x: not x["is_identical"])
        split_ds = split_ds.filter(lambda x: x["do"] in SUPPORTED_DO)

        rows: list[dict[str, Any]] = []
        for sample in split_ds:
            do = sample["do"]
            source = sample["standard"] if direction == "std2dia" else sample["dialect"]
            prompt = template.build_prompt(tokenizer, source, do, direction)
            rows.append(
                {
                    "prompt": prompt,
                    "do": do,
                    "direction": direction,
                    "standard": sample["standard"],
                    "dialect": sample["dialect"],
                    "dialect_eojeol_map": sample.get("dialect_eojeol_map", []),
                }
            )

        result[split_name] = Dataset.from_list(rows)
        logger.info("[%s] %d GRPO prompt examples", split_name, len(rows))

    return DatasetDict(result)


def build_classification_dataset(
    dataset: DatasetDict | Dataset,
) -> DatasetDict:
    """Build 3-class classification dataset.

    Labels: standard=0, gangwondo=1, gyeongsangdo=2.
    Both the standard and dialect sentences from each row are included.
    """
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    result: dict[str, Dataset] = {}
    for split_name, split_ds in dataset.items():
        rows: list[dict[str, Any]] = []
        for sample in split_ds:
            do = sample["do"]
            rows.append({"text": sample["standard"], "label": DIALECT_LABELS["standard"]})
            if do in SUPPORTED_DO:
                rows.append({"text": sample["dialect"], "label": DIALECT_LABELS[do]})

        result[split_name] = Dataset.from_list(rows)
        logger.info("[%s] %d classification examples", split_name, len(rows))

    return DatasetDict(result)
