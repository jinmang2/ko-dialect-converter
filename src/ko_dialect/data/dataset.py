from __future__ import annotations

import logging
import random
from collections import defaultdict
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


def _downsample_rows(
    rows: list[dict[str, Any]],
    max_per_label: int | None,
    standard_cap_ratio: float | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Subsample ``rows`` so each label respects an optional cap.

    - ``standard_cap_ratio``: cap the standard class (label 0) to
      ``ratio * max(count of each dialect label)``. ``2.0`` keeps standard at
      most 2x the largest dialect class. ``None`` disables.
    - ``max_per_label``: hard absolute cap applied to *every* label. ``None``
      disables. Applied after ``standard_cap_ratio``.
    """
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    counts = {label: len(items) for label, items in by_label.items()}

    caps = dict(counts)
    if standard_cap_ratio is not None:
        dialect_max = max(
            (c for label, c in counts.items() if label != DIALECT_LABELS["standard"]),
            default=0,
        )
        std = DIALECT_LABELS["standard"]
        if std in caps:
            caps[std] = min(caps[std], int(standard_cap_ratio * dialect_max))
    if max_per_label is not None:
        for label in caps:
            caps[label] = min(caps[label], max_per_label)

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for label, items in by_label.items():
        if caps[label] < len(items):
            items = rng.sample(items, caps[label])
        out.extend(items)
    rng.shuffle(out)
    return out


def build_classification_dataset(
    dataset: DatasetDict | Dataset,
    filter_identical: bool = True,
    max_per_label: int | None = None,
    standard_cap_ratio: float | None = None,
    downsample_splits: tuple[str, ...] = ("train",),
    seed: int = 42,
) -> DatasetDict:
    """Build 3-class classification dataset.

    Labels: standard=0, gangwondo=1, gyeongsangdo=2.

    The ``standard`` sentence of each row is emitted as label 0. The ``dialect``
    sentence is emitted as its regional label *only when it actually differs from
    the standard form*. When ``standard == dialect`` (``is_identical``) the
    utterance carries no dialectal markers, so emitting it under a dialect label
    would make the same surface string carry two labels — an unlearnable
    contradiction that also poisons the GRPO style reward (which scores
    ``P(dialect) - P(standard)``). ``filter_identical`` drops that dialect copy,
    matching the SFT/GRPO builders.

    Class imbalance (standard dominates) is controllable via ``max_per_label`` and
    ``standard_cap_ratio`` (see :func:`_downsample_rows`). Downsampling is applied
    only to splits in ``downsample_splits`` (default: ``train`` only, so the valid
    split keeps the true label distribution for honest evaluation). Pair this with
    class-weighted loss on the model side for the strongest balance.
    """
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    result: dict[str, Dataset] = {}
    for split_name, split_ds in dataset.items():
        rows: list[dict[str, Any]] = []
        n_skipped = 0
        for sample in split_ds:
            do = sample["do"]
            rows.append({"text": sample["standard"], "label": DIALECT_LABELS["standard"]})
            if do in SUPPORTED_DO:
                if filter_identical and sample.get(
                    "is_identical", sample["standard"] == sample["dialect"]
                ):
                    n_skipped += 1
                    continue
                rows.append({"text": sample["dialect"], "label": DIALECT_LABELS[do]})

        n_built = len(rows)
        if split_name in downsample_splits and (
            max_per_label is not None or standard_cap_ratio is not None
        ):
            rows = _downsample_rows(rows, max_per_label, standard_cap_ratio, seed)

        result[split_name] = Dataset.from_list(rows)
        logger.info(
            "[%s] %d classification examples (built %d, skipped %d identical, "
            "downsampled %d)",
            split_name,
            len(rows),
            n_built,
            n_skipped,
            n_built - len(rows),
        )

    return DatasetDict(result)
