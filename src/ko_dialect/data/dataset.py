from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

from .filtering import carries_dialect_marker, norm_levenshtein
from .prosody import add_sentence_final_marker
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
    include_prosody: bool = False,
) -> DatasetDict:
    """Build SFT dataset from raw dialect DatasetDict.

    Each raw row expands to 1 (dia2std only) or 2 (``both_directions``) examples.

    ``output_mode`` controls how each example is stored:

    - ``"text"`` (default, no regression): the chat template is rendered at build time
      into a ``text`` column. Fast, but changing the template means rebuilding.
    - ``"structured"``: store raw ``source,target,do,direction`` and apply the template
      *at train time*. Lets you swap templates / loss-masking without rebuilding data.

    ``include_prosody`` is an opt-in experiment path. When a raw row has a
    ``prosody_marker`` (for example ``"<UP>"``), the dialect side is rendered with
    that marker at SFT build time: as the target for ``std2dia`` and as the source for
    ``dia2std``. Missing markers are reported per split and fall back to plain dialect.
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
        missing_prosody_markers = 0
        for sample in split_ds:
            do = sample["do"]
            dialect_text = sample["dialect"]
            if include_prosody:
                marker = sample.get("prosody_marker")
                if marker:
                    dialect_text = add_sentence_final_marker(dialect_text, marker)
                else:
                    missing_prosody_markers += 1
            pairs: list[tuple[str, str, Direction]] = [
                (sample["standard"], dialect_text, "std2dia"),
            ]
            if both_directions:
                pairs.append((dialect_text, sample["standard"], "dia2std"))

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

        if include_prosody and missing_prosody_markers:
            logger.warning(
                "[%s] include_prosody=True but %d raw rows had no prosody_marker; "
                "those rows used plain dialect text.",
                split_name,
                missing_prosody_markers,
            )

        result[split_name] = Dataset.from_list(rows)
        logger.info(
            "[%s] %d SFT examples (output_mode=%s, include_prosody=%s)",
            split_name,
            len(rows),
            output_mode,
            include_prosody,
        )

    return DatasetDict(result)


def prosody_marker_coverage(
    dataset: DatasetDict | Dataset,
    filter_identical: bool = True,
) -> dict[str, dict[str, int | float]]:
    """Compute prosody marker coverage under the same row filters as SFT."""
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    coverage = {}
    for split_name, split_ds in dataset.items():
        if filter_identical:
            split_ds = split_ds.filter(lambda x: not x["is_identical"])
        split_ds = split_ds.filter(lambda x: x["do"] in SUPPORTED_DO)

        total = len(split_ds)
        marked = sum(1 for sample in split_ds if sample.get("prosody_marker"))
        coverage[split_name] = {
            "total_rows": total,
            "marked_rows": marked,
            "missing_rows": total - marked,
            "coverage_ratio": round(marked / total, 6) if total else 0.0,
        }
    return coverage


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


def _drop_label_collisions(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Remove every text that appears under more than one label.

    Same surface string carrying two labels is an unlearnable contradiction and
    directly poisons the ``P(dialect) - P(standard)`` reward. Both occurrences are
    dropped (we cannot know which label is correct).
    """
    labels_per_text: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        labels_per_text[r["text"]].add(r["label"])
    conflicted = {t for t, labels in labels_per_text.items() if len(labels) > 1}
    kept = [r for r in rows if r["text"] not in conflicted]
    return kept, len(rows) - len(kept)


def build_classification_dataset(
    dataset: DatasetDict | Dataset,
    filter_identical: bool = True,
    min_norm_levenshtein: float | None = 0.1,
    denoise_standard: bool = True,
    drop_collisions: bool = True,
    max_per_label: int | None = None,
    standard_cap_ratio: float | None = None,
    downsample_splits: tuple[str, ...] = ("train",),
    seed: int = 42,
) -> DatasetDict:
    """Build 3-class classification dataset (a GRPO reward model).

    Labels: standard=0, gangwondo=1, gyeongsangdo=2.

    The classifier is used as the GRPO style reward ``P(dialect) - P(standard)``, so
    its real job is to separate a genuine dialect translation (*True Attempt*) from an
    output that merely copies the standard source (*False Success*). Three quality
    gates, following DIA-REFINE (Park et al., arXiv:2511.06680), keep that boundary
    clean — applied to **every split** because the reward sees the same distribution:

    - ``filter_identical`` — drop the dialect copy when ``standard == dialect``.
    - ``min_norm_levenshtein`` — admit a dialect sample only when the normalized
      char-level Levenshtein distance to its standard form is ``>=`` this threshold
      (paper uses ``0.1``, "ensuring form-level divergence"). This removes the
      near-standard dialect rows that would teach the reward to credit source-copying.
      ``None`` disables.
    - ``denoise_standard`` — drop label-0 ``standard`` sentences that carry
      high-precision dialect endings (AI-Hub transcription noise; see
      :func:`carries_dialect_marker`), which otherwise blur the standard boundary.
    - ``drop_collisions`` — drop any text appearing under more than one label.

    Class imbalance (standard dominates) is controllable via ``max_per_label`` and
    ``standard_cap_ratio`` (see :func:`_downsample_rows`). Downsampling is applied
    only to splits in ``downsample_splits`` (default: ``train`` only, so the valid
    split keeps the true post-filter label distribution for honest evaluation).
    """
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    result: dict[str, Dataset] = {}
    for split_name, split_ds in dataset.items():
        rows: list[dict[str, Any]] = []
        n_identical = n_near = n_polluted = 0
        for sample in split_ds:
            do = sample["do"]
            std_text = sample["standard"]
            if denoise_standard and carries_dialect_marker(std_text):
                n_polluted += 1
            else:
                rows.append({"text": std_text, "label": DIALECT_LABELS["standard"]})
            if do in SUPPORTED_DO:
                if filter_identical and sample.get("is_identical", std_text == sample["dialect"]):
                    n_identical += 1
                    continue
                if (
                    min_norm_levenshtein is not None
                    and norm_levenshtein(std_text, sample["dialect"]) < min_norm_levenshtein
                ):
                    n_near += 1
                    continue
                rows.append({"text": sample["dialect"], "label": DIALECT_LABELS[do]})

        n_built = len(rows)
        n_collisions = 0
        if drop_collisions:
            rows, n_collisions = _drop_label_collisions(rows)

        if split_name in downsample_splits and (
            max_per_label is not None or standard_cap_ratio is not None
        ):
            rows = _downsample_rows(rows, max_per_label, standard_cap_ratio, seed)

        result[split_name] = Dataset.from_list(rows)
        logger.info(
            "[%s] %d classification examples (built %d | dropped: %d identical, "
            "%d near-standard(<%.2f nLev), %d polluted-standard, %d collisions, "
            "%d downsampled)",
            split_name,
            len(rows),
            n_built,
            n_identical,
            n_near,
            min_norm_levenshtein or 0.0,
            n_polluted,
            n_collisions,
            n_built - n_collisions - len(rows),
        )

    return DatasetDict(result)
