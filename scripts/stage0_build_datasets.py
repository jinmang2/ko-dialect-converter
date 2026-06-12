#!/usr/bin/env python3
"""Stage 0: Build SFT and classifier datasets from a raw Arrow dialect dataset."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import fire
from transformers import AutoTokenizer

import ko_dialect.data as dialect_data
from ko_dialect.data import dataset as data_dataset
from ko_dialect.data.prosody import PROSODY_MARKER_POLICY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_raw_manifest(raw_dataset_path: str) -> dict | None:
    raw_path = Path(raw_dataset_path)
    manifest_path = raw_path.with_name(f"{raw_path.name}_manifest.json")
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_prosody_manifest(raw_manifest: dict | None) -> None:
    if raw_manifest is None:
        raise FileNotFoundError(
            "include_prosody=True requires a raw dataset manifest next to the raw "
            "dataset, e.g. dialect_raw_new_manifest.json."
        )

    raw_policy = raw_manifest.get("prosody_marker_policy")
    if not raw_policy:
        raise ValueError("include_prosody=True requires a raw prosody_marker_policy.")
    if raw_policy.get("version") != PROSODY_MARKER_POLICY["version"]:
        raise ValueError(
            "Raw prosody marker policy version does not match the current policy: "
            f"raw={raw_policy.get('version')!r}, "
            f"current={PROSODY_MARKER_POLICY['version']!r}."
        )


def main(
    raw_dataset_path: str,
    output_dir: str = "outputs/datasets",
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    both_directions: bool = True,
    template_name: str = "default",
    output_mode: str = "text",
    include_prosody: bool = False,
    cls_filter_identical: bool = True,
    cls_min_norm_levenshtein: float | None = 0.1,
    cls_denoise_standard: bool = True,
    cls_drop_collisions: bool = True,
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
        include_prosody: Render raw ``prosody_marker`` values into the dialect side of
            SFT examples. This is opt-in because existing text-only SFT data should not
            change by default.
        cls_filter_identical: Drop the dialect copy when standard == dialect (no markers).
            Leave True — disabling reintroduces ~20% contradictory labels.
        cls_min_norm_levenshtein: Admit a dialect sample only when normalized char-level
            Levenshtein(standard, dialect) >= this (DIA-REFINE uses 0.1). Removes
            near-standard rows that teach the reward to credit source-copying. None
            disables. Applied to all splits.
        cls_denoise_standard: Drop label-0 standard sentences carrying high-precision
            dialect endings (AI-Hub transcription noise).
        cls_drop_collisions: Drop any text that appears under more than one label.
        cls_max_per_label: Hard cap on rows per classifier label (None = no cap).
        cls_standard_cap_ratio: Cap the standard class to this multiple of the largest
            dialect class, e.g. 2.0 (None = no cap). Train split only.
        seed: RNG seed for classifier downsampling.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw dataset from %s", raw_dataset_path)
    raw_manifest = load_raw_manifest(raw_dataset_path)
    if include_prosody:
        validate_prosody_manifest(raw_manifest)
    dataset = dialect_data.load_dialect_dataset(raw_dataset_path)
    prosody_marker_coverage = (
        data_dataset.prosody_marker_coverage(dataset) if include_prosody else None
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template = dialect_data.get_template(template_name)

    logger.info(
        "Building SFT dataset (mode=%s, template=%s, include_prosody=%s) ...",
        output_mode,
        template_name,
        include_prosody,
    )
    sft_ds = dialect_data.build_sft_dataset(
        dataset,
        tokenizer,
        template,
        both_directions=both_directions,
        output_mode=output_mode,
        include_prosody=include_prosody,
    )
    sft_ds.save_to_disk(str(out / "sft"))
    (out / "sft_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "sft",
                "template_name": template_name,
                "output_mode": output_mode,
                "both_directions": both_directions,
                "include_prosody": include_prosody,
                "prosody_marker_policy": (
                    PROSODY_MARKER_POLICY if include_prosody else None
                ),
                "prosody_marker_coverage": prosody_marker_coverage,
                "raw_manifest": raw_manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("SFT saved: %s", sft_ds)

    logger.info("Building GRPO prompt dataset ...")
    grpo_ds = dialect_data.build_grpo_dataset(
        dataset, tokenizer, template, direction="std2dia"
    )
    grpo_ds.save_to_disk(str(out / "grpo"))
    logger.info("GRPO saved: %s", grpo_ds)

    logger.info("Building classifier dataset ...")
    cls_ds = dialect_data.build_classification_dataset(
        dataset,
        filter_identical=cls_filter_identical,
        min_norm_levenshtein=cls_min_norm_levenshtein,
        denoise_standard=cls_denoise_standard,
        drop_collisions=cls_drop_collisions,
        max_per_label=cls_max_per_label,
        standard_cap_ratio=cls_standard_cap_ratio,
        seed=seed,
    )
    cls_ds.save_to_disk(str(out / "classifier"))
    logger.info("Classifier saved: %s", cls_ds)


if __name__ == "__main__":
    fire.Fire(main)
