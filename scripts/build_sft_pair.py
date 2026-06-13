#!/usr/bin/env python3
"""Build a matched prosody/control SFT dataset pair for the prosody A/B experiment.

Both datasets come from the *same* raw rows and the *same* template/config — they differ
only in whether the dialect side carries its ``prosody_marker`` (<UP>/<DOWN>/<KEEP>). That
single-variable difference is what makes the downstream A/B (does prosody supervision help
dialect fidelity?) clean.

Usage:
    python scripts/build_sft_pair.py --raw outputs/dialect_raw_prosody \
        --prosody_out outputs/datasets_prosody/sft \
        --control_out outputs/datasets_control/sft
"""

from __future__ import annotations

import logging
from pathlib import Path

import fire
from transformers import AutoTokenizer

import ko_dialect.data as dialect_data
from ko_dialect.data.dataset import load_dialect_dataset, prosody_marker_coverage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_sft_pair")


def main(
    raw: str = "outputs/dialect_raw_prosody",
    prosody_out: str = "outputs/datasets_prosody/sft",
    control_out: str = "outputs/datasets_control/sft",
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    template_name: str = "default",
    output_mode: str = "text",
    both_directions: bool = True,
) -> None:
    """Build the prosody (markers) and control (no markers) SFT datasets."""
    dataset = load_dialect_dataset(raw)
    logger.info("Marker coverage: %s", prosody_marker_coverage(dataset))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    template = dialect_data.get_template(template_name)

    for include_prosody, out in ((True, prosody_out), (False, control_out)):
        logger.info("Building SFT (include_prosody=%s) -> %s", include_prosody, out)
        sft = dialect_data.build_sft_dataset(
            dataset,
            tok,
            template,
            both_directions=both_directions,
            output_mode=output_mode,
            include_prosody=include_prosody,
        )
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        sft.save_to_disk(out)
        logger.info("Saved %s: %s", out, sft)


if __name__ == "__main__":
    fire.Fire(main)
