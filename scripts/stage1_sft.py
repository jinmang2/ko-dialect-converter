#!/usr/bin/env python3
"""Stage 1: QLoRA SFT on Qwen/Qwen2.5-0.5B-Instruct."""

from __future__ import annotations

try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import logging

import hydra
from datasets import load_from_disk
from omegaconf import DictConfig

from ko_dialect.config_utils import from_omegaconf
from ko_dialect.data import subsample_balanced_by_region
from ko_dialect.tracking import setup_tracking
from ko_dialect.training import SFTConfig, eval_budget_warning, train_sft

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="sft", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # training.* maps 1:1 onto SFTConfig fields; model.* and logger.* are remapped.
    sft_cfg = from_omegaconf(
        SFTConfig,
        cfg.training,
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        report_to=list(cfg.logger.report_to),
    )

    logger.info("Loading SFT dataset from %s", cfg.data.sft_dataset_path)
    ds = load_from_disk(cfg.data.sft_dataset_path)
    train_ds = ds["train"]
    if sft_cfg.balance_regions:
        train_ds = subsample_balanced_by_region(
            train_ds, sft_cfg.num_train_samples, seed=sft_cfg.seed
        )
    elif sft_cfg.num_train_samples and len(train_ds) > sft_cfg.num_train_samples:
        train_ds = train_ds.shuffle(seed=sft_cfg.seed).select(range(sft_cfg.num_train_samples))
    eval_ds = ds.get("valid")
    if eval_ds is not None and sft_cfg.num_eval_samples and len(eval_ds) > sft_cfg.num_eval_samples:
        # eval_ds = eval_ds.shuffle(seed=sft_cfg.seed).select(range(sft_cfg.num_eval_samples))
        eval_ds = eval_ds.select(range(sft_cfg.num_eval_samples))  # do not shuffle

    if eval_ds is not None:
        global_batch = max(
            1, sft_cfg.per_device_train_batch_size * sft_cfg.gradient_accumulation_steps
        )
        # Approximate: with packing the real step count is lower, which only makes the
        # warning conservative (it under-reports the eval share, never over-reports).
        steps_per_epoch = max(1, len(train_ds) // global_batch)
        warning = eval_budget_warning(
            sft_cfg.num_eval_samples,
            len(eval_ds),
            int(steps_per_epoch * sft_cfg.num_train_epochs),
            sft_cfg.eval_steps if sft_cfg.eval_strategy == "steps" else steps_per_epoch,
        )
        if warning:
            logger.warning("%s", warning)

    with setup_tracking(cfg.logger, cfg.experiment):
        logger.info("SFTConfig: %s", sft_cfg)
        train_sft(sft_cfg, train_ds, eval_ds)


if __name__ == "__main__":
    main()
