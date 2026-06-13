#!/usr/bin/env python3
"""Push a chosen dialect model to the Hugging Face Hub with an evidence-rich card.

Generates a model card from a leaderboard record (per-region metrics with direction
arrows, Pareto flags, paired-bootstrap significance vs SFT) and uploads the merged model
folder. ``--dry_run`` (the default) writes the card next to the model and uploads
nothing, so you can review before publishing.

Authentication for a real push: ``huggingface-cli login`` or set ``HF_TOKEN``.

Usage:
    # 1) merge the run you decided to ship (GRPO adapter onto the SFT base):
    python scripts/merge_sft_lora.py --adapter_path outputs/grpo_arm2 \
        --base_model outputs/sft_merged --out outputs/grpo_arm2_merged
    # 2) preview the card (no upload):
    python scripts/push_to_hub.py --model_path outputs/grpo_arm2_merged \
        --repo_id your-name/ko-dialect-arm2 --run_tag grpo_arm2 \
        --leaderboard outputs/eval_logs/leaderboard_overall_*.json
    # 3) publish for real:
    python scripts/push_to_hub.py ... --dry_run False
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import fire

from ko_dialect.evaluation.model_card import build_model_card

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("push_to_hub")


def _load_record(leaderboard: str | None) -> dict | None:
    if not leaderboard:
        return None
    matches = sorted(glob.glob(leaderboard))
    if not matches:
        logger.warning("No leaderboard JSON matched %s; card will omit evaluation.", leaderboard)
        return None
    path = matches[-1]
    logger.info("Embedding evaluation evidence from %s", path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(
    model_path: str,
    repo_id: str,
    run_tag: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    leaderboard: str | None = None,
    license: str = "apache-2.0",
    dry_run: bool = True,
    private: bool = False,
) -> None:
    """Write (and optionally upload) a model card + the merged model for ``run_tag``."""
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise SystemExit(f"Model path {model_dir} does not exist (merge it first).")

    record = _load_record(leaderboard)
    card = build_model_card(
        repo_id=repo_id,
        run_tag=run_tag,
        base_model=base_model,
        record=record,
        license=license,
    )
    card_path = model_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    logger.info("Wrote model card -> %s (%d chars)", card_path, len(card))

    if dry_run:
        print("\n[dry_run] Card written locally; nothing uploaded.")
        print(f"  Review: {card_path}")
        print(f"  To publish: rerun with --dry_run False  (repo_id={repo_id})")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    logger.info("Creating repo %s (private=%s)", repo_id, private)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    logger.info("Uploading %s ...", model_dir)
    api.upload_folder(repo_id=repo_id, folder_path=str(model_dir), repo_type="model")
    print(f"\nPublished: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    fire.Fire(main)
