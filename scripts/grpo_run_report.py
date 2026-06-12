#!/usr/bin/env python3
"""Summarize GRPO trainer-state health across completed runs."""

from __future__ import annotations

from pathlib import Path

import fire

from ko_dialect.evaluation.grpo_runs import summarize_trainer_state


def main(*paths: str, zero_std_threshold: float = 0.3) -> None:
    """Print a compact health summary for one or more `trainer_state.json` files."""
    if not paths:
        paths = tuple(
            str(path)
            for path in sorted(Path("outputs").glob("grpo*/checkpoint-*/trainer_state.json"))
        )
    for path in paths:
        summary = summarize_trainer_state(path, zero_std_threshold=zero_std_threshold)
        print(summary)


if __name__ == "__main__":
    fire.Fire(main)
