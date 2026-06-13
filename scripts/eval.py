#!/usr/bin/env python3
"""Unified evaluation entrypoint.

One documented front door over the specialised eval scripts. Each subcommand reads its
shared defaults from ``configs/eval/default.yaml`` (``ko_dialect.evaluation.EvalConfig``)
and forwards remaining flags to the underlying script's ``main``:

    python scripts/eval.py single     --model_path outputs/sft_merged --target_do gangwondo
    python scripts/eval.py leaderboard --all_regions
    python scripts/eval.py prosody_ab --control outputs/sft_control_s15 --prosody outputs/sft_prosody_s15
    python scripts/eval.py checkpoints --grpo_dir outputs/grpo_arm1
    python scripts/eval.py config                     # print resolved EvalConfig + metric registry

Subcommand mains are imported lazily so ``--help`` and ``config`` stay torch-free.
"""

from __future__ import annotations

import fire


def _sibling_main(module_filename: str):
    """Load ``main`` from a sibling script by explicit path.

    Avoids relying on ``scripts/`` being ``sys.path[0]`` and the name collision between
    ``scripts/evaluate.py`` and the HF ``evaluate`` PyPI package (a declared dependency).
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / module_filename
    spec = importlib.util.spec_from_file_location(f"_eval_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


class Eval:
    """Evaluation commands (see module docstring)."""

    def single(self, **kwargs):
        """Single-model metrics (TDR/DFS/eojeol + surface). Wraps scripts/evaluate.py."""
        return _sibling_main("evaluate.py")(**kwargs)

    def leaderboard(self, **kwargs):
        """Cross-run per-region + overall leaderboard. Wraps scripts/eval_leaderboard.py."""
        return _sibling_main("eval_leaderboard.py")(**kwargs)

    def prosody_ab(self, **kwargs):
        """Prosody control-vs-marker A/B. Wraps scripts/eval_prosody_ab.py."""
        return _sibling_main("eval_prosody_ab.py")(**kwargs)

    def checkpoints(self, **kwargs):
        """Within-run checkpoint sweep. Wraps scripts/eval_grpo_checkpoints.py."""
        return _sibling_main("eval_grpo_checkpoints.py")(**kwargs)

    def config(self, config: str | None = None):
        """Print the resolved EvalConfig and the metric registry (no model load)."""
        import json

        from ko_dialect.evaluation import EvalConfig
        from ko_dialect.evaluation.metric_registry import glossary_markdown

        cfg = EvalConfig.load(config)
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
        print("\nMetric registry (selection-relevant order):\n")
        print(glossary_markdown())


if __name__ == "__main__":
    fire.Fire(Eval)
