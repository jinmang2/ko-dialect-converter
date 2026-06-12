from __future__ import annotations

import json
import math

import pytest

from ko_dialect.evaluation import grpo_runs


def test_checkpoint_step_parses_only_checkpoint_dirs():
    assert grpo_runs.checkpoint_step("outputs/grpo/checkpoint-200") == 200
    assert grpo_runs.checkpoint_step("outputs/grpo/not-a-checkpoint") is None
    assert grpo_runs.checkpoint_step("outputs/grpo/checkpoint-final") is None


def test_discover_grpo_stages_sorts_and_ignores_bad_checkpoint_names(tmp_path):
    root = tmp_path / "grpo"
    root.mkdir()
    (root / "checkpoint-200").mkdir()
    (root / "checkpoint-50").mkdir()
    (root / "checkpoint-final").mkdir()
    (root / "adapter_config.json").write_text("{}", encoding="utf-8")

    stages = grpo_runs.discover_grpo_stages(root)

    assert [(stage.tag, stage.adapter is None) for stage in stages] == [
        ("SFT(0)", True),
        ("step-50", False),
        ("step-200", False),
        ("final", False),
    ]


def test_clamp_select_count_rejects_negative_n():
    assert grpo_runs.clamp_select_count(10, 99) == 10
    assert grpo_runs.clamp_select_count(10, 4) == 4
    with pytest.raises(ValueError, match="non-negative"):
        grpo_runs.clamp_select_count(10, -1)


def test_best_by_metric_treats_missing_or_nan_as_default():
    rows = [
        ("missing", {}),
        ("nan", {"reconstruction_bleu": math.nan}),
        ("best", {"reconstruction_bleu": 12.0}),
    ]

    assert grpo_runs.metric_value({"x": math.nan}, "x") == 0.0
    assert grpo_runs.best_by_metric(rows, "reconstruction_bleu")[0] == "best"


def test_summarize_trainer_state_reports_nan_and_zero_std_steps(tmp_path):
    path = tmp_path / "trainer_state.json"
    path.write_text(
        json.dumps(
            {
                "global_step": 30,
                "log_history": [
                    {"step": 10, "grad_norm": 1.0, "frac_reward_zero_std": 0.0},
                    {"step": 20, "grad_norm": math.nan, "frac_reward_zero_std": 0.4},
                    {
                        "step": 30,
                        "grad_norm": 2.0,
                        "frac_reward_zero_std": 0.0,
                        "reward": 1.5,
                        "reward_std": 0.2,
                        "kl": 0.01,
                        "completions/clipped_ratio": 0.1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = grpo_runs.summarize_trainer_state(path, zero_std_threshold=0.3)

    assert summary["global_step"] == 30
    assert summary["last_step"] == 30
    assert summary["reward"] == 1.5
    assert summary["nan_grad_steps"] == [20]
    assert summary["high_zero_std_steps"] == [20]
