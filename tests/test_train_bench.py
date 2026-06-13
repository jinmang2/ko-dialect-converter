"""Tests for the training-throughput bench helpers."""

from __future__ import annotations

import pytest

from ko_dialect.training.bench import compare_throughput, summarize_train_throughput


def test_summarize_drops_warmup_and_computes_rates():
    # 1 warmup step (slow) + 4 steady steps at 0.5s each, 1000 tokens/step
    out = summarize_train_throughput([2.0, 0.5, 0.5, 0.5, 0.5], tokens_per_step=1000, warmup=1)
    assert out["n_steps_total"] == 5
    assert out["n_steps_measured"] == 4
    assert out["step_time_s_mean"] == 0.5
    assert out["steps_per_sec"] == 2.0
    assert out["tokens_per_sec"] == 2000.0


def test_summarize_includes_vram_when_given():
    out = summarize_train_throughput([0.5, 0.5], tokens_per_step=500, peak_vram_mb=4096.4)
    assert out["peak_vram_mb"] == 4096.4


def test_summarize_requires_data():
    with pytest.raises(ValueError):
        summarize_train_throughput([], tokens_per_step=10)


def test_compare_throughput_picks_winner_and_ratios():
    runs = {
        "unsloth": {"tokens_per_sec": 2000.0},
        "deepspeed": {"tokens_per_sec": 1200.0},
    }
    cmp = compare_throughput(runs)
    assert cmp["winner"] == "unsloth"
    assert cmp["ratio_to_winner"]["unsloth"] == 1.0
    assert cmp["ratio_to_winner"]["deepspeed"] == 0.6
