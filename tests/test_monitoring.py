"""Tests for the W&B monitoring panel builders."""

from __future__ import annotations

from ko_dialect.monitoring import (
    log_panels,
    marker_distribution_panel,
    metrics_panel,
    wandb_run_active,
)


def test_marker_distribution_panel_shares_and_counts():
    panel = marker_distribution_panel({"<UP>": 19, "<KEEP>": 52, "<DOWN>": 23, "<WAVE>": 6})
    assert panel["prosody/total"] == 100.0
    assert panel["prosody/share/up"] == 0.19
    assert panel["prosody/share/keep"] == 0.52
    assert panel["prosody/count/wave"] == 6.0


def test_marker_distribution_handles_none_and_empty():
    panel = marker_distribution_panel({None: 3, "<UP>": 1})
    assert panel["prosody/share/none"] == 0.75
    empty = marker_distribution_panel({})
    assert empty["prosody/total"] == 0.0


def test_metrics_panel_flattens_numbers_only():
    panel = metrics_panel(
        {"copy_margin": -7.4, "tdr": 0.62, "name": "grpo_500", "passed": True},
        section="eval/gangwondo",
    )
    assert panel == {"eval/gangwondo/copy_margin": -7.4, "eval/gangwondo/tdr": 0.62}


def test_log_panels_noop_without_active_run():
    # No wandb run in the test process => safe no-op, returns False (never raises).
    assert wandb_run_active() is False
    assert log_panels({"eval/x": 1.0}) is False
    assert log_panels({}) is False
