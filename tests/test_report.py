"""Tests for the consolidated results-report renderer."""

from __future__ import annotations

from ko_dialect.evaluation.report import (
    build_report,
    classify_artifact,
    render_leaderboard,
    render_leaderboard_multi,
    render_quantization,
)


def _multi_v1_payload() -> dict:
    """Minimal ``ko_dialect.leaderboard_multi/v1`` record (overall + one region)."""
    flat = {
        "target_do": "OVERALL",
        "n_samples": 300,
        "select_by": "reconstruction_bleu",
        "best_run": "SFT",
        "ranking": ["SFT", "grpo_500"],
        "rows": {"SFT": {"reconstruction_bleu": 38.4}, "grpo_500": {"reconstruction_bleu": 37.1}},
    }
    region = {**flat, "target_do": "gangwondo", "n_samples": 150}
    return {
        "schema": "ko_dialect.leaderboard_multi/v1",
        "select_by": "reconstruction_bleu",
        "n_samples": 150,
        "regions": ["gangwondo"],
        "per_region": {"gangwondo": region},
        "overall": flat,
    }


def test_classify_artifact_recognises_kinds():
    assert classify_artifact({"ranking": [], "rows": {}}) == "leaderboard"
    assert classify_artifact({"baseline": "fp16", "rows": [{"name": "fp16"}]}) == "quantization"
    assert classify_artifact({"macro_f1": 0.95}) == "classifier"
    assert classify_artifact({"nope": 1}) is None
    assert classify_artifact("not a dict") is None


def test_render_leaderboard_marks_best_and_lists_ranking():
    data = {
        "target_do": "gangwondo",
        "n_samples": 150,
        "select_by": "reconstruction_bleu",
        "best_run": "SFT",
        "ranking": ["SFT", "grpo_500"],
        "rows": {
            "SFT": {"reconstruction_bleu": 38.4, "copy_margin": -4.97, "chrf": 68.6},
            "grpo_500": {"reconstruction_bleu": 37.9, "copy_margin": -2.1, "chrf": 66.0},
        },
    }
    md = render_leaderboard(data)
    assert "gangwondo" in md and "best: **SFT**" in md
    assert "SFT ★" in md  # best run marked
    assert "reconstruction_bleu↑" in md  # registry direction arrow
    assert md.index("SFT") < md.index("grpo_500")  # ranking order preserved


def test_classify_artifact_recognises_multi_leaderboard():
    # Regression (AUDIT H1): the current eval_leaderboard.py writes leaderboard_multi/v1
    # (nested per_region/overall, no top-level ranking/rows). It must NOT be dropped.
    assert classify_artifact(_multi_v1_payload()) == "leaderboard_multi"
    # Detection also works without the schema string, via per_region+overall presence.
    p = _multi_v1_payload()
    del p["schema"]
    assert classify_artifact(p) == "leaderboard_multi"


def test_render_leaderboard_multi_includes_overall_and_each_region():
    md = render_leaderboard_multi(_multi_v1_payload())
    assert "OVERALL" in md and "gangwondo" in md
    assert "SFT ★" in md  # best-run marker propagates through render_leaderboard


def test_build_report_renders_multi_leaderboard_section():
    md = build_report([("leaderboard_multi", _multi_v1_payload())])
    assert "Cross-run leaderboards (per-region + OVERALL)" in md
    assert "OVERALL" in md and "gangwondo" in md


def test_render_quantization_lists_variants():
    data = {"baseline": "fp16", "rows": [{"name": "fp16", "size_mb": 1024.0, "chrf": 68.5}]}
    md = render_quantization(data)
    assert "baseline: fp16" in md and "fp16" in md and "1024" in md


def test_render_quantization_shows_model_and_region_when_present():
    data = {
        "baseline": "fp16",
        "model": "outputs/sft_merged",
        "target_do": "gangwondo",
        "rows": [{"name": "fp16", "size_mb": 1024.0}],
    }
    md = render_quantization(data)
    assert "outputs/sft_merged" in md and "[gangwondo]" in md


def test_build_report_groups_and_skips_unknown():
    artifacts = [
        ("leaderboard", {"target_do": "x", "ranking": [], "rows": {}}),
        ("quantization", {"baseline": "fp16", "rows": []}),
        ("classifier", {"checkpoint": "c", "split": "valid", "macro_f1": 0.9, "accuracy": 0.92}),
    ]
    md = build_report(artifacts)
    assert "## Cross-run leaderboards" in md
    assert "## Quantization" in md
    assert "## Dialect classifier" in md
    assert "macro-F1 0.9" in md


def test_build_report_empty_is_stable():
    md = build_report([])
    assert "No recognised eval artifacts" in md
