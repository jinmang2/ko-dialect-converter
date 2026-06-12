from __future__ import annotations

from ko_dialect.evaluation.model_card import build_model_card

_RECORD = {
    "schema": "ko_dialect.leaderboard_multi/v1",
    "select_by": "reconstruction_bleu",
    "regions": ["gangwondo", "gyeongsangdo"],
    "glossary_markdown": "| metric | dir |\n|---|---|\n| `copy_margin` | ↑ higher |",
    "per_region": {
        "gangwondo": {
            "rows": {
                "SFT": {"reconstruction_bleu": 38.4, "copy_margin": -5.0, "tdr": 0.19},
                "grpo_arm2": {"reconstruction_bleu": 37.1, "copy_margin": -3.4, "tdr": 0.19},
            },
            "pareto_frontier": ["SFT", "grpo_arm2"],
            "significance_vs_baseline": {
                "grpo_arm2": {"delta": -1.38, "p_value": 0.020, "significant_05": True}
            },
        },
        "gyeongsangdo": {
            "rows": {
                "SFT": {"reconstruction_bleu": 40.0, "copy_margin": -2.3, "tdr": 0.76},
                "grpo_arm2": {"reconstruction_bleu": 39.1, "copy_margin": 0.6, "tdr": 0.86},
            },
            "pareto_frontier": ["grpo_arm2"],
            "significance_vs_baseline": {
                "grpo_arm2": {"delta": -0.9, "p_value": 0.40, "significant_05": False}
            },
        },
    },
    "overall": {
        "rows": {"grpo_arm2": {"reconstruction_bleu": 38.0, "copy_margin": -1.4}},
        "pareto_frontier": ["grpo_arm2"],
        "significance_vs_baseline": {
            "grpo_arm2": {"delta": -1.1, "p_value": 0.06, "significant_05": False}
        },
    },
}


def test_card_has_frontmatter_and_sections():
    card = build_model_card(
        repo_id="me/ko-dialect-arm2",
        run_tag="grpo_arm2",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        record=_RECORD,
    )
    assert card.startswith("---\n")
    assert "language:\n  - ko" in card
    assert "## Evaluation" in card
    assert "## How to load" in card
    assert "me/ko-dialect-arm2" in card


def test_card_embeds_per_region_metrics_with_arrows():
    card = build_model_card(
        repo_id="me/x", run_tag="grpo_arm2",
        base_model="base", record=_RECORD,
    )
    assert "copy_margin↑" in card  # direction arrow surfaced
    assert "gangwondo" in card and "gyeongsangdo" in card
    assert "OVERALL" in card
    # gyeongsangdo arm2 is Pareto-optimal -> star present somewhere
    assert "★" in card


def test_card_significance_note_reflects_verdicts():
    card = build_model_card(
        repo_id="me/x", run_tag="grpo_arm2", base_model="base", record=_RECORD
    )
    assert "significant" in card
    assert "n.s." in card  # gyeongsangdo was not significant


def test_card_without_record_omits_eval_but_stays_valid():
    card = build_model_card(
        repo_id="me/x", run_tag="SFT", base_model="base", record=None
    )
    assert "## Evaluation" not in card
    assert "## Training" in card
    assert card.startswith("---\n")


def test_card_baseline_run_has_no_delta_note():
    card = build_model_card(
        repo_id="me/x", run_tag="SFT", base_model="base", record=_RECORD
    )
    # SFT is the baseline; significance dict has no SFT key -> placeholder text
    assert "baseline run" in card
