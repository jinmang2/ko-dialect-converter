from __future__ import annotations

import json

import pytest

from ko_dialect.evaluation.leaderboard import (
    DIALECTNESS_AXIS,
    FIDELITY_AXIS,
    RunSpec,
    aggregate_rows,
    build_dia2std_prompt,
    build_dia2std_prompt_chatml,
    build_leaderboard_payload,
    deltas_vs_baseline,
    discover_runs,
    format_leaderboard_table,
    harmonic_joint,
    joint_score,
    pareto_frontier,
    rank_rows,
)


def _rows():
    return [
        ("SFT", {"reconstruction_bleu": 38.4, "copy_margin": -4.97, "tdr": 0.19}),
        ("grpo_500", {"reconstruction_bleu": 37.1, "copy_margin": -2.37, "tdr": 0.21}),
        ("grpo_arm1", {"reconstruction_bleu": 39.2, "copy_margin": 1.5, "tdr": 0.30}),
    ]


def test_rank_rows_sorts_descending_by_metric():
    ranked = rank_rows(_rows(), "reconstruction_bleu")
    assert [tag for tag, _ in ranked] == ["grpo_arm1", "SFT", "grpo_500"]


def test_rank_rows_missing_and_nan_sink_to_bottom():
    rows = [
        ("a", {"reconstruction_bleu": 10.0}),
        ("b", {}),  # missing
        ("c", {"reconstruction_bleu": float("nan")}),
        ("d", {"reconstruction_bleu": 20.0}),
    ]
    ranked = rank_rows(rows, "reconstruction_bleu")
    assert [tag for tag, _ in ranked][:2] == ["d", "a"]
    assert set(tag for tag, _ in ranked[2:]) == {"b", "c"}


def test_rank_rows_is_stable_on_ties():
    rows = [("x", {"m": 1.0}), ("y", {"m": 1.0}), ("z", {"m": 1.0})]
    assert [t for t, _ in rank_rows(rows, "m")] == ["x", "y", "z"]


def test_deltas_vs_baseline():
    d = deltas_vs_baseline(_rows(), "SFT")
    assert "SFT" not in d
    assert d["grpo_arm1"]["copy_margin"] == pytest.approx(1.5 - (-4.97))
    assert d["grpo_500"]["tdr"] == pytest.approx(0.21 - 0.19)


def test_deltas_vs_baseline_missing_baseline_returns_empty():
    assert deltas_vs_baseline(_rows(), "nope") == {}


def test_format_leaderboard_table_markdown_shape():
    ranked = rank_rows(_rows(), "reconstruction_bleu")
    table = format_leaderboard_table(ranked, metrics=("reconstruction_bleu", "tdr"))
    lines = table.splitlines()
    # headers carry direction arrows (both higher-is-better here)
    assert lines[0] == "| rank | run | reconstruction_bleu↑ | tdr↑ |"
    assert lines[1] == "|---|---|---|---|"
    # first data row is the winner at rank 1
    assert lines[2].startswith("| 1 | grpo_arm1 |")
    assert "n/a" not in table


def test_format_leaderboard_table_renders_na_for_missing():
    table = format_leaderboard_table([("x", {})], metrics=("tdr",))
    assert "n/a" in table


def test_build_leaderboard_payload_is_json_serialisable():
    ranked = rank_rows(_rows(), "reconstruction_bleu")
    payload = build_leaderboard_payload(
        ranked, select_by="reconstruction_bleu", target_do="gangwondo", n_samples=150
    )
    assert payload["best_run"] == "grpo_arm1"
    assert payload["ranking"][0] == "grpo_arm1"
    assert payload["select_by"] == "reconstruction_bleu"
    assert payload["n_samples"] == 150
    # round-trips through JSON without error
    assert json.loads(json.dumps(payload))["best_run"] == "grpo_arm1"


def test_discover_runs_finds_adapter_dirs_and_prepends_sft(tmp_path):
    base = tmp_path / "sft_merged"
    base.mkdir()
    for name in ("grpo_arm1", "grpo_arm2"):
        d = tmp_path / name
        d.mkdir()
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    # a non-adapter dir must be ignored
    (tmp_path / "grpo_logs").mkdir()

    specs = discover_runs(tmp_path, str(base))
    assert specs[0] == RunSpec("SFT", str(base), None)
    tags = [s.tag for s in specs]
    assert tags == ["SFT", "grpo_arm1", "grpo_arm2"]
    assert all(s.base == str(base) for s in specs)


def test_discover_runs_can_exclude_sft(tmp_path):
    base = tmp_path / "sft_merged"
    base.mkdir()
    d = tmp_path / "grpo_x"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    specs = discover_runs(tmp_path, str(base), include_sft=False)
    assert [s.tag for s in specs] == ["grpo_x"]


def test_build_dia2std_prompt_contains_text():
    p = build_dia2std_prompt("밥 뭇나?")
    assert "밥 뭇나?" in p
    assert "표준어" in p


class _ChatTemplateTokenizer:
    """Minimal stand-in that records the messages it was handed."""

    def __init__(self):
        self.seen: list[dict[str, str]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.seen = messages
        assert tokenize is False
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def test_chatml_reverse_prompt_matches_the_training_format():
    """The plain prompt drops everything training conditioned on; this one must not.

    The plain REVERSE_PROMPT has no ChatML wrapper, no system prompt and no region, so a
    score measured with it partly reflects out-of-format generalisation
    (docs/DATA_ANALYSIS.md §4.2). This is the faithful builder.
    """
    tok = _ChatTemplateTokenizer()
    prompt = build_dia2std_prompt_chatml(tok, "밥 뭇나?", "gyeongsangdo")

    roles = [m["role"] for m in tok.seen]
    assert roles == ["system", "user"]  # no assistant turn — it is a prompt
    assert tok.seen[0]["content"] == "당신은 한국어 방언 변환 전문가입니다."
    assert "경상도" in tok.seen[1]["content"]  # region injected, in Korean
    assert "밥 뭇나?" in tok.seen[1]["content"]
    assert prompt.endswith("<|im_start|>assistant\n")  # ready to generate


def test_chatml_and_plain_reverse_prompts_actually_differ():
    tok = _ChatTemplateTokenizer()
    text = "밥 뭇나?"
    plain = build_dia2std_prompt(text)
    chatml = build_dia2std_prompt_chatml(tok, text, "gyeongsangdo")

    assert plain != chatml
    assert "<|im_start|>" not in plain  # the gap the format_sensitivity metric measures
    assert "경상도" not in plain


# --- Pareto frontier & joint score (multi-axis view) ---


def test_pareto_frontier_keeps_nondominated_tradeoff():
    rows = [
        ("SFT", {DIALECTNESS_AXIS: -5.0, FIDELITY_AXIS: 38.0}),  # max fidelity
        ("arm2", {DIALECTNESS_AXIS: -2.0, FIDELITY_AXIS: 37.0}),  # max dialectness
        ("dominated", {DIALECTNESS_AXIS: -6.0, FIDELITY_AXIS: 36.0}),  # worse on both
    ]
    frontier = set(pareto_frontier(rows))
    assert frontier == {"SFT", "arm2"}
    assert "dominated" not in frontier


def test_pareto_frontier_single_dominator():
    rows = [
        ("best", {DIALECTNESS_AXIS: 5.0, FIDELITY_AXIS: 40.0}),
        ("a", {DIALECTNESS_AXIS: 1.0, FIDELITY_AXIS: 30.0}),
        ("b", {DIALECTNESS_AXIS: 4.0, FIDELITY_AXIS: 39.0}),
    ]
    assert pareto_frontier(rows) == ["best"]


def test_pareto_frontier_handles_missing_axis():
    rows = [("x", {DIALECTNESS_AXIS: 1.0}), ("y", {FIDELITY_AXIS: 1.0})]
    # neither dominates the other (each missing the other's axis -> -inf)
    assert set(pareto_frontier(rows)) == {"x", "y"}


def test_harmonic_joint_punishes_imbalance():
    balanced = harmonic_joint([0.5, 0.5])
    imbalanced = harmonic_joint([0.9, 0.1])
    assert balanced == pytest.approx(0.5)
    assert imbalanced < balanced
    assert harmonic_joint([0.5, 0.0]) == 0.0
    assert harmonic_joint([]) == 0.0


def test_joint_score_in_unit_range():
    m = {DIALECTNESS_AXIS: 0.0, FIDELITY_AXIS: 50.0}
    js = joint_score(m)
    assert 0.0 < js <= 1.0


def test_payload_v2_has_pareto_and_joint():
    rows = [
        ("SFT", {FIDELITY_AXIS: 38.0, DIALECTNESS_AXIS: -5.0}),
        ("arm2", {FIDELITY_AXIS: 37.0, DIALECTNESS_AXIS: -2.0}),
    ]
    ranked = rank_rows(rows, FIDELITY_AXIS)
    payload = build_leaderboard_payload(
        ranked, select_by=FIDELITY_AXIS, target_do="gangwondo", n_samples=150
    )
    assert payload["schema"] == "ko_dialect.leaderboard/v2"
    assert set(payload["pareto_frontier"]) == {"SFT", "arm2"}
    assert "joint_score" in payload and set(payload["joint_score"]) == {"SFT", "arm2"}
    assert json.loads(json.dumps(payload))["pareto_axes"] == [DIALECTNESS_AXIS, FIDELITY_AXIS]


def test_format_table_header_has_direction_arrows():
    table = format_leaderboard_table([("SFT", {"copy_margin": -5.0})], metrics=("copy_margin",))
    assert "copy_margin↑" in table  # higher-is-better arrow surfaced in header


# --- multi-region aggregation ---


def test_aggregate_rows_weighted_mean():
    per_region = {
        "gangwondo": [("SFT", {"reconstruction_bleu": 40.0, "copy_margin": -5.0})],
        "gyeongsangdo": [("SFT", {"reconstruction_bleu": 30.0, "copy_margin": 1.0})],
    }
    # weights 100 vs 300 -> weighted recon = (40*100 + 30*300)/400 = 32.5
    agg = aggregate_rows(per_region, weights={"gangwondo": 100, "gyeongsangdo": 300})
    tag, m = agg[0]
    assert tag == "SFT"
    assert m["reconstruction_bleu"] == pytest.approx(32.5)
    assert m["copy_margin"] == pytest.approx((-5.0 * 100 + 1.0 * 300) / 400)
    assert m["n_regions"] == 2


def test_aggregate_rows_equal_weight_default():
    per_region = {
        "a": [("x", {"reconstruction_bleu": 10.0})],
        "b": [("x", {"reconstruction_bleu": 20.0})],
    }
    agg = dict((t, m) for t, m in aggregate_rows(per_region))
    assert agg["x"]["reconstruction_bleu"] == pytest.approx(15.0)


def test_aggregate_rows_run_missing_in_one_region():
    per_region = {
        "a": [("SFT", {"reconstruction_bleu": 10.0}), ("arm2", {"reconstruction_bleu": 12.0})],
        "b": [("SFT", {"reconstruction_bleu": 20.0})],  # arm2 absent here
    }
    agg = dict((t, m) for t, m in aggregate_rows(per_region))
    assert agg["arm2"]["reconstruction_bleu"] == pytest.approx(12.0)
    assert agg["arm2"]["n_regions"] == 1
    assert agg["SFT"]["n_regions"] == 2


def test_format_table_flags_frontier_with_star():
    ranked = [
        ("SFT", {"reconstruction_bleu": 38.0}),
        ("arm2", {"reconstruction_bleu": 37.0}),
    ]
    table = format_leaderboard_table(ranked, metrics=("reconstruction_bleu",), frontier=["arm2"])
    assert "arm2 ★" in table
    assert "| SFT |" in table  # not flagged
