"""Tests for config-driven evaluation defaults (EvalConfig)."""

from __future__ import annotations

import textwrap

from ko_dialect.evaluation.config import EvalConfig


def test_load_defaults_when_no_file(tmp_path):
    cfg = EvalConfig.load(tmp_path / "missing.yaml")
    assert cfg.n_samples == 150
    assert cfg.regions == ["gangwondo", "gyeongsangdo"]


def test_load_reads_yaml(tmp_path):
    p = tmp_path / "eval.yaml"
    p.write_text(
        textwrap.dedent(
            """
            n_samples: 300
            split: test
            regions: [gangwondo]
            """
        )
    )
    cfg = EvalConfig.load(p)
    assert cfg.n_samples == 300
    assert cfg.split == "test"
    assert cfg.regions == ["gangwondo"]


def test_overrides_win_and_none_is_ignored(tmp_path):
    p = tmp_path / "eval.yaml"
    p.write_text("n_samples: 300\nbatch_size: 16\n")
    cfg = EvalConfig.load(p, n_samples=50, batch_size=None)
    assert cfg.n_samples == 50  # explicit override wins
    assert cfg.batch_size == 16  # None override ignored -> keeps yaml value


def test_unknown_yaml_keys_ignored(tmp_path):
    p = tmp_path / "eval.yaml"
    p.write_text("n_samples: 7\nbogus_key: 123\n")
    cfg = EvalConfig.load(p)
    assert cfg.n_samples == 7
    assert not hasattr(cfg, "bogus_key")


def test_repo_default_config_is_loadable():
    """The shipped configs/eval/default.yaml must parse into a valid EvalConfig."""
    cfg = EvalConfig.load("configs/eval/default.yaml")
    assert cfg.base_model
    assert cfg.regions
