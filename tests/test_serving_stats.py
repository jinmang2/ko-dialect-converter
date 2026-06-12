from __future__ import annotations

import pytest

from ko_dialect.evaluation.serving import (
    directory_size_mb,
    format_tradeoff_table,
    percentile,
    quantization_tradeoff,
    summarize_latencies,
)


def test_percentile_endpoints_and_interpolation():
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert percentile(vals, 0) == 0.0
    assert percentile(vals, 100) == 4.0
    assert percentile(vals, 50) == 2.0
    assert percentile(vals, 25) == 1.0


def test_percentile_single_value():
    assert percentile([0.7], 95) == 0.7


def test_percentile_validates_inputs():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 150)


def test_summarize_latencies_basic_shape():
    lat = [0.10, 0.20, 0.30, 0.40]
    s = summarize_latencies(lat, new_tokens=[10, 20, 30, 40])
    assert s["n_requests"] == 4
    assert s["latency_ms_p50"] == pytest.approx(250.0)
    assert s["latency_ms_mean"] == pytest.approx(250.0)
    assert s["latency_ms_max"] == pytest.approx(400.0)
    # 4 requests over 1.0s total -> 4 req/s
    assert s["requests_per_sec"] == pytest.approx(4.0)
    # 100 tokens over 1.0s -> 100 tok/s
    assert s["tokens_per_sec"] == pytest.approx(100.0)


def test_summarize_latencies_without_tokens_omits_throughput_tokens():
    s = summarize_latencies([0.5, 0.5])
    assert "tokens_per_sec" not in s
    assert s["requests_per_sec"] == pytest.approx(2.0)


def test_summarize_latencies_requires_data():
    with pytest.raises(ValueError):
        summarize_latencies([])


def test_directory_size_mb_counts_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * (1024 * 1024))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * (512 * 1024))
    assert directory_size_mb(str(tmp_path)) == 1.5
    assert directory_size_mb(str(tmp_path / "a.bin")) == 1.0


def test_quantization_tradeoff_computes_deltas_vs_baseline():
    variants = [
        {
            "name": "fp16",
            "size_mb": 1000.0,
            "latency_ms_p50": 100.0,
            "chrf": 50.0,
            "copy_margin": -7.0,
        },
        {
            "name": "bnb4",
            "size_mb": 300.0,
            "latency_ms_p50": 80.0,
            "chrf": 48.5,
            "copy_margin": -7.4,
        },
    ]
    summary = quantization_tradeoff(variants, baseline="fp16")
    fp16, bnb4 = summary["rows"]
    assert "size_pct" not in fp16  # baseline carries no deltas
    assert bnb4["size_pct"] == 30.0
    assert bnb4["speedup"] == 1.25
    assert bnb4["chrf_drop"] == 1.5
    assert bnb4["copy_margin_drop"] == 0.4


def test_format_tradeoff_table_renders_markdown():
    summary = quantization_tradeoff(
        [
            {
                "name": "fp16",
                "size_mb": 1000.0,
                "latency_ms_p50": 100.0,
                "chrf": 50.0,
                "copy_margin": -7.0,
            }
        ],
        baseline="fp16",
    )
    table = format_tradeoff_table(summary)
    assert table.startswith("| variant |")
    assert "fp16" in table
