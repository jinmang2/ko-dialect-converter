from __future__ import annotations

import pytest

from ko_dialect.evaluation.serving import percentile, summarize_latencies


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
