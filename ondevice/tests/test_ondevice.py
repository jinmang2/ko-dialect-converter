"""Offline regression tests for the on-device harness (no phone, no GPU).

Covers the parts that must not silently drift: llama-bench parsing, the §6 schema
round-trip, the Pareto dominance, and — most important — that the demo / eval ChatML
prompt is byte-identical to the canonical training template.

    pytest ondevice/tests/test_ondevice.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_OND = _HERE.parents[1]
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_OND / "bench"))
sys.path.insert(0, str(_OND / "report"))
sys.path.insert(0, str(_OND / "eval"))
sys.path.insert(0, str(_REPO / "src"))


def test_parse_bench_json_and_markdown():
    from parse_bench import parse_text, summarize

    js = (
        '[{"n_threads":6,"n_prompt":64,"n_gen":0,"avg_ts":210.5},'
        '{"n_threads":6,"n_prompt":0,"n_gen":128,"avg_ts":41.2}]'
    )
    s = summarize(parse_text(js))
    assert s["prefill_tok_s"] == 210.5 and s["decode_tok_s"] == 41.2
    assert s["n_prompt_tokens"] == 64 and s["n_gen_tokens"] == 128

    md = (
        "| model | test | t/s |\n"
        "| ----- | ---- | --- |\n"
        "| qwen  | pp64 | 198.30 ± 2.1 |\n"
        "| qwen  | tg128| 39.90 ± 0.3 |\n"
    )
    s2 = summarize(parse_text(md))
    assert s2["prefill_tok_s"] == 198.3 and s2["decode_tok_s"] == 39.9


def test_schema_roundtrip(tmp_path):
    from schema import BenchRow, read_jsonl, write_jsonl

    row = BenchRow(variant="Q4_K_M", decode_tok_s=44.5, ondisk_mb=340.0)
    p = write_jsonl([row], tmp_path / "m.jsonl")
    back = read_jsonl(p)
    assert back[0]["variant"] == "Q4_K_M" and back[0]["decode_tok_s"] == 44.5
    assert back[0]["chrF"] is None  # unfilled stays explicit None, never 0


def test_pareto_frontier_mixed_directions():
    from pareto import pareto_frontier

    rows = [
        ("Q8_0", {"decode_tok_s": 30.1, "recon_bleu": 55.0, "ondisk_mb": 520.0}),
        ("Q4_K_M", {"decode_tok_s": 44.5, "recon_bleu": 54.2, "ondisk_mb": 340.0}),
        ("Q4_K_S", {"decode_tok_s": 47.0, "recon_bleu": 50.1, "ondisk_mb": 300.0}),
        # strictly dominated by Q4_K_M (slower, worse quality, bigger)
        ("DUD", {"decode_tok_s": 20.0, "recon_bleu": 40.0, "ondisk_mb": 600.0}),
    ]
    fr = set(pareto_frontier(rows, ("decode_tok_s", "recon_bleu", "ondisk_mb")))
    assert "DUD" not in fr
    assert {"Q8_0", "Q4_K_M", "Q4_K_S"} <= fr


def test_chatml_parity_with_training_template():
    """The phone prompt MUST equal tokenizer.apply_chat_template (parity = valid measurement)."""
    from transformers import AutoTokenizer

    from ko_dialect.data.template import SYSTEM_PROMPT, get_template

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    t = get_template("default")
    source, do = "어데 가노?", "gyeongsangdo"
    canon = t.build_prompt(tok, source, do, "dia2std")
    user = t.format_user(source, do, "dia2std")
    manual = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    assert manual == canon


# --------------------------------------------------------------------------- #
# VALIDITY GATE
# --------------------------------------------------------------------------- #
def test_decoding_is_deterministic_greedy():
    """The scoring path must be greedy + seeded; drift here invalidates quant-vs-fp16."""
    from decoding import GREEDY_SEED, hf_greedy_kwargs, llamacpp_greedy_body

    body = llamacpp_greedy_body("x", n_predict=32)
    assert body["temperature"] == 0.0
    assert body["top_k"] == 1
    assert body["top_p"] == 1.0 and body["min_p"] == 0.0
    assert body["seed"] == GREEDY_SEED
    assert body["return_tokens"] is True  # needed by the equivalence check

    hf = hf_greedy_kwargs()
    assert hf["do_sample"] is False and hf["num_beams"] == 1


def test_eval_set_hash_is_reproducible_and_content_sensitive():
    """Same selected pairs → same id_hash; any change → different hash (reproducibility)."""
    from make_eval_prompts import _region_hash

    samples = [
        {"source": "밥 뭇나?", "reference": "밥 먹었니?"},
        {"source": "어데 가노?", "reference": "어디 가니?"},
    ]
    h1 = _region_hash(samples)
    h2 = _region_hash([dict(s) for s in samples])  # identical content
    assert h1 == h2 and len(h1) == 64
    # order change → different hash
    assert _region_hash(list(reversed(samples))) != h1
    # content change → different hash
    perturbed = [{"source": "밥 뭇나?", "reference": "밥 먹었어?"}, samples[1]]
    assert _region_hash(perturbed) != h1


def test_strip_stop_trims_chatml_markers():
    from decoding import strip_stop

    assert strip_stop("밥 먹었니?<|im_end|>\n자취") == "밥 먹었니?"
    assert strip_stop("  깔끔  ") == "깔끔"


@pytest.mark.skipif(
    not (os.environ.get("ONDEVICE_DESKTOP_EP") and os.environ.get("ONDEVICE_PHONE_EP")),
    reason="set ONDEVICE_DESKTOP_EP and ONDEVICE_PHONE_EP (both serving the same gguf) to run",
)
def test_token_equivalence_desktop_vs_phone_live():
    """Live gate: desktop llama.cpp vs forwarded phone llama.cpp must emit identical ids."""
    from check_token_equivalence import check

    assert check(
        desktop=os.environ["ONDEVICE_DESKTOP_EP"],
        phone=os.environ["ONDEVICE_PHONE_EP"],
        k=int(os.environ.get("ONDEVICE_EQUIV_K", "5")),
    )
