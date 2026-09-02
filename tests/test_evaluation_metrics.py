from __future__ import annotations

import pytest
import torch

from ko_dialect.evaluation.metrics import (
    compute_bleu,
    compute_chrf,
    compute_dfs,
    compute_eojeol_accuracy,
    compute_tdr,
    evaluate_all,
)

# ---------------------------------------------------------------------------
# compute_eojeol_accuracy
# ---------------------------------------------------------------------------


def test_eojeol_accuracy_perfect():
    outputs = ["나는 집에 갔당"]
    maps = [[{"dialect": "갔당", "standard": "갔다"}]]
    assert compute_eojeol_accuracy(outputs, maps) == 1.0


def test_eojeol_accuracy_miss():
    outputs = ["나는 집에 갔다"]
    maps = [[{"dialect": "갔당", "standard": "갔다"}]]
    assert compute_eojeol_accuracy(outputs, maps) == 0.0


def test_eojeol_accuracy_partial():
    outputs = ["나는 집에 갔당 먹었지"]
    maps = [
        [
            {"dialect": "갔당", "standard": "갔다"},
            {"dialect": "먹겠나", "standard": "먹겠어"},
        ]
    ]
    assert compute_eojeol_accuracy(outputs, maps) == pytest.approx(0.5)


def test_eojeol_accuracy_empty_map():
    assert compute_eojeol_accuracy(["hello"], [[]]) == 0.0


def test_eojeol_accuracy_empty_outputs():
    assert compute_eojeol_accuracy([], []) == 0.0


# ---------------------------------------------------------------------------
# compute_tdr  (uses tiny_textcnn fixture from conftest)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    pad_token = "<pad>"

    def __call__(self, texts, **kwargs):
        return {"input_ids": torch.zeros(len(texts), 10, dtype=torch.long)}


def test_tdr_returns_float(tiny_textcnn):
    outputs = ["안녕", "반가워"]
    tdr = compute_tdr(outputs, "gangwondo", tiny_textcnn, _FakeTokenizer())
    assert 0.0 <= tdr <= 1.0


def test_tdr_empty_returns_zero(tiny_textcnn):
    assert compute_tdr([], "gangwondo", tiny_textcnn, _FakeTokenizer()) == 0.0


# ---------------------------------------------------------------------------
# compute_dfs  (no real embedder needed — synthetic tensors via lambda)
# ---------------------------------------------------------------------------


def _make_embed_fn(vecs: dict[str, torch.Tensor]):
    """Returns an embed_fn that looks up pre-computed tensors by text."""

    def embed_fn(texts):
        return torch.stack([vecs[t] for t in texts])

    return embed_fn


def test_dfs_positive_when_closer_to_dialect():
    # out is identical to dialect_ref → cos(out,dia)=1, cos(out,std)=0 → DFS > 0
    v = torch.tensor([1.0, 0.0])
    std = torch.tensor([0.0, 1.0])
    embed = _make_embed_fn({"out": v, "dia": v, "std": std})
    dfs = compute_dfs(["out"], ["dia"], ["std"], embed)
    assert dfs > 0.0


def test_dfs_negative_when_closer_to_standard():
    # out is identical to std → cos(out,dia)=0, cos(out,std)=1 → DFS < 0
    v = torch.tensor([1.0, 0.0])
    dia = torch.tensor([0.0, 1.0])
    embed = _make_embed_fn({"out": v, "dia": dia, "std": v})
    dfs = compute_dfs(["out"], ["dia"], ["std"], embed)
    assert dfs < 0.0


def test_dfs_empty_returns_zero():
    embed = _make_embed_fn({})
    assert compute_dfs([], [], [], embed) == 0.0


# ---------------------------------------------------------------------------
# compute_bleu / compute_chrf
# ---------------------------------------------------------------------------


def test_compute_bleu_perfect():
    sent = "나는 집에 갔당"
    score = compute_bleu([sent], [sent])
    assert score == pytest.approx(100.0)


def test_compute_bleu_empty():
    assert compute_bleu([], []) == 0.0


def test_compute_chrf_perfect():
    sent = "나는 집에 갔당"
    score = compute_chrf([sent], [sent])
    assert score == pytest.approx(100.0)


def test_compute_chrf_empty():
    assert compute_chrf([], []) == 0.0


# ---------------------------------------------------------------------------
# evaluate_all  — bleu/chrf keys always present
# ---------------------------------------------------------------------------


def test_evaluate_all_has_bleu_chrf(tiny_textcnn):
    outputs = ["안녕"]
    refs = ["안녕"]
    stds = ["hello"]
    maps = [[]]
    result = evaluate_all(
        outputs=outputs,
        dialect_refs=refs,
        standard_refs=stds,
        dialect_eojeol_maps=maps,
        target_do="gangwondo",
        classifier=tiny_textcnn,
        cls_tokenizer=_FakeTokenizer(),
        embed_fn=None,
    )
    assert "bleu" in result
    assert "chrf" in result
    assert "tdr" in result
    assert "eojeol_accuracy" in result
    assert "dfs" not in result  # embed_fn=None → dfs skipped


# ---------------------------------------------------------------------------
# Dual-prompt reconstruction: the plain-prompt score is kept for continuity, the
# training-format score is the ranking basis, and the gap between them is reported.
# See docs/DATA_ANALYSIS.md §4.2.
# ---------------------------------------------------------------------------


def _evaluate(tiny_textcnn, **kwargs):
    return evaluate_all(
        outputs=["밥 먹었나"],
        dialect_refs=["밥 먹었나"],
        standard_refs=["밥 먹었니"],
        dialect_eojeol_maps=[[]],
        target_do="gangwondo",
        classifier=tiny_textcnn,
        cls_tokenizer=_FakeTokenizer(),
        embed_fn=None,
        **kwargs,
    )


def test_recon_keys_absent_when_no_reverse_pass(tiny_textcnn):
    result = _evaluate(tiny_textcnn)
    assert "reconstruction_bleu" not in result
    assert "reconstruction_bleu_chatml" not in result
    assert "format_sensitivity" not in result


def test_plain_only_reverse_pass_stays_backward_compatible(tiny_textcnn):
    """A caller that has not adopted the chatml pass must keep working unchanged."""
    result = _evaluate(tiny_textcnn, reverse_outputs=["밥 먹었니"])
    assert result["reconstruction_bleu"] > 0
    assert "reconstruction_bleu_chatml" not in result
    assert "format_sensitivity" not in result


def test_format_sensitivity_is_the_gain_from_the_training_format(tiny_textcnn):
    # chatml reconstruction is exact, plain is not → positive sensitivity.
    result = _evaluate(
        tiny_textcnn,
        reverse_outputs=["전혀 다른 문장"],
        reverse_outputs_chatml=["밥 먹었니"],
    )
    assert result["reconstruction_bleu_chatml"] > result["reconstruction_bleu"]
    assert result["format_sensitivity"] == pytest.approx(
        result["reconstruction_bleu_chatml"] - result["reconstruction_bleu"]
    )
    assert result["format_sensitivity"] > 0


def test_format_sensitivity_is_zero_when_prompt_format_does_not_matter(tiny_textcnn):
    same = ["밥 먹었니"]
    result = _evaluate(tiny_textcnn, reverse_outputs=same, reverse_outputs_chatml=list(same))
    assert result["format_sensitivity"] == pytest.approx(0.0)
