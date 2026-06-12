"""CPU tests for the shared eval generation harness.

The GPU path (real model load) is exercised in `gpu`-marked tests elsewhere; here we cover
the pure filesystem resolution and the batching / left-padding / post-transform contract of
``generate_batched`` with lightweight fakes, so a regression in the canonical loop is caught
without a GPU.
"""

from __future__ import annotations

import json

import torch

from ko_dialect.evaluation.generation import generate_batched, resolve_eval_model


def test_resolve_eval_model_detects_full_model(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    resolved, is_adapter = resolve_eval_model(str(tmp_path))
    assert resolved == str(tmp_path)
    assert is_adapter is False


def test_resolve_eval_model_detects_adapter(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 8}))
    resolved, is_adapter = resolve_eval_model(str(tmp_path))
    assert resolved == str(tmp_path)
    assert is_adapter is True


class _FakeEnc(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    """Minimal tokenizer asserting the harness flips padding to 'left' during generation."""

    def __init__(self):
        self.padding_side = "right"
        self.pad_token_id = 0

    def __call__(self, chunk, return_tensors=None, padding=None, truncation=None, max_length=None):
        assert self.padding_side == "left", "decoder-only batched gen must left-pad"
        ids = torch.zeros((len(chunk), 3), dtype=torch.long)
        return _FakeEnc(input_ids=ids)

    def batch_decode(self, ids, skip_special_tokens=True):
        return [f"  out{i}  " for i in range(ids.shape[0])]


class _FakeModel:
    def generate(self, input_ids=None, max_new_tokens=2, **_):
        bsz, prompt_len = input_ids.shape
        return torch.zeros((bsz, prompt_len + max_new_tokens), dtype=torch.long)


def test_generate_batched_batches_and_strips():
    tok = _FakeTokenizer()
    prompts = [f"p{i}" for i in range(5)]
    out = generate_batched(_FakeModel(), tok, prompts, device="cpu", batch_size=2)
    assert len(out) == 5
    assert all(o.startswith("out") and o == o.strip() for o in out)  # stripped
    assert tok.padding_side == "right"  # restored after generation


def test_generate_batched_applies_post_transform():
    tok = _FakeTokenizer()
    out = generate_batched(
        _FakeModel(), tok, ["a", "b"], device="cpu", batch_size=8, post=str.upper
    )
    assert out == ["OUT0", "OUT1"]
