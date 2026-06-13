from __future__ import annotations

import pytest
import torch

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")


@pytest.fixture
def tiny_textcnn():
    """A minimal TextCNN (3-class) for CPU tests."""
    from ko_dialect.models.classifier import TextCNNConfig, TextCNNForSequenceClassification

    cfg = TextCNNConfig(
        vocab_size=100,
        embed_dim=16,
        filter_sizes=[2, 3],
        num_filters=[4, 4],
        dropout=0.0,
        num_labels=3,
    )
    return TextCNNForSequenceClassification(cfg)


@pytest.fixture
def mock_tokenizer():
    """A minimal tokenizer-like object for template tests (no HF download)."""
    from unittest.mock import MagicMock

    tok = MagicMock()
    tok.pad_token = "<pad>"
    tok.eos_token = "</s>"
    tok.eos_token_id = 2

    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        parts = [f"[{m['role']}] {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("[assistant]")
        return "\n".join(parts)

    tok.apply_chat_template.side_effect = apply_chat_template
    return tok
