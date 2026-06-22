from __future__ import annotations

import torch

from ko_dialect.models.classifier import (
    LABEL2ID,
    TextCNNConfig,
    TextCNNForSequenceClassification,
)


def test_label_mapping():
    assert LABEL2ID["standard"] == 0
    assert LABEL2ID["gangwondo"] == 1
    assert LABEL2ID["gyeongsangdo"] == 2


def test_textcnn_forward_shape(tiny_textcnn):
    bsz, seq_len = 4, 20
    input_ids = torch.randint(0, 100, (bsz, seq_len))
    out = tiny_textcnn(input_ids)
    assert out.logits.shape == (bsz, 3)
    assert out.loss is None


def test_textcnn_forward_with_labels(tiny_textcnn):
    bsz, seq_len = 4, 20
    input_ids = torch.randint(0, 100, (bsz, seq_len))
    labels = torch.tensor([0, 1, 2, 1])
    out = tiny_textcnn(input_ids, labels=labels)
    assert out.loss is not None
    assert out.loss.item() > 0


def test_textcnn_config_defaults():
    cfg = TextCNNConfig()
    assert cfg.num_labels == 3
    assert len(cfg.filter_sizes) == len(cfg.num_filters)
    # Default metadata must be 3-class and consistent with the head.
    assert len(cfg.id2label) == 3
    assert cfg.label2id == {"standard": 0, "gangwondo": 1, "gyeongsangdo": 2}


def test_textcnn_config_six_class_metadata_matches_head():
    # Regression: a 6-region (old_dialect) classifier must keep num_labels=6 and carry
    # 6-entry label metadata. If id2label defaulted to the 3-entry map, transformers
    # would derive num_labels=6 vs len(id2label)=3 and silently break the head.
    cfg = TextCNNConfig(num_labels=6)
    assert cfg.num_labels == 6
    assert len(cfg.id2label) == 6
    assert cfg.label2id["chungcheongdo"] == 5


def test_textcnn_six_class_forward_shape():
    cfg = TextCNNConfig(
        vocab_size=100, embed_dim=16, num_filters=[4, 4], filter_sizes=[2, 3], num_labels=6
    )
    model = TextCNNForSequenceClassification(cfg)
    out = model(torch.randint(0, 100, (4, 20)))
    assert out.logits.shape == (4, 6)


def test_textcnn_pad_token_zeroed(tiny_textcnn):
    pad_idx = tiny_textcnn.config.pad_token_id
    weight = tiny_textcnn.embed.weight.data
    assert weight[pad_idx].abs().sum().item() == 0.0


def test_textcnn_save_load(tiny_textcnn, tmp_path):
    tiny_textcnn.save_pretrained(str(tmp_path))
    loaded = TextCNNForSequenceClassification.from_pretrained(str(tmp_path))
    input_ids = torch.randint(0, 100, (2, 15))
    orig = tiny_textcnn(input_ids).logits
    relo = loaded(input_ids).logits
    assert torch.allclose(orig, relo, atol=1e-5)
