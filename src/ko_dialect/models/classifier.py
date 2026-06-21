from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from ko_dialect.data.labels import DIALECT_LABELS

# Region label map comes from the single source of truth (ko_dialect.data.labels):
# standard=0, gangwon=1, gyeongsang=2, jeolla=3, jeju=4, chungcheong=5. The *default*
# config stays 3-class; a 6-class classifier slices these to num_labels in __init__.
LABEL2ID: dict[str, int] = dict(DIALECT_LABELS)
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}


class TextCNNConfig(PretrainedConfig):
    model_type = "textcnn"
    # Trainer's eval loop accumulates every non-loss output tensor across the
    # WHOLE eval set. Without this, hidden_states (B, 512) for all 150k+ valid
    # rows piles onto the GPU (multi-GiB spike + O(n^2) concat slowdown) and
    # also breaks compute_metrics by making predictions a (logits, hidden) tuple.
    keys_to_ignore_at_inference = ["hidden_states"]

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 256,
        filter_sizes: list[int] | None = None,
        num_filters: list[int] | None = None,
        dropout: float = 0.3,
        num_labels: int = 3,
        id2label: dict | None = None,
        label2id: dict | None = None,
        pad_token_id: int = 0,
        class_weights: list[float] | None = None,
        **kwargs,
    ):
        # Default the label metadata to the first ``num_labels`` regions so it always
        # matches the head size. Passing the full 6-entry map when num_labels=3 would
        # let transformers derive num_labels=6 from len(id2label) and silently resize
        # the classifier head (and vice-versa: a 6-class head with 3-entry metadata).
        if label2id is None:
            label2id = {k: v for k, v in LABEL2ID.items() if v < num_labels}
        if id2label is None:
            id2label = {v: k for k, v in label2id.items()}
        super().__init__(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.filter_sizes = filter_sizes or [2, 3, 4, 5]
        self.num_filters = num_filters or [128, 128, 128, 128]
        self.dropout = dropout
        # Per-class CrossEntropyLoss weights (len == num_labels) to counter the
        # standard-vs-dialect imbalance. ``None`` => unweighted loss. Serialized
        # with the config so a saved classifier remembers how it was trained.
        self.class_weights = class_weights


@dataclass
class TextCNNOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None


class TextCNNForSequenceClassification(PreTrainedModel):
    """TextCNN style classifier (Kim 2014) adapted for 3-class dialect detection."""

    config_class = TextCNNConfig
    base_model_prefix = "textcnn"

    def __init__(self, config: TextCNNConfig):
        super().__init__(config)
        self.embed = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, n_filter, (f_size, config.embed_dim))
                for n_filter, f_size in zip(config.num_filters, config.filter_sizes)
            ]
        )
        feature_dim = sum(config.num_filters)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, config.num_labels),
        )
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> TextCNNOutput:
        # input_ids: (B, L)
        x = self.embed(input_ids).unsqueeze(1)  # (B, 1, L, E)

        pools: list[torch.Tensor] = []
        for conv in self.convs:
            c = torch.relu(conv(x)).squeeze(3)  # (B, n_filter, L')
            p = torch.max_pool1d(c, c.size(2)).squeeze(2)  # (B, n_filter)
            pools.append(p)

        hidden = torch.cat(pools, dim=1)  # (B, feature_dim)
        logits = self.classifier(hidden)  # (B, num_labels)

        loss: torch.Tensor | None = None
        if labels is not None:
            weight = None
            if self.config.class_weights is not None:
                weight = torch.as_tensor(
                    self.config.class_weights, dtype=logits.dtype, device=logits.device
                )
            loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)

        return TextCNNOutput(loss=loss, logits=logits, hidden_states=hidden)
