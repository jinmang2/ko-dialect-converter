from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput

# 3-class labels: standard / gangwondo / gyeongsangdo
LABEL2ID: dict[str, int] = {"standard": 0, "gangwondo": 1, "gyeongsangdo": 2}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}


class TextCNNConfig(PretrainedConfig):
    model_type = "textcnn"

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
        **kwargs,
    ):
        super().__init__(
            num_labels=num_labels,
            id2label=id2label or ID2LABEL,
            label2id=label2id or LABEL2ID,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.filter_sizes = filter_sizes or [2, 3, 4, 5]
        self.num_filters = num_filters or [128, 128, 128, 128]
        self.dropout = dropout


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
            loss = nn.CrossEntropyLoss()(logits, labels)

        return TextCNNOutput(loss=loss, logits=logits, hidden_states=hidden)
