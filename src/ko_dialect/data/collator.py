from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class ClassifierCollator:
    """Pad-to-max_length collator for TextCNN (integer input_ids, no attention_mask)."""

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 128
    text_column: str = "text"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [f[self.text_column] for f in features]
        labels = [f["label"] for f in features]

        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
