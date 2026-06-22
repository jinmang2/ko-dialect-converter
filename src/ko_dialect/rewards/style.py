from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from ko_dialect.data.labels import DO_TO_LABEL, LABEL_STANDARD

from ._utils import extract_texts


def make_style_reward(classifier, tokenizer, max_length: int = 128):
    """Factory: returns a TRL-compatible r_style bound to (classifier, tokenizer).

    Reward = p(target_do) - p(standard)  — from Thank-you-BART cal_sc_loss.
    Single batched inference per call (replaces per-sample loop).
    """

    @torch.no_grad()
    def r_style(prompts, completions, do, direction=None, **kw):
        texts = extract_texts(completions)
        device = next(classifier.parameters()).device
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        probs = F.softmax(classifier(**enc).logits, dim=-1)  # (B, num_classes)

        rewards = []
        for i, d in enumerate(do):
            target_label = DO_TO_LABEL[str(d)]
            r = float(probs[i, target_label] - probs[i, LABEL_STANDARD])
            # flip for dia→std direction
            if direction is not None and direction[i] == "dia2std":
                r = -r
            rewards.append(r)
        return rewards

    r_style.__name__ = "r_style"
    return r_style
