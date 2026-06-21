from __future__ import annotations

import torch

from ._utils import extract_texts


def make_fluency_reward(model, tokenizer, max_length: int = 128, scale: float = 4.0):
    """Factory: a reference-LM perplexity reward, bound to a frozen (model, tokenizer).

    Degenerate GRPO outputs (token repetition, gibberish, truncated fragments) can
    still satisfy style/edit signals while being unreadable. A frozen language model's
    per-token negative log-likelihood (NLL) of the *generation* is a cheap fluency
    axis: fluent text is high-probability, gibberish is not. One forward pass per
    sample, no gold needed (plan Option D, §4-D).

    NLL → (0, 1] mapping::

        r = exp(-nll / scale)

    where ``nll`` is the mean per-token cross-entropy of the completion. ``scale``
    is the calibration knob: ``r = exp(-nll/scale)``, so at ``nll = scale`` the
    reward is ``1/e ≈ 0.37``. **Calibration / dialect guard:** a reference LM trained
    mostly on standard Korean assigns *legitimately* higher NLL to valid dialect, so a
    sharp mapping would punish the very thing we want. We therefore (a) keep ``scale``
    loose (default 4.0 nats — only clearly degenerate, very-high-NLL text is pushed
    toward 0) and (b) register this reward at a LOW default weight (0.2) so it
    stabilises without dominating. Tighten ``scale`` only if degenerate outputs
    persist; raising it further softens the penalty.

    Reusing the GRPO base model as this reference LM is cheap and recommended (same
    tokenizer/domain); any frozen causal LM + tokenizer handle works.

    Cite: the fluency axis of text-style-transfer evaluation, e.g. Pauli, Augenstein
    & Assent 2025, "Mind the Style Gap" (arXiv:2502.15022); LM-perplexity fluency is the
    standard naturalness proxy in TST (plan §4-D, Option D).
    """
    if scale <= 0:
        raise ValueError(f"fluency reward needs scale > 0, got {scale}.")

    @torch.no_grad()
    def r_fluency(prompts, completions, **kw):
        texts = extract_texts(completions)
        device = next(model.parameters()).device
        rewards = []
        for text in texts:
            if not text.strip():
                # Empty/whitespace output: no fluent content to credit.
                rewards.append(0.0)
                continue
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                # A single token has no next-token target to score: stay neutral
                # rather than emit a spurious 0/1.
                rewards.append(0.5)
                continue
            out = model(**enc, labels=input_ids)
            nll = float(out.loss)  # mean per-token cross-entropy (nats)
            rewards.append(float(torch.exp(torch.tensor(-nll / scale))))
        return rewards

    r_fluency.__name__ = "r_fluency"
    return r_fluency
