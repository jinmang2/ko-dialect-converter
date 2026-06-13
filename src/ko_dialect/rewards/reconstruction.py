from __future__ import annotations

import torch
from sacrebleu.metrics import BLEU

from ._utils import extract_texts

_bleu = BLEU(effective_order=True)


def _default_reverse_prompt(dialect_text: str) -> str:
    """Reverse (dialect → standard) prompt; mirror of the std2dia training prompt."""
    return f"다음 방언 문장을 표준어로 바꿔줘.\n방언: {dialect_text}\n표준어: "


def make_reconstruction_reward(
    model,
    tokenizer,
    *,
    max_new_tokens: int = 64,
    batch_size: int = 8,
    build_reverse_prompt=None,
    scale: float = 100.0,
):
    """Cycle-consistency / back-translation reward (Dual-RL, Luo et al. 2019).

    Reverse-generates each completion (a dialect sentence) back to standard Korean with a
    fixed back-translator, then scores ``BLEU(backtranslate(gen), source)``. The intuition
    is round-trip recoverability: a *content-faithful* style transfer can be translated
    back to (approximately) the input, so meaning survives the round trip; a hallucinated
    or over-corrected output has lost content and back-translates poorly. Crucially this is
    **proxy-independent** of the TextCNN classifier and the ``eojeol_map`` — it uses neither
    — so it cannot be gamed by the same reward-hacking route that inflates the style proxy
    (plan §3 A1/A4: proxy-independent selection signal; this is its reward-side counterpart).

    Mapping::

        reward = sentence_BLEU(backtranslate(gen), source) / scale    ∈ [0, 1]   (scale=100)

    ``source`` follows ``direction`` (``standard`` for std2dia, ``dialect`` for dia2std); the
    reverse prompt is the opposite-direction instruction (overridable via
    ``build_reverse_prompt``). Only std2dia rows use the dialect→standard reverse prompt;
    for dia2std the completion is already standard so the "reverse" is standard→dialect.

    COST / 6GB note (plan R3, phase-2, smoke-gated): this adds **one extra generation pass
    per completion every step**, roughly doubling GRPO rollout time and adding the
    back-translator's activations to VRAM. On the 6GB 2060 it must be smoke-checked first —
    reduce ``batch_size`` / ``max_new_tokens`` and confirm headroom before enabling in
    training. Reusing the frozen SFT base as the back-translator (same tokenizer/domain) is
    recommended; passing the live policy gives true Dual-RL cycle-consistency but couples the
    reward to the changing policy (noisier early). Greedy decoding keeps the signal
    deterministic within a step.

    Cite: Luo et al. 2019, "A Dual Reinforcement Learning Framework for Unsupervised Text
    Style Transfer" (back-translation / cycle-consistency content signal); plan
    .omc/plans/ralplan-grpo-rewards.md §3 (proxy-independent), §4 Option B, §11 Phase-2.
    """
    rev_prompt = build_reverse_prompt or _default_reverse_prompt

    @torch.no_grad()
    def _generate(prompt_list: list[str], device) -> list[str]:
        prev = tokenizer.padding_side
        tokenizer.padding_side = "left"
        out: list[str] = []
        try:
            for i in range(0, len(prompt_list), batch_size):
                chunk = prompt_list[i : i + batch_size]
                enc = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=448,
                ).to(device)
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                new = gen[:, enc["input_ids"].shape[1] :]
                out.extend(t.strip() for t in tokenizer.batch_decode(new, skip_special_tokens=True))
        finally:
            tokenizer.padding_side = prev
        return out

    @torch.no_grad()
    def r_reconstruction(
        prompts, completions, standard, dialect, direction=None, **kw
    ) -> list[float]:
        texts = extract_texts(completions)
        device = next(model.parameters()).device

        # Build the reverse prompt for each completion and the matching source target.
        rev_prompts: list[str] = []
        sources: list[str] = []
        for i, gen in enumerate(texts):
            dir_i = direction[i] if direction is not None else "std2dia"
            sources.append(standard[i] if dir_i == "std2dia" else dialect[i])
            rev_prompts.append(rev_prompt(gen) if gen.strip() else rev_prompt(""))

        back = _generate(rev_prompts, device)
        rewards = []
        for b, src in zip(back, sources):
            if not b.strip():
                rewards.append(0.0)
                continue
            score = _bleu.sentence_score(b, [src]).score  # ∈ [0, 100]
            rewards.append(min(1.0, max(0.0, score / scale)))
        return rewards

    r_reconstruction.__name__ = "r_reconstruction"
    return r_reconstruction
