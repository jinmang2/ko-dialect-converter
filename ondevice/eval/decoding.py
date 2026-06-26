"""Canonical deterministic decoding — the single source shared by the phone capture
(gen_capture.py, llama.cpp /completion) and the desktop fp16 reference (desktop_ref.py,
HF transformers). VALIDITY GATE part 2/3.

Scoring only compares like-with-like if BOTH sides decode identically. So greedy is
pinned here once: temperature 0 + top_k 1 + fixed seed + same stop tokens. Any drift in
these knobs invalidates "quant cost vs fp16" — so the test suite asserts against these
constants rather than letting each script hardcode its own.

(demo/index.html is intentionally NOT bound to this — it is a feel/recording surface,
not a scoring surface. Only the scoring path must be deterministic.)
"""

from __future__ import annotations

GREEDY_SEED = 0
STOP = ["<|im_end|>", "<|im_start|>"]
MAX_NEW_TOKENS = 64  # matches configs/eval/default.yaml max_new_tokens


def llamacpp_greedy_body(prompt: str, n_predict: int = MAX_NEW_TOKENS) -> dict:
    """Request body for llama-server /completion forcing greedy + reproducibility.

    temperature 0 already short-circuits to argmax, but top_k/top_p/min_p are pinned too
    so the result is identical across llama.cpp builds that change default sampler chains.
    """
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": False,
        "cache_prompt": True,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": GREEDY_SEED,
        "n_probs": 0,
        "stop": STOP,
        "return_tokens": True,  # so the equivalence check can compare token ids
    }


def hf_greedy_kwargs(max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """transformers `model.generate` kwargs for the desktop fp16 reference (greedy)."""
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }


def strip_stop(text: str) -> str:
    """Trim anything from the first stop marker on (server usually does this already)."""
    cut = len(text)
    for s in STOP:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()
