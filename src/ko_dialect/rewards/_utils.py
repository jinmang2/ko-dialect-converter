from __future__ import annotations

from collections.abc import Callable

# Punctuation/symbols that commonly cling to a Korean eojeol boundary. We strip
# these from both ends before matching so that "갔어," / "갔어." / "갔어\"" all
# resolve to the same gold eojeol "갔어". Internal characters are left untouched.
_BOUNDARY_PUNCT = (
    ".,!?;:\"'()[]{}<>"  # ASCII
    "．，！？；：…·"  # CJK
    "「」『』《》〈〉“”‘’"  # CJK quotes/brackets
    "~`@#$%^&*-_=+/\\|"  # misc symbols
)


def extract_texts(completions: list) -> list[str]:
    """Extract plain text from TRL completions (string or conversational format)."""
    if not completions:
        return []
    first = completions[0]
    if isinstance(first, str):
        return list(completions)
    out = []
    for c in completions:
        if isinstance(c, list):
            # conversational: [[{"role": "assistant", "content": "..."}], ...]
            out.append(c[0]["content"] if c else "")
        elif isinstance(c, dict):
            out.append(c.get("content", ""))
        else:
            out.append(str(c))
    return out


def normalize_eojeol(word: str) -> str:
    """Strip surrounding whitespace and boundary punctuation from an eojeol.

    This makes eojeol matching robust to trailing/leading punctuation that the model
    emits (e.g. "갔어." or "\"갔어\""), which otherwise breaks naive whitespace splits
    and lets a present-but-punctuated word be miscounted as removed (reward hacking).
    """
    return word.strip().strip(_BOUNDARY_PUNCT).strip()


def tokenize_words(text: str) -> set[str]:
    """Whitespace-split ``text`` into a set of punctuation-normalized eojeols."""
    return {w for w in (normalize_eojeol(tok) for tok in text.split()) if w}


def rescale_to_unit(fn: Callable, lo: float, hi: float) -> Callable:
    """Wrap a reward fn so its outputs are affine-mapped from ``[lo, hi]`` to ``[0, 1]``.

    Different rewards live on different native scales (e.g. r_style ∈ [-1, 1] vs
    r_content ∈ [0, 1]). Because GRPO sums the *weighted* rewards before group
    normalization, a wider-range reward silently dominates the mix. Rescaling each
    reward to a common [0, 1] basis makes the configured weights the only knob that
    governs the mixture. Values are clamped to [0, 1] to tolerate out-of-range outputs.
    """
    span = hi - lo
    if span <= 0:
        return fn

    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        return [min(1.0, max(0.0, (r - lo) / span)) for r in out]

    wrapped.__name__ = getattr(fn, "__name__", "reward")
    return wrapped
