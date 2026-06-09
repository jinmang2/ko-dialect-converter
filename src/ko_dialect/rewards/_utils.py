from __future__ import annotations


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
