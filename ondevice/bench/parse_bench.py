"""Parse `llama-bench` output → structured prefill/decode throughput.

`llama-bench` runs two kinds of micro-tests per model:
  * **pp** (prompt processing) — n_prompt>0, n_gen=0 → avg_ts = *prefill* tok/s
  * **tg** (text generation)   — n_prompt=0, n_gen>0 → avg_ts = *decode*  tok/s

Prefer JSON (`llama-bench -o json`): stable, already rep-averaged. Markdown table
parsing is a fallback for logs captured without `-o json`.

Pure functions (no I/O in parsing) so they unit-test offline without a phone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import fire


def parse_json(text: str) -> list[dict[str, Any]]:
    """Parse `llama-bench -o json` output (a JSON array of test objects)."""
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("results", data.get("tests", [data]))
    return list(data)


# markdown row: | model | size | params | backend | threads | test | t/s |
_MD_TS = re.compile(r"([0-9]+\.?[0-9]*)\s*±")
_MD_TEST = re.compile(r"\b(pp|tg)\s*(\d+)\b", re.IGNORECASE)


def parse_markdown(text: str) -> list[dict[str, Any]]:
    """Fallback: scrape the markdown table `llama-bench` prints by default."""
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "|" not in line or "t/s" in line or set(line.strip()) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        joined = " ".join(cells)
        m_test, m_ts = _MD_TEST.search(joined), _MD_TS.search(joined)
        if not (m_test and m_ts):
            continue
        kind, size = m_test.group(1).lower(), int(m_test.group(2))
        out.append(
            {
                "test_kind": kind,
                "n_prompt": size if kind == "pp" else 0,
                "n_gen": size if kind == "tg" else 0,
                "avg_ts": float(m_ts.group(1)),
            }
        )
    return out


def _kind(test: dict[str, Any]) -> str:
    if "test_kind" in test:
        return test["test_kind"]
    return "pp" if int(test.get("n_gen", 0) or 0) == 0 else "tg"


def summarize(tests: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse pp/tg test rows into one speed record (prefill + decode tok/s)."""
    prefill = decode = None
    n_prompt = n_gen = n_threads = None
    for t in tests:
        ts = t.get("avg_ts")
        if ts is None:
            continue
        if _kind(t) == "pp":
            prefill = float(ts)
            n_prompt = int(t.get("n_prompt") or n_prompt or 0) or n_prompt
        else:
            decode = float(ts)
            n_gen = int(t.get("n_gen") or n_gen or 0) or n_gen
        if t.get("n_threads") is not None:
            n_threads = int(t["n_threads"])
    return {
        "prefill_tok_s": prefill,
        "decode_tok_s": decode,
        "n_prompt_tokens": n_prompt,
        "n_gen_tokens": n_gen,
        "n_threads": n_threads,
    }


def parse_text(text: str) -> list[dict[str, Any]]:
    """Auto-detect JSON vs markdown."""
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            return parse_json(text)
        except json.JSONDecodeError:
            pass
    return parse_markdown(text)


def main(input: str, output: str | None = None) -> None:
    """CLI: parse a llama-bench log file → summarized speed JSON on stdout/file."""
    text = Path(input).read_text(encoding="utf-8")
    summary = summarize(parse_text(text))
    blob = json.dumps(summary, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(blob, encoding="utf-8")
    print(blob)


if __name__ == "__main__":
    fire.Fire(main)
