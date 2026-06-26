"""Drive the phone's llama-server over a forwarded port → capture outputs (desktop).

`adb forward tcp:8080 tcp:8080` makes the phone server reachable at 127.0.0.1:8080,
so the desktop can feed the fixed prompts and capture generations WITHOUT a python/jq
toolchain on the phone (decision O3: phone = generate, desktop = orchestrate + score).
Generation still runs on-device; only the driving + later scoring is desktop-side.

Decoding is the shared deterministic greedy (decoding.llamacpp_greedy_body) — identical
to the desktop fp16 reference (desktop_ref.py), so chrF/recon_bleu deltas are pure
quantization cost (VALIDITY GATE). Token ids are captured too, for the equivalence check.

  in : ondevice/data/eval_prompts.jsonl   (from make_eval_prompts.py)
  out: ondevice/eval/outputs/<variant>.jsonl  {idx, do, output, tokens}
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import fire
from decoding import MAX_NEW_TOKENS, llamacpp_greedy_body, strip_stop


def _completion(
    endpoint: str, prompt: str, n_predict: int, timeout: float
) -> tuple[str, list[int]]:
    """Non-streaming greedy /completion. Returns (text, token_ids)."""
    body = json.dumps(llamacpp_greedy_body(prompt, n_predict)).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    text = strip_stop(obj.get("content") or "")
    # llama-server returns generated token ids under "tokens" when return_tokens=True
    tokens = obj.get("tokens") or []
    if tokens and isinstance(tokens[0], dict):  # some builds wrap them
        tokens = [t.get("id") for t in tokens]
    return text, [t for t in tokens if isinstance(t, int)]


def capture(
    variant: str,
    endpoint: str = "http://127.0.0.1:8080",
    prompts: str = "ondevice/data/eval_prompts.jsonl",
    out_dir: str = "ondevice/eval/outputs",
    n_predict: int = MAX_NEW_TOKENS,
    timeout: float = 120.0,
    limit: int | None = None,
) -> None:
    """Generate on the phone server for every prompt; write {idx, do, output, tokens} JSONL."""
    rows = [
        json.loads(line)
        for line in Path(prompts).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit:
        rows = rows[:limit]

    out_path = Path(out_dir) / f"{variant}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            output, tokens = _completion(endpoint, r["prompt"], n_predict, timeout)
            f.write(
                json.dumps(
                    {"idx": r["idx"], "do": r["do"], "output": output, "tokens": tokens},
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"  [{variant}] {i + 1}/{len(rows)}")
    print(f"wrote {len(rows)} generations → {out_path}")


if __name__ == "__main__":
    fire.Fire(capture)
