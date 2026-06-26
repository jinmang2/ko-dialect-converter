"""Drive the phone's llama-server over a forwarded port → capture outputs (desktop).

`adb forward tcp:8080 tcp:8080` makes the phone server reachable at 127.0.0.1:8080,
so the desktop can feed the fixed prompts and capture generations WITHOUT a python/jq
toolchain on the phone (decision O3: phone = generate, desktop = orchestrate + score).
Generation still runs on-device; only the driving + later scoring is desktop-side.

Greedy + seed=0 + same stop tokens as the demo → reproducible, comparable to the
desktop fp16 baseline (brief §11 quality-equivalence check).

  in : ondevice/data/eval_prompts.jsonl   (from make_eval_prompts.py)
  out: ondevice/eval/outputs/<variant>.jsonl  {idx, do, output}
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import fire

STOP = ["<|im_end|>", "<|im_start|>"]


def _completion(endpoint: str, prompt: str, n_predict: int, timeout: float) -> str:
    """Non-streaming /completion call (greedy). Returns generated text."""
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "stream": False,
            "cache_prompt": True,
            "temperature": 0,
            "seed": 0,
            "stop": STOP,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return (obj.get("content") or "").strip()


def capture(
    variant: str,
    endpoint: str = "http://127.0.0.1:8080",
    prompts: str = "ondevice/data/eval_prompts.jsonl",
    out_dir: str = "ondevice/eval/outputs",
    n_predict: int = 64,
    timeout: float = 120.0,
    limit: int | None = None,
) -> None:
    """Generate on the phone server for every prompt; write {idx, do, output} JSONL."""
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
            output = _completion(endpoint, r["prompt"], n_predict, timeout)
            f.write(
                json.dumps({"idx": r["idx"], "do": r["do"], "output": output}, ensure_ascii=False)
                + "\n"
            )
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"  [{variant}] {i + 1}/{len(rows)}")
    print(f"wrote {len(rows)} generations → {out_path}")


if __name__ == "__main__":
    fire.Fire(capture)
