"""VALIDITY GATE 3/3 — prove phone scoring == desktop scoring at the token level.

ChatML byte-parity (input) is necessary but not sufficient: the phone could still
tokenize or decode differently. So for a handful of prompts we hit BOTH a desktop
llama.cpp server and the (adb-forwarded) phone llama.cpp server with identical greedy
requests and assert:

  1. /tokenize(prompt) is identical  → same input token ids (tokenizer parity)
  2. /completion token ids identical → same output under greedy (decode parity)

Same model (same GGUF), same engine, greedy → the ONLY difference is the hardware. If
ids match, the phone's chrF/recon_bleu are the desktop's. If they diverge, on-device
numbers are NOT comparable and the gate fails.

Usage (both servers up; serve the SAME gguf on each):
  # desktop:  ondevice/serve/launch_server.sh -m kodialect-f16.gguf --port 8081
  # phone:    ondevice/serve/launch_server.sh -m kodialect-f16.gguf   (+ adb forward tcp:8080)
  python ondevice/eval/check_token_equivalence.py \
      --desktop http://127.0.0.1:8081 --phone http://127.0.0.1:8080 --k 5
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import fire
from decoding import llamacpp_greedy_body


def _post(endpoint: str, path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _tokenize(endpoint: str, prompt: str, timeout: float) -> list[int]:
    obj = _post(endpoint, "/tokenize", {"content": prompt}, timeout)
    toks = obj.get("tokens") or []
    return [t["id"] if isinstance(t, dict) else t for t in toks]


def _completion_tokens(endpoint: str, prompt: str, n_predict: int, timeout: float) -> list[int]:
    obj = _post(endpoint, "/completion", llamacpp_greedy_body(prompt, n_predict), timeout)
    toks = obj.get("tokens") or []
    return [t["id"] if isinstance(t, dict) else t for t in toks if (isinstance(t, (int, dict)))]


def check(
    desktop: str,
    phone: str,
    prompts: str = "ondevice/data/eval_prompts.jsonl",
    k: int = 5,
    n_predict: int = 32,
    timeout: float = 120.0,
) -> bool:
    """Compare desktop vs phone token ids over the first ``k`` prompts. Exit 1 on mismatch."""
    rows = [
        json.loads(line)
        for line in Path(prompts).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:k]

    all_ok = True
    for i, r in enumerate(rows):
        p = r["prompt"]
        d_in, p_in = _tokenize(desktop, p, timeout), _tokenize(phone, p, timeout)
        d_out, p_out = (
            _completion_tokens(desktop, p, n_predict, timeout),
            _completion_tokens(phone, p, n_predict, timeout),
        )
        in_ok, out_ok = d_in == p_in, d_out == p_out
        all_ok = all_ok and in_ok and out_ok
        status = "OK " if (in_ok and out_ok) else "FAIL"
        print(
            f"[{status}] prompt {i} ({r['do']})  input_ids={'=' if in_ok else '≠'}  "
            f"output_ids={'=' if out_ok else '≠'}  ({len(d_out)} gen toks)"
        )
        if not in_ok:
            print(f"    desktop in : {d_in[:24]}")
            print(f"    phone   in : {p_in[:24]}")
        if not out_ok:
            print(f"    desktop out: {d_out[:24]}")
            print(f"    phone   out: {p_out[:24]}")

    print(
        f"\n{'PASS — phone == desktop (scoring is valid)' if all_ok else 'FAIL — token divergence; on-device numbers NOT comparable'}"
    )
    if not all_ok:
        sys.exit(1)
    return all_ok


if __name__ == "__main__":
    fire.Fire(check)
