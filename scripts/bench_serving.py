#!/usr/bin/env python3
"""Serving benchmark for a chosen dialect-conversion model: latency + quality in one pass.

Translates a held-out slice one request at a time (batch=1 — the realistic single-user
serving latency, not the throughput-optimised batched path) and reports p50/p95/mean
latency, tokens/sec, plus chrF / copy_margin / exact-match against the reference so a
serving decision sees speed *and* quality together.

Pair with the leaderboard: rank runs with ``scripts/eval_leaderboard.py``, merge the
winner with ``scripts/merge_sft_lora.py``, then bench it here before shipping.

Usage:
    python scripts/bench_serving.py --model_path outputs/sft_merged --n 50
    python scripts/bench_serving.py --model_path outputs/grpo_arm1_merged \
        --target_do gangwondo --direction std2dia --n 50 --max_new_tokens 64
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import fire

from ko_dialect.data import ChatTemplate
from ko_dialect.evaluation import load_eval_samples, resolve_best_checkpoint
from ko_dialect.evaluation.serving import summarize_latencies

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bench")


def main(
    model_path: str = "outputs/sft_merged",
    raw_dataset_path: str = "outputs/dialect_raw_new",
    split: str = "valid",
    target_do: str = "gangwondo",
    direction: str = "std2dia",
    n: int = 50,
    max_new_tokens: int = 64,
    warmup: int = 3,
    out_dir: str = "outputs/eval_logs",
) -> None:
    """Benchmark single-request serving latency and translation quality."""
    import torch
    from sacrebleu.metrics import CHRF
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_best_checkpoint(model_path)
    rows = load_eval_samples(raw_dataset_path, split, target_do, direction, n)
    if not rows:
        raise SystemExit(f"No eval samples for {target_do}/{direction} in {split}.")

    tok = AutoTokenizer.from_pretrained(resolved)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    template = ChatTemplate()
    prompts = [template.build_prompt(tok, r["source"], target_do, direction) for r in rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        resolved, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    def gen_one(prompt: str) -> tuple[str, int]:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=448).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        new_ids = out[0, enc["input_ids"].shape[1] :]
        text = tok.decode(new_ids, skip_special_tokens=True).strip()
        return text, int(new_ids.shape[0])

    # Warmup (CUDA kernel autotune / allocator) — excluded from the measured stats.
    with torch.no_grad():
        for p in prompts[: max(0, warmup)]:
            gen_one(p)

        latencies: list[float] = []
        new_tokens: list[int] = []
        gens: list[str] = []
        for p in prompts:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            text, n_new = gen_one(p)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            new_tokens.append(n_new)
            gens.append(text)

    refs = [r["reference"] for r in rows]
    srcs = [r["source"] for r in rows]
    chrf = CHRF().corpus_score(gens, [refs]).score
    chrf_src = CHRF().corpus_score(gens, [srcs]).score
    exact = sum(int(g.strip() == r.strip()) for g, r in zip(gens, refs))

    lat = summarize_latencies(latencies, new_tokens)
    quality = {
        "chrf_vs_ref": round(chrf, 3),
        "copy_margin": round(chrf - chrf_src, 3),
        "exact_match": exact,
        "exact_match_pct": round(100 * exact / len(rows), 2),
    }

    print("\n" + "=" * 72)
    print(f"SERVING BENCH — {resolved}  ({target_do}/{direction}, n={len(rows)})")
    print("=" * 72)
    print(f"  device           : {device}")
    print(f"  latency p50/p95  : {lat['latency_ms_p50']} / {lat['latency_ms_p95']} ms")
    print(f"  latency mean/max : {lat['latency_ms_mean']} / {lat['latency_ms_max']} ms")
    print(
        f"  throughput       : {lat['requests_per_sec']} req/s, "
        f"{lat.get('tokens_per_sec', 0)} tok/s"
    )
    print(f"  chrF vs ref      : {quality['chrf_vs_ref']}")
    print(
        f"  copy_margin      : {quality['copy_margin']}  (>0 = genuine conversion, ~0 = copy-bias)"
    )
    print(
        f"  exact match      : {quality['exact_match']}/{len(rows)} "
        f"({quality['exact_match_pct']}%)  [low is normal for dialect]"
    )

    payload = {
        "schema": "ko_dialect.serving_bench/v1",
        "model": resolved,
        "target_do": target_do,
        "direction": direction,
        "n_samples": len(rows),
        "max_new_tokens": max_new_tokens,
        "device": device,
        "latency": lat,
        "quality": quality,
    }
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_root / f"serving_bench_{Path(resolved).name}_{ts}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
