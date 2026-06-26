"""Desktop fp16 reference generation on the SAME fixed eval set (VALIDITY GATE 1/3).

Runs the canonical HF-transformers fp16 `sft_merged` (the pre-GGUF desktop model) greedily
over the exact `eval_prompts.jsonl` prompts, so `score_offline.py --variant desktop_fp16`
gives the fp16 baseline column. Then "quantized chrF − fp16 chrF" is a same-set, same-
decoding delta — the honest "quantization cost" number.

Greedy is `generate_batched(..., do_sample=False)`; decoding intent is shared with the
phone path via decoding.MAX_NEW_TOKENS. For an *engine-isolated* baseline instead (only
quantization differs, not the inference engine), serve `kodialect-f16.gguf` on a desktop
llama-server and run `gen_capture.py --variant f16` — both are valid, they answer
slightly different questions (conversion+quant cost vs quant-only cost).

  in : ondevice/data/eval_prompts.jsonl
  out: ondevice/eval/outputs/desktop_fp16.jsonl  {idx, do, output}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from decoding import MAX_NEW_TOKENS, strip_stop

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from ko_dialect.evaluation.generation import generate_batched, load_generation_model  # noqa: E402


def generate(
    model_path: str = "outputs/sft_merged",
    base_model: str | None = None,
    prompts: str = "ondevice/data/eval_prompts.jsonl",
    variant: str = "desktop_fp16",
    out_dir: str = "ondevice/eval/outputs",
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = 16,
) -> None:
    """Greedy fp16 generation over the fixed eval set → {idx, do, output} JSONL."""
    rows = [
        json.loads(line)
        for line in Path(prompts).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model, tokenizer = load_generation_model(model_path, base_model)

    outs = generate_batched(
        model,
        tokenizer,
        [r["prompt"] for r in rows],
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        post=strip_stop,
    )

    out_path = Path(out_dir) / f"{variant}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            f.write(
                json.dumps({"idx": r["idx"], "do": r["do"], "output": o}, ensure_ascii=False) + "\n"
            )
    print(f"wrote {len(rows)} fp16 generations → {out_path}")
    print(f"next: python ondevice/eval/score_offline.py --variant {variant}")


if __name__ == "__main__":
    fire.Fire(generate)
