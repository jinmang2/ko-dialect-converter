#!/usr/bin/env python3
"""Serve SFT + every GRPO arm together and compare their outputs side by side.

The leaderboard showed no single winner — SFT (fidelity) and arm2 (conversion) are both
Pareto-optimal. So instead of committing to one model, this hosts them *all at once* and
shows each model's translation for the same input. You decide per example, or just keep
the system up and inspect.

VRAM design (6 GB RTX 2060): the GRPO adapters were all trained on the SAME base
(`outputs/sft_merged`). We load that base **once** and attach each run as a named PEFT
adapter (LoRA deltas are only a few MB each), then hot-swap with ``set_adapter`` per
request. "SFT" is the base with adapters disabled. One backbone in memory, N models served.

Usage:
    # ad-hoc single sentence across all discovered runs:
    python scripts/serve_compare.py --text "밥 먹었니?" --target_do gangwondo
    # a batch from the held-out set, only specific runs, with references:
    python scripts/serve_compare.py --runs "grpo_arm1 grpo_arm2" \
        --raw_dataset_path outputs/dialect_raw_new --n 10 --target_do gangwondo
"""

from __future__ import annotations

import logging
from pathlib import Path

import fire

from ko_dialect.data import ChatTemplate
from ko_dialect.evaluation import TOKENIZER_MAX_LENGTH, load_eval_samples
from ko_dialect.evaluation.leaderboard import RunSpec, discover_runs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serve_compare")

BASE = "outputs/sft_merged"
SFT_TAG = "SFT"


def _resolve_specs(runs: str | None, base: str) -> list[RunSpec]:
    """SFT baseline + the requested (or all discovered) GRPO adapter runs."""
    if not runs:
        return discover_runs("outputs", base, include_sft=True, sft_tag=SFT_TAG)
    specs = [RunSpec(SFT_TAG, base, None)]
    for run in (r.strip() for r in runs.replace(",", " ").split() if r.strip()):
        adapter = run if "/" in run else f"outputs/{run}"
        specs.append(RunSpec(Path(adapter).name, base, adapter))
    return specs


class MultiAdapterTranslator:
    """One base in VRAM, N named adapters hot-swapped per request."""

    def __init__(self, specs: list[RunSpec], base: str, max_new_tokens: int = 64):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.specs = specs
        self.tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.template = ChatTemplate()

        # base (outputs/sft_merged) is always a local dir by design; match the tokenizer's
        # local_files_only so a missing/partial dir fails offline instead of silently
        # triggering a Hub fetch for the model while the tokenizer hard-fails.
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float16, device_map="auto", local_files_only=True
        )
        self.peft_model = None
        self.adapter_tags: set[str] = set()
        for spec in specs:
            if spec.adapter is None:
                continue
            if self.peft_model is None:
                from peft import PeftModel

                self.peft_model = PeftModel.from_pretrained(
                    model, spec.adapter, adapter_name=spec.tag
                )
            else:
                self.peft_model.load_adapter(spec.adapter, adapter_name=spec.tag)
            self.adapter_tags.add(spec.tag)
            logger.info("Attached adapter %s", spec.tag)
        self.model = (self.peft_model or model).eval()
        self.device = next(self.model.parameters()).device

    def _generate(self, prompts: list[str]) -> list[str]:
        torch = self.torch
        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TOKENIZER_MAX_LENGTH,
        ).to(self.device)
        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1] :]
        return [t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True)]

    def translate(self, sources: list[str], target_do: str, direction: str) -> dict[str, list[str]]:
        """Return ``{run_tag: [outputs]}`` for every served run on the same inputs."""
        prompts = [self.template.build_prompt(self.tok, s, target_do, direction) for s in sources]
        results: dict[str, list[str]] = {}
        for spec in self.specs:
            if spec.adapter is None:  # SFT = base with adapters disabled
                if self.peft_model is not None:
                    with self.peft_model.disable_adapter():
                        results[spec.tag] = self._generate(prompts)
                else:
                    results[spec.tag] = self._generate(prompts)
            else:
                self.peft_model.set_adapter(spec.tag)
                results[spec.tag] = self._generate(prompts)
        return results


def main(
    text: str | None = None,
    runs: str | None = None,
    raw_dataset_path: str = "outputs/dialect_raw_new",
    split: str = "valid",
    target_do: str = "gangwondo",
    direction: str = "std2dia",
    n: int = 8,
    base: str = BASE,
    max_new_tokens: int = 64,
) -> None:
    """Compare all served runs on ad-hoc ``text`` or a held-out batch."""
    specs = _resolve_specs(runs, base)
    logger.info("Serving %d models: %s", len(specs), ", ".join(s.tag for s in specs))

    if text:
        rows = [{"source": text, "reference": None}]
    else:
        rows = load_eval_samples(raw_dataset_path, split, target_do, direction, n)
    sources = [r["source"] for r in rows]

    translator = MultiAdapterTranslator(specs, base, max_new_tokens=max_new_tokens)
    outputs = translator.translate(sources, target_do, direction)
    tags = [s.tag for s in specs]

    print("\n" + "=" * 80)
    print(f"MULTI-MODEL COMPARE  {target_do}/{direction}  ({len(tags)} models)")
    print("=" * 80)
    for i, row in enumerate(rows):
        print(f"\n[{i + 1}] 입력: {row['source']}")
        for tag in tags:
            print(f"    {tag:12s}: {outputs[tag][i]}")
        if row["reference"] is not None:
            print(f"    {'정답':12s}: {row['reference']}")


if __name__ == "__main__":
    fire.Fire(main)
