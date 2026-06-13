---
name: grpo-troubleshooting
description: >
  Get Stage-3 GRPO running and converging on a small GPU. Use when GRPO
  "won't run", crashes on launch, OOMs, is impossibly slow, throws
  bitsandbytes/libnvJitLink errors, "Attempting to unscale FP16 gradients",
  TRL "unexpected keyword argument 'max_prompt_length'", or when training runs
  but reward-hacks / doesn't converge (completions pinned at max length,
  grad_norm nan). Trigger on "GRPO 안 돌아가", "GRPO 터진다", "reward hacking",
  "max_prompt_length 에러", "fp16 에러", "4bit 너무 느려".
---

# grpo-troubleshooting

Playbook for **Stage-3 GRPO** (`scripts/stage3_grpo.py`, `src/ko_dialect/training/grpo_trainer.py`)
on the dev box: **RTX 2060 (Turing SM 7.5, 6 GB)**, conda env `balaenoptera`
(trl 1.5.1, transformers 5.5.4, torch 2.11+cu130, unsloth 2026.6.1).

**One command that bakes in every fix below:** `scripts/run_grpo.sh`
(append Hydra overrides, e.g. `scripts/run_grpo.sh training.max_steps=200 logger=wandb`).

## The mental model

GRPO = generate N completions per prompt → score with rewards → policy-gradient.
On a weak GPU the **generation loop**, not the optimizer, dominates. Pick the backend
that makes *generation* fast and fits 6 GB, then worry about reward shaping.

## Issues in the order you hit them

### 1. Wrong interpreter / missing deps
`ModuleNotFoundError: trl` (or torch). The system `python` is not the project env.
→ Use `/home/jinmang2/miniconda3/envs/balaenoptera/bin/python` (has `ko_dialect`
editable + trl + unsloth). `run_grpo.sh` hardcodes it; override with `PYTHON=...`.

### 2. bitsandbytes: `OSError: libnvJitLink.so.13: cannot open shared object file`
bnb 0.49 (pulled in by unsloth even for 16-bit) needs CUDA-13 `libnvJitLink.so.13`,
which ships in the torch-cu130 wheels but isn't on the loader path.
→ `export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cu13/lib:$LD_LIBRARY_PATH`.
Verify: `python -c "from bitsandbytes.cextension import lib; print(lib is not None)"` → `True`.
Symptom if unfixed: 4-bit load dies with `Native code ... cquantize_blockwise_fp16_nf4`
→ transformers `RuntimeError: ... issues during automatic conversion of the weights`.

### 3. TRL `GRPOConfig.__init__() got an unexpected keyword argument 'max_prompt_length'`
trl 1.5.1 **removed** `max_prompt_length` from `GRPOConfig`.
→ Do **not** pass it (the dataclass field and the TRLGRPOConfig kwarg are both kept
commented in `grpo_trainer.py`). Prompts here are short, so no prompt truncation needed.
Check any TRL knob with: `python -c "from trl import GRPOConfig as C; print('NAME' in C.__dataclass_fields__)"`.

### 4. `backend=hf` (full fine-tune) → FP16 error + OOM
Two failures at once on this GPU:
- `ValueError: Attempting to unscale FP16 gradients` — the train() loop casts the whole
  model to fp16, so a full fine-tune has fp16 master weights the grad-scaler can't unscale.
- OOM — AdamW fp32 states for 0.5B (~4 GB) + grads + activations blow past 6 GB.
→ Don't full-fine-tune on a 2060. Use LoRA (only ~8.8M params train → fp32 optimizer
state is tiny, base stays frozen so no fp16-unscale).

### 5. 4-bit QLoRA *works* but is ~44x too slow
Measured: `load_in_4bit=true` → **~146 s/step**; `load_in_4bit=false` (16-bit LoRA)
→ **~3.4 s/step**, peak VRAM ~2.7 GB. Turing has no fast 4-bit kernel, so bnb
dequantizes every matmul per generated token — death by a thousand cuts in GRPO's
generation loop.
→ **Default to `backend=unsloth load_in_4bit=false apply_lora=true`** (16-bit LoRA).
Reserve 4-bit for Ampere+ with fast kernels (and ideally vLLM rollouts).

### 6. vLLM version mismatch
This env has vLLM 0.22 but trl 1.5.1 supports 0.12–0.18 (UserWarning on import).
→ Keep `use_vllm=false` on this box. Generation falls back to HF (fine for 16-bit LoRA).

### 7. It runs but reward-hacks / won't converge
Watch the TRL log dict each step:
- **`completions/mean_length == max_new_tokens`** for many steps = verbosity hacking
  (padding to the cap to incidentally match more gold words). In the 40-step probe this
  hit 11/40 steps. → Enable the DAPO-style `length` reward
  (`- {name: length, weight: 0.2}`, now on by default) and keep `max_new_tokens` near the
  gold length (~64, not 128). Tune `length_max_ratio` / `length_tolerance`.
- **reward scale skew** — `r_style ∈ [-1,1]` vs `r_content/r_length ∈ [0,1]`; GRPO sums
  *weighted* rewards before normalizing advantage, so style dominates by scale.
  → `normalize_rewards=true` (default) rescales each to [0,1] so weights are the only knob.
- **`grad_norm: nan`** on occasional steps (≈2/40), correlated with degenerate very-short
  completions. The fp16 grad-scaler skips those steps; non-fatal. If frequent, lower `lr`
  or `beta`, or raise `per_device_train_batch_size` for steadier groups.
- **`r_style` stuck negative** = SFT base under-dialectifies; the classifier (the reward
  model, `outputs/classifier_clean`, macro-F1 ~0.95) sees outputs as standard. Needs more
  steps and/or higher style weight — this is the signal RL must move, expect it slow.

### 8. `RuntimeError: The size of tensor a (256) must match the size of tensor b (293)` mid-run
unsloth builds the 4-D causal mask at `max_seq_length`, but `prompt + completion` exceeded
it. Because trl 1.5.1 dropped `max_prompt_length`, long prompts aren't truncated. Dialect
prompts are short (median ~73 tok) with a tail to ~520. Two-part fix (both applied):
- `max_seq_length=512` in the config (covers 99.98% of prompt+64).
- `stage3_grpo.py:filter_long_prompts()` drops rows with prompt > `max_seq_length -
  max_new_tokens - 8`, removing the ~0.1% outliers that still overflow. Adjust the budget
  if you change `max_new_tokens`/`max_seq_length`.

### 9. It "converges" (reward ↑) but the model gets WORSE — reward over-optimization
The classifier is a **hackable proxy**. In the 500-step run, reward/TDR rose the whole
time, but a checkpoint sweep on valid showed the opposite for real quality:

```
stage     tdr     chrf    bleu   eojeol
SFT(0)   0.187   68.60   57.08   0.179   <- chrf-best is the SFT base!
step-100 0.207   64.57   51.45   0.211
step-300 0.227   60.88   46.68   0.229
step-500 0.213   60.13   45.70   0.236
```

TDR climbed even **past the gold dialect's own TDR** (policy out-dialects real dialect),
while chrF/BLEU fell monotonically — the policy exaggerates / injects wrong-region
markers / corrupts proper nouns to please the classifier. eojeol-accuracy rising while
chrF drops = the targeted-eojeol signal is right but the global `style` reward drags the
whole sentence.

How to detect / handle:
- **Always evaluate with an independent metric**, not reward. Run
  `scripts/eval_grpo_checkpoints.py --grpo_dir <dir> --target_do <do>` to print the
  per-step TDR/chrF/BLEU/eojeol curve and pick the **chrf-best** checkpoint (often NOT
  the last). `scripts/compare_sft_grpo.py` does SFT-vs-GRPO side-by-side with samples.
- **Rebalance rewards** so fidelity anchors dominate the unbounded `style` push:
  demote `style` (0.5), raise `content` (chrF, 1.0), add `edit` (eojeol-targeted, 0.5),
  keep `length` (0.2). `normalize_rewards=true`.
- **Raise `beta`** (KL to SFT) — e.g. 0.04 → 0.1 — to keep the policy near the faithful
  SFT outputs (whose chrF is the ceiling).
- DAPO knobs available in trl 1.5.1: `epsilon_high` (Clip-Higher, anti entropy-collapse),
  `mask_truncated_completions` (drop token-cap cutoffs). `loss_type` already defaults to
  `dapo`. Note: an in-training generation eval callback is avoided here — generating
  while the training batch is resident OOMs the 6GB GPU; use the post-hoc sweep instead.

## Fast iteration loop

`scripts/smoke_grpo.py` runs a few steps on a tiny slice. Env knobs:
`SMOKE_4BIT=0|1`, `SMOKE_N=<rows>`, `SMOKE_STEPS=<n>`. Use it to A/B a backend or reward
set in ~1 min before committing to a long run. Always prepend the LD_LIBRARY_PATH fix
(or just call it through the same env `run_grpo.sh` sets up).

## Known-good launch

```bash
scripts/run_grpo.sh \
  training.max_steps=200 training.logging_steps=10 \
  logger=wandb experiment.run_name=grpo-16bitlora-len
```
Config defaults already point at `outputs/sft_merged` (base), `outputs/classifier_clean`
(reward model), `outputs/datasets/grpo` (data), backend unsloth 16-bit LoRA, rewards
style+content+length normalized.
