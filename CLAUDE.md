# CLAUDE.md — KoDialect Project Guide

This file is read by Claude Code at the start of every session.
Keep it up-to-date as the codebase evolves.

---

## Project Overview

**KoDialect** — Korean dialect → standard Korean translation using a small causal LM.

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` (≈ 0.8 B params) |
| Training method | QLoRA (4-bit NF4) via Unsloth + TRL `SFTTrainer` |
| Alternative | PTQ + LoRA (bitsandbytes AWQ/GPTQ post-quant + adapter) |
| Export | GGUF (Q4_K_M) via llama.cpp for CPU/edge serving |
| Target hardware | NVIDIA RTX 2060 — Turing (SM 7.5), 8 GB VRAM, **no native BF16** |

### Hardware constraint
RTX 2060 is Turing (SM 7.5). BF16 requires SM 8.0+ (Ampere).
**Always use `torch_dtype=torch.float16` or 4-bit quantization.**
Never set `bf16=True` in TrainingArguments — use `fp16=True`.

---

## Key Commands

```bash
# Setup (dev) — after refactoring is complete
pip install -e ".[dev]"
pre-commit install

# Lint / format (works now)
ruff check .
black .
isort .

# Tests (no GPU needed) — after test suite is added
pytest tests/ -x -q -m "not gpu"

# Stage-3 GRPO (RTX 2060). Wrapper sets the bnb LD_LIBRARY_PATH + env and runs stage3
# with 2060-safe defaults (unsloth 16-bit LoRA, sft_merged + classifier_clean wired in).
scripts/run_grpo.sh training.max_steps=200 logger=wandb   # extra args = Hydra overrides
# Fast iteration / debugging (tiny slice, ~1 min): SMOKE_4BIT=0 SMOKE_N=64 SMOKE_STEPS=5
python scripts/smoke_grpo.py
# When GRPO won't run / OOMs / reward-hacks → .claude/skills/grpo-troubleshooting/SKILL.md
```

> **Note:** `scripts/`, `configs/`, `tests/` 디렉토리는 리빌딩 세션에서 생성 예정.

---

## Development Conventions

- **Python 3.11+**, typed where practical (`from __future__ import annotations`)
- **Formatter**: `black` (line length 88) + `isort`
- **Linter**: `ruff` — treat warnings as errors in CI
- **No notebooks in `src/`** — use `scripts/` for one-off experiments
- **Config files are YAML** under `configs/` — never hardcode hyperparams in Python
- Commit messages: imperative mood, `<scope>: <what>` (e.g. `train: add gradient checkpointing`)

## What NOT to do

- Do **not** set `bf16=True` anywhere — RTX 2060 will silently degrade or crash
- Do **not** load full-precision weights without quantization on this GPU (OOM)
- Do **not** commit `*.pt`, `*.bin`, `*.gguf`, `*.safetensors` — large files go to HuggingFace Hub or are `.gitignore`d
- Do **not** add `*.json` to git unless it is a config file (data JSON is gitignored)
- Do **not** modify `pyproject.toml` dependencies without also testing install

---

## CI/CD Cheatsheet

| Trigger | What happens |
|---|---|
| push / PR to `main` | Lint + import tests (no GPU) |
| PR opened/synced | Auto-label by changed file paths |
| Issue opened | Auto-label by keyword in title/body |
| CI fails on `main` | Auto-creates GitHub Issue with failure link |
| Issue/PR inactive 30 days | Stale bot warns, closes after 7 more days |

---

## Workflow with Claude Code

- Run `/code-review` before opening a PR
- Run `/simplify` after large refactors
- Use `/schedule` for recurring experiment monitoring
- Claude Code reads this file automatically — keep it accurate
- For GPU-dependent changes, test locally before asking Claude to verify

---

## Glossary

| Term | Meaning |
|---|---|
| QLoRA | Quantized LoRA — train adapter on 4-bit frozen backbone |
| PTQ | Post-Training Quantization (e.g. GPTQ, AWQ) |
| GGUF | llama.cpp weight format for CPU/edge inference |
| SFTTrainer | TRL's Supervised Fine-Tuning Trainer |
| Unsloth | Optimized CUDA kernels for LoRA on small GPUs |
