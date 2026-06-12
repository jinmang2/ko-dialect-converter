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

# Lint / format (works now) — ruff is the single tool (replaces black + isort)
ruff check . --fix     # lint + import order
ruff format .          # style (line length 100)

# Tests (no GPU needed) — after test suite is added
pytest tests/ -x -q -m "not gpu"

# Stage-3 GRPO (RTX 2060). Wrapper sets the bnb LD_LIBRARY_PATH + env and runs stage3
# with 2060-safe defaults (unsloth 16-bit LoRA, sft_merged + classifier_clean wired in).
scripts/run_grpo.sh training.max_steps=200 logger=wandb   # extra args = Hydra overrides
# Fast iteration / debugging (tiny slice, ~1 min): SMOKE_4BIT=0 SMOKE_N=64 SMOKE_STEPS=5
python scripts/smoke_grpo.py
# When GRPO won't run / OOMs / reward-hacks → .claude/skills/grpo-troubleshooting/SKILL.md

# Evaluation & serving (after a run completes)
# Cross-run leaderboard: ranks SFT + every outputs/grpo* arm by reconstruction_bleu↑
# (proxy-independent; plan §3 A4). Every metric prints with a direction arrow (↑/↓ — e.g.
# copy_margin is HIGHER-is-better) + a glossary. Reports a Pareto frontier over
# (copy_margin↑ × reconstruction_bleu↑), DFS (DIA-REFINE arXiv:2511.06680), and paired-
# bootstrap significance vs SFT (Koehn 2004). PER-REGION + weighted OVERALL aggregate.
python scripts/eval_leaderboard.py --target_do gangwondo --n 150          # one region
python scripts/eval_leaderboard.py --target_do gangwondo,gyeongsangdo     # several + OVERALL
python scripts/eval_leaderboard.py --all_regions --n 150                  # every region in data
# (jeju/jeolla/chungcheong appear automatically once their data + classifier labels land)
# Multi-model serving: ONE base (sft_merged) in VRAM, hot-swap adapters per request —
# compare SFT / arm1 / arm2 side by side without committing to one (6GB-safe):
python scripts/serve_compare.py --text "밥 먹었니?" --target_do gangwondo
python scripts/serve_compare.py --runs "grpo_arm1 grpo_arm2" --n 10 --target_do gangwondo
# Publish a chosen model to HF Hub with an evidence-rich card (metrics+Pareto+significance).
# --dry_run (default) writes README.md locally; --dry_run False uploads (needs HF_TOKEN):
python scripts/push_to_hub.py --model_path outputs/grpo_arm2_merged \
    --repo_id <you>/ko-dialect-arm2 --run_tag grpo_arm2 \
    --leaderboard 'outputs/eval_logs/leaderboard_overall_*.json'
# Single-run SFT-vs-one-GRPO deep dive (stratified buckets + qualitative samples):
python scripts/compare_sft_grpo.py --grpo_dir outputs/grpo_arm1 --target_do gangwondo
# Checkpoint sweep WITHIN one run (find the over-optimisation point):
python scripts/eval_grpo_checkpoints.py --grpo_dir outputs/grpo_arm1 --n 150
# Latency/throughput bench for one merged model (p50/p95, tok/s + chrF/copy_margin):
python scripts/merge_sft_lora.py --adapter_path outputs/grpo_arm1 \
    --base_model outputs/sft_merged --out outputs/grpo_arm1_merged
python scripts/bench_serving.py --model_path outputs/grpo_arm1_merged --target_do gangwondo
```

### Evaluation references (grounding for the leaderboard)
- **DIA-REFINE** (arXiv:2511.06680) — TDR + DFS metrics; n-gram metrics reward source-copying → motivates copy_margin.
- **MO-GRPO** (arXiv:2509.22047) — J-score/TDR overlap training rewards → circularity; select on proxy-independent signals only.
- **Mind the Style Gap** (arXiv:2502.15022) — copy_margin / J-score limits; LLM-judge ≤ human on content.
- **Koehn 2004** (EMNLP) — paired bootstrap resampling for chrF/BLEU significance (reliable ≥ ~300 sents).
- **TST trade-off** (arXiv:2010.12771 / 2010.12742) — style vs content are negatively correlated → report a Pareto frontier + harmonic-mean joint, don't force one rank.

> **Note:** `scripts/`, `configs/`, `tests/` 디렉토리는 구축 완료. 평가는 `scripts/eval.py`
> (config-driven, `configs/eval/default.yaml`)로 통합되어 있고, 양자화/학습 처리량 벤치는
> `scripts/quantize_eval.py` · `scripts/bench_train.py` 참고.

---

## Development Conventions

- **Python 3.11+**, typed where practical (`from __future__ import annotations`)
- **Formatter**: `ruff format` (line length 100) — single tool, also does import sorting via the `I` lint rule (black + isort were removed)
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
