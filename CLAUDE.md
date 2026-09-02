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
| Target hardware | NVIDIA RTX 2060 — Turing (SM 7.5), 6 GB VRAM, **no native BF16** |

### Hardware constraint
RTX 2060 is Turing (SM 7.5). BF16 requires SM 8.0+ (Ampere).
**Always use `torch_dtype=torch.float16` or 4-bit quantization.**
Never set `bf16=True` in TrainingArguments — use `fp16=True`.

---

## Key Commands

```bash
# Setup — uv owns the environment (.venv from pyproject.toml + uv.lock). No conda.
uv sync --extra dev              # everyday dev env      (= make install)
uv sync --extra dev --extra gpu  # + unsloth/xformers    (= make install-gpu)
uv run pre-commit install
# Run anything in the env with `uv run <cmd>` (or activate .venv). `make install-vllm`
# adds vLLM from its pinned git rev; `make install-webui` installs open-webui isolated.

# Lint / format (works now) — ruff is the single tool (replaces black + isort)
uv run ruff check . --fix     # lint + import order
uv run ruff format .          # style (line length 100)

# Tests (no GPU needed) — after test suite is added
uv run pytest tests/ -x -q -m "not gpu"

# Stage-3 GRPO (RTX 2060). Wrapper sets the bnb LD_LIBRARY_PATH + env and runs stage3
# with 2060-safe defaults (unsloth 16-bit LoRA, sft_merged + classifier_clean wired in).
scripts/run_grpo.sh training.max_steps=200 logger=wandb   # extra args = Hydra overrides
# Fast iteration / debugging (tiny slice, ~1 min): SMOKE_4BIT=0 SMOKE_N=64 SMOKE_STEPS=5
uv run python scripts/smoke_grpo.py
# When GRPO won't run / OOMs / reward-hacks → .claude/skills/grpo-troubleshooting/SKILL.md

# Evaluation & serving (after a run completes)
# Cross-run leaderboard: ranks SFT + every outputs/grpo* arm by reconstruction_bleu↑
# (proxy-independent; plan §3 A4). Every metric prints with a direction arrow (↑/↓ — e.g.
# copy_margin is HIGHER-is-better) + a glossary. Reports a Pareto frontier over
# (copy_margin↑ × reconstruction_bleu↑), DFS (DIA-REFINE arXiv:2511.06680), and paired-
# bootstrap significance vs SFT (Koehn 2004). PER-REGION + weighted OVERALL aggregate.
uv run python scripts/eval_leaderboard.py --target_do gangwondo --n 150          # one region
uv run python scripts/eval_leaderboard.py --target_do gangwondo,gyeongsangdo     # several + OVERALL
uv run python scripts/eval_leaderboard.py --all_regions --n 150                  # every region in data
# (jeju/jeolla/chungcheong appear automatically once their data + classifier labels land)
# Multi-model serving: ONE base (sft_merged) in VRAM, hot-swap adapters per request —
# compare SFT / arm1 / arm2 side by side without committing to one (6GB-safe):
uv run python scripts/serve_compare.py --text "밥 먹었니?" --target_do gangwondo
uv run python scripts/serve_compare.py --runs "grpo_arm1 grpo_arm2" --n 10 --target_do gangwondo
# Publish a chosen model to HF Hub with an evidence-rich card (metrics+Pareto+significance).
# --dry_run (default) writes README.md locally; --dry_run False uploads (needs HF_TOKEN):
uv run python scripts/push_to_hub.py --model_path outputs/grpo_arm2_merged \
    --repo_id <you>/ko-dialect-arm2 --run_tag grpo_arm2 \
    --leaderboard 'outputs/eval_logs/leaderboard_overall_*.json'
# Single-run SFT-vs-one-GRPO deep dive (stratified buckets + qualitative samples):
uv run python scripts/compare_sft_grpo.py --grpo_dir outputs/grpo_arm1 --target_do gangwondo
# Checkpoint sweep WITHIN one run (find the over-optimisation point):
uv run python scripts/eval_grpo_checkpoints.py --grpo_dir outputs/grpo_arm1 --n 150
# Latency/throughput bench for one merged model (p50/p95, tok/s + chrF/copy_margin):
uv run python scripts/merge_sft_lora.py --adapter_path outputs/grpo_arm1 \
    --base_model outputs/sft_merged --out outputs/grpo_arm1_merged
uv run python scripts/bench_serving.py --model_path outputs/grpo_arm1_merged --target_do gangwondo
```

### Corpus + eval-validity facts (measured — `docs/DATA_ANALYSIS.md`)
Regenerate with `uv run python scripts/analyze_data.py all`. Load-bearing findings:
- **Data covers 2 of the 5 labelled regions.** `labels.py` reserves ids for jeollado /
  jejudo / chungcheongdo, but `raw_data/` holds only 139-1 (강원도·경상도). Also note
  `prepare_data.py` defaults to `raw_data/new_dialect`, which does not exist on disk.
- **~42% of rows are `is_identical`** (dialect text == standard text) and every path drops
  them. `speech_kind` almost fully explains which: `read` 4.5% identical vs `say` 62.9%.
- **Transfer is small and length-preserving** — usable pairs change 25–33% of eojeol, and
  24–29% differ by a single eojeol. This is the data-side case for `copy_margin`.
- **first-n eval selection skews easy, by a different amount per region** (TVD 46.9%
  gangwon vs 15.2% gyeongsang). `load_eval_samples(..., strategy="stratified")` fixes it
  deterministically (TVD → 0.2%); the default stays `head` pending a re-run decision.
- **`reconstruction_bleu` (the ranking metric) is measured off-template** — `REVERSE_PROMPT`
  is plain text with no ChatML, system prompt, or region, unlike everything the model saw
  in training. Open decision, see DATA_ANALYSIS.md §5.
- `speech_kind` / `intent` / `emotion` are populated and read by nothing;
  `dialect_eojeol_map` is 43.2% empty, so `eojeol_accuracy` covers ~57% of rows.

### Evaluation references (grounding for the leaderboard)
- **DIA-REFINE** (arXiv:2511.06680) — TDR + DFS metrics; n-gram metrics reward source-copying → motivates copy_margin.
- **MO-GRPO** (Ichihara et al., arXiv:2509.22047) — per-objective z-norm-then-sum aggregation (prevents one reward axis dominating). The "select on proxy-independent signals only" rule is *our* inference from its reward-hacking analysis, not an explicit paper recommendation (see docs/REFERENCES.md A2).
- **Mind the Style Gap** (Pauli, Augenstein & Assent, EMNLP Findings 2025; arXiv:2502.15022) — copy_margin / J-score limits; content metrics must be *style-aware*; same-size LLM autoraters underperform style-aware metrics (NOT a blanket "LLM-judge ≤ human" claim).
- **Koehn 2004** (EMNLP) — paired bootstrap resampling for chrF/BLEU significance (reliable ≥ ~300 sents).
- **TST trade-off** — style vs content in tension: survey arXiv:2010.12742 (Hu et al. 2022); negative-correlation finding arXiv:2312.14708 (Mukherjee, Kasner & Dušek 2022); direct-reward TST arXiv:2010.12771 → report a Pareto frontier + harmonic joint, don't force one rank. (Full grounding + corrections: docs/REFERENCES.md.)

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
