# KoDialect

**Korean dialect ↔ standard Korean** translation with a small causal LM
(`Qwen/Qwen2.5-0.5B-Instruct`), fine-tuned via **QLoRA SFT → GRPO** and exported to
**GGUF (Q4_K_M)** for CPU/edge serving.

Target hardware: a single **RTX 2060** (Turing, 6 GB, fp16-only — never bf16).

- **Dev guide / conventions:** [`CLAUDE.md`](CLAUDE.md)
- **Experiment results & verdicts:** [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)

---

## Pipeline

```
Stage 0  build datasets   raw Arrow → SFT / GRPO / classifier datasets
Stage 1  SFT              QLoRA on Qwen2.5-0.5B (std↔dialect)
Stage 2  classifier       TextCNN dialect classifier (= GRPO style reward, macro-F1 ≈ 0.95)
Stage 3  GRPO             fidelity-anchored RL on top of the SFT model
Eval/serve               cross-run leaderboard, multi-adapter serving, GGUF export, HF push
```

## Install

```bash
make install-torch     # torch (CUDA build) first
make install           # the package + core deps  (pip install -e ".[dev]")
make install-unsloth   # 2060-safe LoRA kernels
make install-llama     # llama-cpp-python (GGUF serving)   — optional
make check             # sanity: torch/CUDA, trl/peft/transformers
```

## Data preparation

Download the AI-Hub *중·노년층 한국어 방언* data (강원도/경상도, more regions optional)
from [aihub.or.kr](https://aihub.or.kr), unzip under `data/`, then:

```bash
sh scripts/unzip.sh                              # unzip raw archives
python scripts/prepare_data.py --data_path data  # raw JSON → Arrow (outputs/dialect_raw_new)
python scripts/stage0_build_datasets.py --raw_dataset_path outputs/dialect_raw_new
```

> Some AI-Hub JSON files are malformed; see the upstream issue notes in `prepare_data.py`.

## Train

```bash
# Stage 1 — SFT (Hydra config: configs/sft.yaml)
python scripts/stage1_sft.py data.sft_dataset_path=outputs/datasets/sft

# Stage 2 — dialect classifier (GRPO reward model)
python scripts/stage2_train_classifier.py data.cls_dataset_path=outputs/datasets/classifier

# Stage 3 — GRPO (RTX 2060 wrapper sets bnb env + 2060-safe defaults)
scripts/run_grpo.sh training.max_steps=200 logger=wandb
```

GRPO won't run / OOMs / reward-hacks → `.claude/skills/grpo-troubleshooting/SKILL.md`.

## Evaluate & serve

```bash
# Unified entrypoint (config-driven via configs/eval/default.yaml):
python scripts/eval.py config                       # show resolved config + metric registry
python scripts/eval.py leaderboard --all_regions    # == scripts/eval_leaderboard.py
python scripts/eval.py single --model_path outputs/sft_merged --target_do gangwondo

# Cross-run leaderboard — ranks SFT + every GRPO arm per region and overall by
# reconstruction_bleu↑ (proxy-independent), with DFS, Pareto frontier, and
# paired-bootstrap significance (Koehn 2004). Every metric shows its ↑/↓ direction.
# Add --wandb_project ko-dialect to push per-region metric panels to W&B.
python scripts/eval_leaderboard.py --all_regions --n 150

# Quantization trade-off (quality × size × latency) for a finished model:
python scripts/quantize_eval.py --model_path outputs/sft_merged --target_do gangwondo

# Training throughput bench (unsloth vs DeepSpeed ZeRO-offload):
python scripts/bench_train.py --backend unsloth --max_steps 20

# Serve SFT + GRPO arms together (one base in VRAM, hot-swapped adapters) and compare:
python scripts/serve_compare.py --text "밥 먹었니?" --target_do gangwondo

# Latency/throughput + quality bench for one merged model:
python scripts/bench_serving.py --model_path outputs/sft_merged --target_do gangwondo

# Export to GGUF (Q4_K_M) for CPU/edge serving via llama.cpp:
python scripts/export_gguf.py convert  --model outputs/sft_merged --out outputs/gguf/model_f16.gguf
python scripts/export_gguf.py quantize --input outputs/gguf/model_f16.gguf
python scripts/export_gguf.py serve    --gguf outputs/gguf/model_Q4_K_M.gguf

# Publish a chosen model to the HF Hub with an evidence-rich card:
python scripts/push_to_hub.py --model_path outputs/grpo_500_merged \
    --repo_id <you>/ko-dialect --run_tag grpo_500 \
    --leaderboard 'outputs/eval_logs/leaderboard_overall_*.json'
```

## Key findings

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the full tables. In brief:

- **GRPO's advantage is dialect-dependent** — SFT leads on the subtle Gangwon dialect,
  GRPO clearly wins on the distinct Gyeongsang dialect (SFT is Pareto-dominated there).
  `grpo_500` is the most robust single model (on the frontier in every scope).
- **Prosody (F0-marker) supervision is a content-fidelity regularizer**, not a dialectness
  lever — it significantly raises reconstruction_bleu without changing the dialect axes.
  Marker scheme is now **K-ToBI-grounded** (Jun 2000 boundary tones), declination-aware,
  with `<WAVE>` reactivated via F0 coefficient-of-variation and per-eojeol markers available.
  A v2 A/B confirms this holds across schemes: the principled K-ToBI markers replicate the
  v1 recon_bleu gain (significant, both regions) but **do not** turn it into a dialect gain.
- **4-bit PTQ is a memory win, not a latency win on RTX 2060** — bitsandbytes NF4 cuts the
  footprint 4× with no quality loss but runs ~2× *slower* (Turing dequant overhead); use
  GGUF Q4_K_M for speed. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) §3–4.

## Citations

```bibtex
@article{park-etal-2025-steering,
  title  = {Steering LLMs toward Korean Local Speech: Iterative Refinement Framework for Faithful Dialect Translation},
  author = {Park, Keunhyeung and Yu, Seunguk and Kim, Youngbin},
  journal = {arXiv preprint arXiv:2511.06680},
  year   = {2025},
  url    = {https://arxiv.org/abs/2511.06680},
}
```
