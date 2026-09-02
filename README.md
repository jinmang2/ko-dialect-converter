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
make install           # the package + core deps, exactly as uv.lock resolves them
make install-gpu       # + unsloth/xformers: 2060-safe LoRA kernels (CUDA only)
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

## Results

Full tables + reproducing commands in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md);
`python scripts/report.py` regenerates a live `docs/RESULTS.md` from `outputs/eval_logs/`.

### What's built
A complete, config-driven pipeline on a single 6 GB GPU, with an evaluation methodology
designed against known TST/dialect pitfalls (source-copying, reward circularity):

- **Selection is proxy-independent** — runs are ranked by `reconstruction_bleu`↑ with a
  Pareto frontier over `copy_margin`↑, DFS, and paired-bootstrap significance (Koehn 2004);
  classifier-derived metrics (tdr/dfs/jscore) are monitoring-only to avoid circularity.
- **Tooling:** unified `scripts/eval.py` (config-driven), cross-run leaderboard (per-region +
  weighted overall), quantization trade-off + serving/training benches, multi-adapter serving,
  GGUF export, HF-Hub publishing, generated results report, W&B + MLflow monitoring, SFT early
  stopping. ~210 CPU tests, ruff-clean, CI (lint + import + tests).

### Dialect classifier (Stage 2, = GRPO style reward)
| model | macro-F1 | accuracy |
|---|---|---|
| TextCNN on denoised data | **0.95** | 0.97 |

### Does GRPO beat SFT? — cross-run leaderboard (n=150)
**Dialect-dependent, no single winner.** SFT leads on the subtle **Gangwon** dialect; GRPO
clearly wins on the distinct **Gyeongsang** dialect (SFT is Pareto-dominated there).
`grpo_500` is the most robust single model — on the Pareto frontier in every scope, recon
deficit vs SFT never significant.

### Prosody supervision — A/B (v2 K-ToBI markers vs matched control, markers stripped, n=150)
| metric | Gangwon Δ(pros−ctrl) | Gyeongsang Δ(pros−ctrl) |
|---|---|---|
| **reconstruction_bleu↑** | **+2.5** (p=0.008) | **+4.8** (p<0.001) |
| copy_margin / tdr / dfs / eojeol | flat → slightly negative | flat → slightly negative |

**Prosody is a content-fidelity regularizer, not a dialectness lever** — it raises faithfulness
significantly but moves no dialect axis. Holds across two independent marker schemes (v1 ad-hoc
and v2 K-ToBI, Jun 2000 boundary tones; declination-aware, `<WAVE>` via F0 CoV; per-eojeol mode
available).

### Quantization (PTQ) — measured on `sft_merged`
| variant | peak VRAM | p50 latency | chrF |
|---|---|---|---|
| fp16 | 1024 MB | 488 ms | 68.5 |
| **bnb 4-bit (NF4)** | **518 MB (51%)** | **985 ms (~2× slower)** | 70.7 |

4-bit is a **~2× memory win and a latency loss** on RTX 2060 (Turing dequant overhead; only the
weights shrink while activations/KV dominate VRAM), with no quality loss → use **GGUF Q4_K_M**
for speed.

### Training throughput
unsloth: **286 tok/s, 1.3 GB peak VRAM** (bs=1/seq=256). DeepSpeed and vanilla bnb-LoRA training
do **not** run on this box (no CUDA toolkit/`nvcc`; Turing BF16-unscale wall) → unsloth is the
only working training path here. Config + bench stay ready for a multi-GPU/toolkit machine.

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
