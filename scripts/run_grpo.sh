#!/usr/bin/env bash
# Stage 3 GRPO launcher — bakes in the env fixes found while getting GRPO to run on a
# 6GB RTX 2060 (Turing). See .claude/skills/grpo-troubleshooting/SKILL.md for the full
# debugging story. Pass any extra args straight through as Hydra overrides, e.g.:
#
#   scripts/run_grpo.sh training.max_steps=200 logger=wandb
#   scripts/run_grpo.sh training.rewards='[{name:style,weight:1.0},{name:content,weight:0.5}]'
#
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) Interpreter: the uv-managed project venv, which has ko_dialect + trl + unsloth.
#    Override with PYTHON=... if your env differs.
PYTHON="${PYTHON:-./.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "[run_grpo] no interpreter at $PYTHON — create it with:" >&2
  echo "  uv sync --extra dev --extra gpu" >&2
  exit 1
fi

# 2) bitsandbytes fix: bnb 0.49 looks for libnvJitLink.so.13 (CUDA 13 / torch cu130) but
#    it is not on the default loader path, so `import bitsandbytes` fails with
#    "OSError: libnvJitLink.so.13: cannot open shared object file". unsloth pulls bnb in
#    regardless of 4-bit, so expose the wheel-bundled lib dir.
SITE_PKGS="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
NVJITLINK_DIR="$SITE_PKGS/nvidia/cu13/lib"
if [[ -d "$NVJITLINK_DIR" ]]; then
  export LD_LIBRARY_PATH="$NVJITLINK_DIR:${LD_LIBRARY_PATH:-}"
fi

# 3) Single-GPU, quiet tokenizers (avoids fork warnings during dataset map).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# 3b) expandable_segments: on this 6GB card the default CUDA allocator fragments near the
#     ceiling (~5.3GB) and thrashes mid-run — observed a 4.5x slowdown (9 -> 39 s/step).
#     Expandable segments grow smoothly: it both removes the thrash AND cuts reserved VRAM
#     (5.3GB -> ~3.0GB here), so it is strictly better on Turing/6GB. Override if needed.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[run_grpo] python=$PYTHON"
echo "[run_grpo] LD_LIBRARY_PATH+=$NVJITLINK_DIR"
echo "[run_grpo] extra args: $*"

# 4) Run. Config defaults (configs/grpo.yaml + configs/training/grpo.yaml) already point
#    at outputs/sft_merged, outputs/classifier_clean, outputs/datasets/grpo and the
#    2060-safe unsloth 16-bit-LoRA backend.
exec "$PYTHON" scripts/stage3_grpo.py "$@"
