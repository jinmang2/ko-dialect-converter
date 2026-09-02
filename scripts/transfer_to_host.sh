#!/usr/bin/env bash
# Move this working tree to another machine over Tailscale, without shipping 76 GB of
# regenerable bytes.
#
# The repo is 79 GB on disk but only ~73 MB of that is in git. Everything else is
# gitignored output, and most of it is *derived* — datasets rebuild from raw, checkpoints
# rebuild from training, optimizer state is useless unless you resume the exact run. This
# script encodes which tier is which, measured on 2026-08-04:
#
#   code     ~80 MB   git tree + configs + notebooks + the gitignored files git would lose
#   results  ~3.1 GB  models, adapters, GRPO checkpoint adapters, eval_logs  <- irreplaceable
#   raw      ~33 GB   raw_data/ — re-downloadable (~5.3 GB of zips) but needs a live AI-Hub
#                     approval, which expires. Send it if you want to be independent of that.
#   (never)  ~40 GB   optimizer.pt/rng/scheduler, outputs/datasets*, outputs/dialect_raw*,
#                     old outputs/classifier, speedruns, caches, llama.cpp
#
# Usage:
#   scripts/transfer_to_host.sh <tailscale-host> [--tier code|results|raw|all] [--go]
#
# Dry run by default — it prints what WOULD move and the byte total. Add --go to send.
#   scripts/transfer_to_host.sh mypc --tier results          # preview
#   scripts/transfer_to_host.sh mypc --tier results --go     # send
#
# The destination path mirrors the source path under the remote user's home.
set -euo pipefail

HOST="${1:-}"
TIER="results"
GO=0
DEST_DIR="ko_dialect"

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --go) GO=1; shift ;;
    --dest) DEST_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "usage: $0 <tailscale-host> [--tier code|results|raw|all] [--dest DIR] [--go]" >&2
  exit 1
fi

cd "$(dirname "$0")/.."

# --- never worth sending -----------------------------------------------------------
# Training optimiser state: 18 GB across 384 files, and worthless unless you resume the
# identical run on the identical GPU. The model weights next to it are what you want.
COMMON_EXCLUDES=(
  --exclude='optimizer.pt' --exclude='rng_state.pth'
  --exclude='scheduler.pt' --exclude='scaler.pt'
  --exclude='__pycache__/' --exclude='.mypy_cache/' --exclude='.ruff_cache/'
  --exclude='.pytest_cache/' --exclude='unsloth_compiled_cache/'
  --exclude='llama.cpp/'            # re-clone; 587 MB of upstream source
  --exclude='.venv/' --exclude='venv/'
)

# Derived from raw_data via prepare_data.py / stage0_build_datasets.py. Rebuilding costs
# minutes; sending costs 12 GB.
DERIVED_EXCLUDES=(
  --exclude='outputs/datasets*'
  --exclude='outputs/dialect_raw*'
  --exclude='outputs/speedrun_*'
  --exclude='outputs/classifier/'   # superseded: macro-F1 0.776 vs classifier_clean 0.947
  --exclude='outputs/classifier_clean/checkpoint-*'  # root holds the final model (162 MB)
  --exclude='outputs/sft/checkpoint-*'
  --exclude='outputs/sft_*/checkpoint-*'
  --exclude='data/aihub_samples/'   # re-fetchable with scripts/aihub_fetch.py
  --exclude='wandb/'                # runs live on the W&B server
)
# NOTE: outputs/grpo_*/checkpoint-* IS kept — eval_grpo_checkpoints.py sweeps those
# adapters to find the over-optimisation point, and they are ~90 MB per run once the
# optimiser state above is stripped.

case "$TIER" in
  code)
    # Includes gitignored-but-needed files: .env.local (API key) and the deepspeed config
    # that .gitignore's blanket `*.json` rule swallows.
    PATHS=(scripts src configs tests docs ondevice notebooks
           README.md CLAUDE.md AGENTS.md Makefile pyproject.toml .gitignore .env.local)
    EXTRA=()
    ;;
  results)
    PATHS=(outputs)
    EXTRA=("${DERIVED_EXCLUDES[@]}")
    ;;
  raw)
    PATHS=(raw_data)
    EXTRA=()
    ;;
  all)
    PATHS=(.)
    EXTRA=("${DERIVED_EXCLUDES[@]}" --exclude='.git/')
    ;;
  *)
    echo "unknown tier: $TIER (code|results|raw|all)" >&2; exit 1 ;;
esac

RSYNC_ARGS=(-avh --partial --info=progress2 "${COMMON_EXCLUDES[@]}" "${EXTRA[@]}")
[[ $GO -eq 1 ]] || RSYNC_ARGS+=(--dry-run)

echo "=========================================================="
echo " host : $HOST   tier : $TIER   dest : ~/$DEST_DIR"
echo " mode : $([[ $GO -eq 1 ]] && echo 'TRANSFER' || echo 'DRY RUN (add --go to send)')"
echo "=========================================================="

ssh "$HOST" "mkdir -p ~/$DEST_DIR"
rsync "${RSYNC_ARGS[@]}" "${PATHS[@]}" "$HOST:~/$DEST_DIR/"

if [[ $GO -eq 0 ]]; then
  echo
  echo "dry run only. Re-run with --go to actually transfer."
fi
