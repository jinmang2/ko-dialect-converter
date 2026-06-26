#!/usr/bin/env bash
# launch_server.sh — boot llama-server on the phone (Termux, S25 Ultra) for KoDialect.
#
# Sole responsibility: bring the server up reproducibly with the knobs the brief's
# measurement tracks need (Tier0/1 = serve a GGUF variant; Tier2 = KV-cache flags).
# Quality scoring stays on the desktop (O3) — this script never evaluates.
#
# Usage (all flags optional; sane S25U defaults):
#   ./launch_server.sh -m models/kodialect-Q4_K_M.gguf
#   ./launch_server.sh -m models/kodialect-Q4_K_M.gguf -c 4096 --fa on \
#       --cache-type-k q8_0 --cache-type-v q8_0 -t 6
#   ./launch_server.sh -m m.gguf --prefix-cache --n-keep 64   # Tier2 prefix reuse
#
# Env overrides: LLAMA_CPP_DIR (default: $HOME/llama.cpp), HOST, PORT.
set -euo pipefail

# ---- defaults (Snapdragon 8 Elite: 2 prime + 6 perf cores → 6 perf threads) ----
MODEL=""
N_CTX=4096
FLASH_ATTN="on"            # -fa: REQUIRED whenever KV cache is quantized (brief §3 Tier2)
CACHE_TYPE_K="f16"
CACHE_TYPE_V="f16"
N_THREADS=6
N_GPU_LAYERS=0            # CPU-only path (brief: llama.cpp CPU is the measured runtime)
PREFIX_CACHE=0           # Tier2 Exp A: reuse fixed system-prompt KV across turns
N_KEEP=-1                # sink tokens kept on context shift (-1 = server default)
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
EXTRA=()

usage() { sed -n '2,20p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)          MODEL="$2"; shift 2;;
    -c|--ctx-size)       N_CTX="$2"; shift 2;;
    --fa|--flash-attn)   FLASH_ATTN="$2"; shift 2;;
    --cache-type-k)      CACHE_TYPE_K="$2"; shift 2;;
    --cache-type-v)      CACHE_TYPE_V="$2"; shift 2;;
    -t|--threads)        N_THREADS="$2"; shift 2;;
    --n-gpu-layers)      N_GPU_LAYERS="$2"; shift 2;;
    --prefix-cache)      PREFIX_CACHE=1; shift;;
    --n-keep)            N_KEEP="$2"; shift 2;;
    --host)              HOST="$2"; shift 2;;
    --port)              PORT="$2"; shift 2;;
    --)                  shift; EXTRA+=("$@"); break;;
    -h|--help)           usage 0;;
    *)                   echo "unknown arg: $1" >&2; usage 1;;
  esac
done

[[ -n "$MODEL" ]] || { echo "error: -m/--model <gguf> is required" >&2; usage 1; }
[[ -f "$MODEL" ]] || { echo "error: model not found: $MODEL" >&2; exit 1; }

# locate llama-server (cmake build layout or flat dir)
SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
[[ -x "$SERVER_BIN" ]] || SERVER_BIN="$LLAMA_CPP_DIR/llama-server"
[[ -x "$SERVER_BIN" ]] || {
  echo "error: llama-server not found under $LLAMA_CPP_DIR" >&2
  echo "       build it: cmake -B build && cmake --build build -j" >&2
  exit 1
}

CMD=(
  "$SERVER_BIN"
  --model "$MODEL"
  --ctx-size "$N_CTX"
  --threads "$N_THREADS"
  --n-gpu-layers "$N_GPU_LAYERS"
  --host "$HOST" --port "$PORT"
  --flash-attn "$FLASH_ATTN"
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
)
# Guard rail (brief §3): quantized KV without -fa silently falls back to a slow
# per-attention dequant path. Refuse the footgun rather than mis-measure.
if [[ ( "$CACHE_TYPE_K" != "f16" || "$CACHE_TYPE_V" != "f16" ) && "$FLASH_ATTN" != "on" ]]; then
  echo "error: quantized KV cache (k=$CACHE_TYPE_K v=$CACHE_TYPE_V) requires --fa on" >&2
  echo "       (without flash-attn every attention dequants → silent slow fallback)" >&2
  exit 1
fi
[[ "$PREFIX_CACHE" -eq 1 ]] && CMD+=(--cache-reuse 256)   # reuse matching prompt prefix KV
[[ "$N_KEEP" != "-1" ]] && CMD+=(--keep "$N_KEEP")
[[ ${#EXTRA[@]} -gt 0 ]] && CMD+=("${EXTRA[@]}")

echo "# device : $(getprop ro.product.model 2>/dev/null || uname -m)"
echo "# model  : $MODEL ($(du -m "$MODEL" 2>/dev/null | cut -f1) MB on disk)"
echo "# kv     : k=$CACHE_TYPE_K v=$CACHE_TYPE_V  fa=$FLASH_ATTN  ctx=$N_CTX  threads=$N_THREADS"
echo "# serve  : http://$HOST:$PORT   (webui + /completion + /v1/chat/completions)"
echo "# ----"
echo "+ ${CMD[*]}"
exec "${CMD[@]}"
