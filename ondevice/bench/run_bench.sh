#!/usr/bin/env bash
# run_bench.sh — phone-side SPEED measurement via llama-bench (brief §3 Tier1).
#
# llama-bench is a microbenchmark: it reports prefill (pp) and decode (tg) tok/s at
# fixed synthetic token counts, with its own warmup + repetition averaging. This is
# the number that goes in the portfolio table (brief §4b: table = llama-bench median,
# never the live demo). Quality (chrF/recon_bleu) is a SEPARATE track driven from the
# desktop (eval/gen_capture.py → eval/score_offline.py).
#
# Usage (Termux):
#   ./run_bench.sh -m models/kodialect-Q4_K_M.gguf -v Q4_K_M
#   ./run_bench.sh -m m.gguf -v Q4_K_M -p 64 -n 128 -r 5 -t 6 --fa 1 --ctk q8_0 --ctv q8_0
#
# Env: LLAMA_CPP_DIR (default $HOME/llama.cpp), OUTDIR (default ./logs).
set -euo pipefail

MODEL="" ; VARIANT=""
N_PROMPT=64 ; N_GEN=128 ; REPS=5 ; THREADS=6
FA=0 ; CTK="f16" ; CTV="f16"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
OUTDIR="${OUTDIR:-./logs}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)   MODEL="$2"; shift 2;;
    -v|--variant) VARIANT="$2"; shift 2;;
    -p|--prompt)  N_PROMPT="$2"; shift 2;;
    -n|--gen)     N_GEN="$2"; shift 2;;
    -r|--reps)    REPS="$2"; shift 2;;
    -t|--threads) THREADS="$2"; shift 2;;
    --fa)         FA="$2"; shift 2;;
    --ctk)        CTK="$2"; shift 2;;
    --ctv)        CTV="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

[[ -n "$MODEL" && -f "$MODEL" ]] || { echo "error: -m <gguf> required & must exist" >&2; exit 1; }
[[ -n "$VARIANT" ]] || VARIANT="$(basename "$MODEL" .gguf)"

BENCH_BIN="$LLAMA_CPP_DIR/build/bin/llama-bench"
[[ -x "$BENCH_BIN" ]] || BENCH_BIN="$LLAMA_CPP_DIR/llama-bench"
[[ -x "$BENCH_BIN" ]] || { echo "error: llama-bench not found under $LLAMA_CPP_DIR" >&2; exit 1; }

mkdir -p "$OUTDIR"
DEVICE="$(getprop ro.product.model 2>/dev/null || uname -m)"
LOG="$OUTDIR/bench_${VARIANT}_p${N_PROMPT}_n${N_GEN}.json"

echo "# device=$DEVICE variant=$VARIANT pp=$N_PROMPT tg=$N_GEN reps=$REPS threads=$THREADS fa=$FA kv=$CTK/$CTV"
# -o json → stable structured output; -r does warmup + repetition median internally.
"$BENCH_BIN" \
  -m "$MODEL" \
  -p "$N_PROMPT" -n "$N_GEN" \
  -r "$REPS" -t "$THREADS" \
  -fa "$FA" -ctk "$CTK" -ctv "$CTV" \
  -o json | tee "$LOG"

echo "# raw llama-bench json → $LOG"
echo "# next (desktop or phone python): record the schema row:"
echo "#   python logger.py --variant $VARIANT --bench_log $LOG --model_path $MODEL \\"
echo "#     --device $DEVICE --flash_attn $([ "$FA" = 1 ] && echo True || echo False) \\"
echo "#     --cache_type_k $CTK --cache_type_v $CTV --n_threads $THREADS"
