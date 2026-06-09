#!/usr/bin/env bash
# Run KoDialect evaluation inside a detached tmux session you can drag-scroll.
#
# Quick start:
#   scripts/run_eval_tmux.sh run      # start eval in the background
#   scripts/run_eval_tmux.sh attach   # watch it live  (drag UP to scroll logs)
#   scripts/run_eval_tmux.sh logs     # tail the logfile from outside tmux
#   scripts/run_eval_tmux.sh stop     # kill the session
#   scripts/run_eval_tmux.sh status   # is it running?
#
# Inside the session:
#   - Drag the mouse UPWARD     -> scroll back through logs (the feature you wanted)
#   - Mouse wheel              -> scroll too
#   - Drag-select then release  -> copies text;  press q to exit scroll mode
#   - Ctrl-b d                  -> detach (eval keeps running in the background)
#
# Override any eval hyperparameter via env vars, e.g.:
#   BATCH_SIZE=32 N_SAMPLES=0 scripts/run_eval_tmux.sh run
set -euo pipefail

# --- Resolve paths ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TMUX_CONF_PATH="${SCRIPT_DIR}/eval.tmux.conf"

# Isolated tmux server + session so we never touch your global tmux/config.
SOCKET="ko_eval"
SESSION="${SESSION:-eval}"
TM=(tmux -L "${SOCKET}" -f "${TMUX_CONF_PATH}")

# --- Eval configuration (override via env) ------------------------------------
# Defaults match README usage: scripts/evaluate.py outputs/grpo outputs/dialect_raw_new outputs/classifier
MODEL_PATH="${MODEL_PATH:-outputs/sft}"
RAW_DATASET_PATH="${RAW_DATASET_PATH:-outputs/dialect_raw_new}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-outputs/classifier}"
TARGET_DO="${TARGET_DO:-gangwondo}"
SPLIT="${SPLIT:-valid}"
N_SAMPLES="${N_SAMPLES:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# Conda: a fresh tmux shell is non-interactive and does NOT auto-activate your
# env, so the eval would run against base python (missing sentencepiece etc).
# Capture the env active at launch (override with CONDA_ENV=...) and re-activate
# it inside the session. Set CONDA_ENV="" to skip activation entirely.
CONDA_ENV="${CONDA_ENV-${CONDA_DEFAULT_ENV:-}}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/eval_logs}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eval_${TARGET_DO}_${TS}.log"
RESULT_FILE="${RESULT_FILE:-${LOG_DIR}/eval_${TARGET_DO}_${TS}.json}"

# Stable pointer to the most recent run's log, for `logs`.
LATEST_LINK="${LOG_DIR}/latest.log"

c_bold=$'\033[1m'; c_grn=$'\033[32m'; c_yel=$'\033[33m'; c_cyn=$'\033[36m'; c_off=$'\033[0m'
info() { printf '%s[eval]%s %s\n' "${c_cyn}" "${c_off}" "$*"; }
ok()   { printf '%s[eval]%s %s\n' "${c_grn}" "${c_off}" "$*"; }
warn() { printf '%s[eval]%s %s\n' "${c_yel}" "${c_off}" "$*"; }

session_exists() { "${TM[@]}" has-session -t "${SESSION}" 2>/dev/null; }

conda_activate_snippet() {
  # Emit a `conda activate <env> && ` prefix, or nothing if no env / conda not found.
  [ -z "${CONDA_ENV}" ] && return 0
  local base=""
  if [ -n "${CONDA_EXE:-}" ]; then
    base="$(dirname "$(dirname "${CONDA_EXE}")")"
  elif [ -n "${CONDA_PREFIX:-}" ]; then
    # CONDA_PREFIX points at the active env; strip envs/<name> to get base.
    base="${CONDA_PREFIX%%/envs/*}"
  fi
  if [ -n "${base}" ] && [ -f "${base}/etc/profile.d/conda.sh" ]; then
    printf 'source "%s/etc/profile.d/conda.sh" && conda activate "%s" && ' \
      "${base}" "${CONDA_ENV}"
  else
    # Fall back to whatever `conda` is on PATH (tmux inherits the launch env).
    printf 'conda activate "%s" && ' "${CONDA_ENV}"
  fi
}

build_cmd() {
  # Printed and executed inside the session; tee keeps a logfile while showing live.
  # Trailing `exec bash` keeps the pane alive after the command ends (even on a
  # crash) so you can attach and drag-scroll the logs instead of the pane closing.
  local activate; activate="$(conda_activate_snippet)"
  cat <<EOF
${activate}cd "${PROJECT_ROOT}" && \
python scripts/evaluate.py \
  --model_path "${MODEL_PATH}" \
  --raw_dataset_path "${RAW_DATASET_PATH}" \
  --classifier_path "${CLASSIFIER_PATH}" \
  --target_do "${TARGET_DO}" \
  --split "${SPLIT}" \
  --n_samples ${N_SAMPLES} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --batch_size ${BATCH_SIZE} \
  --output_file "${RESULT_FILE}" \
  2>&1 | tee "${LOG_FILE}"; \
echo; echo "=== eval finished (exit \${PIPESTATUS[0]}) — results: ${RESULT_FILE} ==="; \
echo "Drag up to scroll logs (q to exit scroll), Ctrl-b d to detach, or 'exit' to close."; \
exec bash
EOF
}

cmd_run() {
  if session_exists; then
    warn "session '${SESSION}' already running. Use 'attach' or 'stop' first."
    exit 1
  fi
  mkdir -p "${LOG_DIR}"
  : > "${LOG_FILE}"; ln -sf "${LOG_FILE}" "${LATEST_LINK}"

  local run_cmd; run_cmd="$(build_cmd)"
  # TMUX_CONF_PATH is exported so the in-session `prefix r` reload works.
  TMUX_CONF_PATH="${TMUX_CONF_PATH}" \
    "${TM[@]}" new-session -d -s "${SESSION}" -x 220 -y 50 "${run_cmd}"
  "${TM[@]}" set-environment -t "${SESSION}" TMUX_CONF_PATH "${TMUX_CONF_PATH}" || true

  ok   "started session '${SESSION}' (socket ${SOCKET})"
  info "conda=${CONDA_ENV:-<none>}  model=${MODEL_PATH}  do=${TARGET_DO}  n=${N_SAMPLES}  batch=${BATCH_SIZE}"
  info "logfile: ${LOG_FILE}"
  echo
  printf '  %sattach & watch:%s  %s\n' "${c_bold}" "${c_off}" "scripts/run_eval_tmux.sh attach"
  printf '  %stail logfile:%s    %s\n' "${c_bold}" "${c_off}" "scripts/run_eval_tmux.sh logs"
  printf '  %sstop:%s            %s\n' "${c_bold}" "${c_off}" "scripts/run_eval_tmux.sh stop"
  echo
  info "Inside the session: drag the mouse UPWARD to scroll back through logs."
}

cmd_attach() {
  session_exists || { warn "no session '${SESSION}'. Start it: scripts/run_eval_tmux.sh run"; exit 1; }
  if [[ -n "${TMUX:-}" ]]; then
    warn "you're already inside tmux; switching client..."
    "${TM[@]}" switch-client -t "${SESSION}" 2>/dev/null || \
      exec "${TM[@]}" attach -t "${SESSION}"
  else
    exec "${TM[@]}" attach -t "${SESSION}"
  fi
}

cmd_logs() {
  local f="${LATEST_LINK}"
  [[ -e "${f}" ]] || { warn "no log yet at ${f}"; exit 1; }
  info "tailing $(readlink -f "${f}")  (Ctrl-c to stop)"
  tail -n 40 -F "${f}"
}

cmd_status() {
  if session_exists; then
    ok "running:"; "${TM[@]}" list-sessions
    "${TM[@]}" list-panes -t "${SESSION}" -F \
      '  pane #{pane_index}: #{pane_current_command} (#{pane_width}x#{pane_height})' || true
  else
    warn "no session '${SESSION}' on socket ${SOCKET}."
  fi
}

cmd_stop() {
  session_exists || { warn "no session '${SESSION}' to stop."; exit 0; }
  "${TM[@]}" kill-session -t "${SESSION}"
  ok "killed session '${SESSION}'."
}

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

case "${1:-run}" in
  run)     cmd_run ;;
  attach|a) cmd_attach ;;
  logs|tail) cmd_logs ;;
  status|st) cmd_status ;;
  stop|kill) cmd_stop ;;
  help|-h|--help) usage ;;
  *) warn "unknown command: ${1}"; usage; exit 1 ;;
esac
