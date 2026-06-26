#!/usr/bin/env bash
# mem_logger.sh — poll a process's peak resident set (VmHWM) → JSONL.
#
# llama-bench reports speed but not memory; peak RSS is one of the 6 required axes
# (brief §9). VmHWM in /proc/PID/status is the kernel's high-water mark, so a single
# read after the run already gives the peak — but we sample so a thermal/throttle
# dip in throughput can be lined up against memory over time.
#
# Usage:
#   ./mem_logger.sh <PID> [interval_sec] [out.jsonl]
#   ./mem_logger.sh "$(pgrep -f llama-server | head -1)" 0.5 ../bench/logs/rss.jsonl
set -euo pipefail

PID="${1:?usage: mem_logger.sh <PID> [interval_sec] [out.jsonl]}"
INTERVAL="${2:-0.5}"
OUT="${3:-/dev/stdout}"

[[ -d "/proc/$PID" ]] || { echo "error: no process $PID" >&2; exit 1; }
[[ "$OUT" != "/dev/stdout" ]] && mkdir -p "$(dirname "$OUT")"

# epoch-seconds counter (date math kept simple; relative t is what matters for curves)
t0=$(date +%s 2>/dev/null || echo 0)
peak_kb=0
echo "# polling VmHWM of PID $PID every ${INTERVAL}s → $OUT (Ctrl-C to stop)" >&2

while [[ -d "/proc/$PID" ]]; do
  # VmHWM = peak RSS; VmRSS = current RSS (both kB)
  hwm=$(awk '/^VmHWM:/{print $2}' "/proc/$PID/status" 2>/dev/null || echo "")
  rss=$(awk '/^VmRSS:/{print $2}' "/proc/$PID/status" 2>/dev/null || echo "")
  [[ -z "$hwm" ]] && break
  (( hwm > peak_kb )) && peak_kb=$hwm
  now=$(date +%s 2>/dev/null || echo 0)
  printf '{"t_s": %d, "vm_hwm_mb": %.1f, "vm_rss_mb": %.1f}\n' \
    "$(( now - t0 ))" "$(awk "BEGIN{print $hwm/1024}")" "$(awk "BEGIN{print ${rss:-0}/1024}")" >> "$OUT"
  sleep "$INTERVAL"
done

echo "# peak_rss_mb = $(awk "BEGIN{print $peak_kb/1024}")" >&2
printf '{"peak_rss_mb": %.1f, "pid": %s}\n' "$(awk "BEGIN{print $peak_kb/1024}")" "$PID"
