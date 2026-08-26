#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
LAUNCHER=$SCRIPT_DIR/pi05_sorting_roll_v15_h100x4_fullft20h.sh
SESSION=sorting_roll_v15_h100x4_auto20h
AUTO_LOG=$PROJECT_ROOT/log/pi05_sorting_roll_v15_h100x4_auto20h.log
CANARY_LOG=$PROJECT_ROOT/log/pi05_sorting_roll_v15_h100x4_fullft_canary_seed1000.log
FORMAL_LOG=$PROJECT_ROOT/log/pi05_sorting_roll_v15_h100x4_fullft28k_seed1000.log
POLL_SECONDS=${POLL_SECONDS:-30}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

wait_for_job() {
  local label=$1
  local training_log=$2
  local pid_file=${training_log}.pid
  local pid rc

  [[ -s "$pid_file" ]] || {
    log "$label did not create pid file: $pid_file"
    return 1
  }
  pid=$(<"$pid_file")
  [[ "$pid" =~ ^[0-9]+$ ]] || {
    log "$label wrote an invalid pid: $pid"
    return 1
  }

  log "$label running with supervisor pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done

  rc=$(sed -n 's/.*train command exited rc=\([0-9][0-9]*\).*/\1/p' \
    "$training_log" | tail -n 1)
  [[ "$rc" == 0 ]] || {
    log "$label failed or has no clean exit marker; rc=${rc:-missing}"
    tail -n 40 "$training_log" || true
    return 1
  }
  log "$label completed successfully"
}

run_pipeline() {
  log "checking 4xH100 hardware and v15 data"
  bash "$LAUNCHER" hardware-check
  bash "$LAUNCHER" canary-dry-run

  log "starting fresh 200-step full-parameter canary"
  bash "$LAUNCHER" canary
  wait_for_job "fresh canary" "$CANARY_LOG"

  log "resuming full-parameter canary to step 250"
  bash "$LAUNCHER" canary-resume
  wait_for_job "resumed canary" "$CANARY_LOG"

  log "auditing parameters, loss, gradients, checkpoints, and resume"
  bash "$LAUNCHER" canary-audit
  bash "$LAUNCHER" recommend-20h

  log "checking the 28k formal configuration"
  bash "$LAUNCHER" dry-run

  log "starting 28k full-parameter formal training"
  bash "$LAUNCHER" start
  wait_for_job "formal training" "$FORMAL_LOG"
  log "v15 4xH100 full-parameter training completed"
}

start_tmux() {
  command -v tmux >/dev/null || {
    log "tmux is unavailable"
    return 1
  }
  tmux has-session -t "$SESSION" 2>/dev/null && {
    log "tmux session already exists: $SESSION"
    return 1
  }
  mkdir -p "$(dirname "$AUTO_LOG")"
  if [[ -e "$AUTO_LOG" ]]; then
    mv "$AUTO_LOG" "$AUTO_LOG.previous_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  tmux new-session -d -s "$SESSION" \
    "cd '$PROJECT_ROOT' && bash '$0' run >> '$AUTO_LOG' 2>&1"
  sleep 2
  tmux has-session -t "$SESSION" 2>/dev/null || {
    tail -n 60 "$AUTO_LOG" >&2 || true
    log "tmux pipeline exited during startup"
    return 1
  }
  log "tmux pipeline started: $SESSION"
  log "log: $AUTO_LOG"
}

show_status() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    printf '总控 tmux：运行中（%s）\n' "$SESSION"
  else
    printf '总控 tmux：未运行\n'
  fi
  [[ -f "$AUTO_LOG" ]] && tail -n 30 "$AUTO_LOG"
  bash "$LAUNCHER" status
}

case "${1:-start}" in
  start)
    start_tmux
    ;;
  run)
    run_pipeline
    ;;
  status)
    show_status
    ;;
  resume-formal)
    bash "$LAUNCHER" tmux-resume
    ;;
  *)
    printf 'usage: %s {start|status|resume-formal}\n' "$0" >&2
    exit 2
    ;;
esac

