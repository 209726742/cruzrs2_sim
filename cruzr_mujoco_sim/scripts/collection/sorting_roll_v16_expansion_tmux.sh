#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PACKAGE_ROOT="$PROJECT_ROOT/cruzr_mujoco_sim"
COLLECTOR="$PACKAGE_ROOT/scripts/collection/sorting_roll_v16_expansion_collect.sh"
CAMPAIGN=${SORTING_ROLL_V16_EXPANSION_CAMPAIGN:-sorting_roll_v16_stage80_20260828}
OUTPUT_ROOT=${SORTING_ROLL_V16_EXPANSION_OUTPUT_ROOT:-$PACKAGE_ROOT/output/sorting_roll_expert/$CAMPAIGN}
MANIFEST=${SORTING_ROLL_V16_EXPANSION_MANIFEST:-$OUTPUT_ROOT/campaign_manifest.json}
GPU_CSV=${SORTING_ROLL_V16_GPUS:-0,1,2,3}
MODE=${2:-representative}
SESSION=${SORTING_ROLL_V16_TMUX_SESSION:-sorting_roll_v16_${CAMPAIGN}_${MODE}}
LOG_ROOT="$PROJECT_ROOT/cruzr_mujoco_sim/log"
LOG="$LOG_ROOT/${SESSION}.log"
ACTION=${1:-status}

case "$ACTION" in
  start)
    if [[ "$MODE" != "representative" && "$MODE" != "all" ]]; then
      echo "start mode must be representative or all" >&2
      exit 2
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux session already exists: $SESSION" >&2
      exit 1
    fi
    mkdir -p "$LOG_ROOT"
    tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" \
      "exec env SORTING_ROLL_V16_EXPANSION_CAMPAIGN='$CAMPAIGN' SORTING_ROLL_V16_EXPANSION_OUTPUT_ROOT='$OUTPUT_ROOT' SORTING_ROLL_V16_EXPANSION_MANIFEST='$MANIFEST' SORTING_ROLL_V16_GPUS='$GPU_CSV' bash '$COLLECTOR' '$MODE' >'$LOG' 2>&1"
    echo "started session=$SESSION log=$LOG"
    ;;
  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "session=$SESSION state=running"
    else
      echo "session=$SESSION state=stopped"
    fi
    if [[ -f "$LOG" ]]; then
      tail -n 30 "$LOG"
    fi
    ;;
  *)
    echo "usage: $0 {start|status} [representative|all]" >&2
    exit 2
    ;;
esac
