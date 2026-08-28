#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CONVERTER=$PROJECT_ROOT/cruzr_mujoco_sim/scripts/collection/cruzr_real_to_lerobot_v30.py
ISAAC_PY=${ISAAC_PY:-/isaac-sim/python.sh}
SESSION=${SESSION:-cruzr-real-pi05-convert}
OUTPUT=${OUTPUT:-$PROJECT_ROOT/cruzr_mujoco_sim/out/datasets/cruzr_real_clamp_23ep_lerobot_v30_20260828}
REPO_ID=${REPO_ID:-local/cruzr_real_clamp_23ep}
TASK=${TASK:-Clamp the target object.}
LOG_FILE=${LOG_FILE:-$PROJECT_ROOT/log/real_data_conversion/pi05_v30.log}
EXIT_FILE=$LOG_FILE.exit

run_conversion() {
  mkdir -p "$(dirname "$LOG_FILE")"
  printf '[%s] conversion start/resume output=%s repo_id=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OUTPUT" "$REPO_ID" >> "$LOG_FILE"
  set +e
  HF_HUB_OFFLINE=1 PYTHONPATH="$PROJECT_ROOT" "$ISAAC_PY" "$CONVERTER" convert \
    --output "$OUTPUT" \
    --repo-id "$REPO_ID" \
    --task "$TASK" >> "$LOG_FILE" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$EXIT_FILE"
  printf '[%s] conversion exited status=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$LOG_FILE"
  return "$status"
}

show_status() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    printf 'running: tmux session %s\n' "$SESSION"
  else
    printf 'not running: tmux session %s\n' "$SESSION"
  fi
  if [[ -f $EXIT_FILE ]]; then
    printf 'last exit status: %s\n' "$(<"$EXIT_FILE")"
  fi
  if [[ -f $OUTPUT/meta/conversion_progress.json ]]; then
    "$ISAAC_PY" - "$OUTPUT/meta/conversion_progress.json" <<'PY'
import json
from pathlib import Path
import sys

progress = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
episodes = progress.get("episodes", [])
frames = sum(item.get("frames", 0) for item in episodes)
print(
    f"progress: status={progress.get('status')} "
    f"episodes={len(episodes)}/{len(progress.get('source_order', []))} "
    f"frames={frames}"
)
PY
  fi
  if [[ -f $LOG_FILE ]]; then
    printf '%s\n' '--- log tail ---'
    tail -n 20 "$LOG_FILE"
  fi
}

case "${1:-status}" in
  run)
    run_conversion
    ;;
  start|resume)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      printf 'conversion is already running in tmux session %s\n' "$SESSION" >&2
      exit 1
    fi
    rm -f "$EXIT_FILE"
    tmux new-session -d -s "$SESSION" "bash '$0' run"
    printf 'started tmux session %s\n' "$SESSION"
    printf 'status: %s status\n' "$0"
    printf 'attach: tmux attach -t %s\n' "$SESSION"
    ;;
  status)
    show_status
    ;;
  audit)
    HF_HUB_OFFLINE=1 PYTHONPATH="$PROJECT_ROOT" "$ISAAC_PY" "$CONVERTER" audit \
      --output "$OUTPUT" \
      --repo-id "$REPO_ID"
    ;;
  *)
    printf 'usage: %s {start|resume|status|audit}\n' "$0" >&2
    exit 2
    ;;
esac
