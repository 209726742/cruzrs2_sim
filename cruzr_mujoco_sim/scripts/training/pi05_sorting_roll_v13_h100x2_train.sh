#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
V13_LAUNCHER=$SCRIPT_DIR/pi05_sorting_roll_v13_train.sh
H100_NUM_WORKERS=${H100_NUM_WORKERS:-8}

CANARY_OUTPUT=$SIM_ROOT/out/training/pi05_sorting_roll_v13_h100x2_canary_seed1000
CANARY_LOG=$PROJECT_ROOT/log/pi05_sorting_roll_v13_h100x2_canary_seed1000.log
FORMAL_OUTPUT_H100=$SIM_ROOT/out/training/pi05_sorting_roll_v13_h100x2_expert20k_seed1000
FORMAL_LOG_H100=$PROJECT_ROOT/log/pi05_sorting_roll_v13_h100x2_expert20k_seed1000.log

usage() {
  printf '%s\n' \
    'Usage:' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh canary-dry-run' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh canary' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh canary-resume' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh canary-status' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh dry-run' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh start' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh resume' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh status'
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

hardware_preflight() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  local rows
  rows=$(nvidia-smi -i 0,1 \
    --query-gpu=index,name,memory.total,mig.mode.current \
    --format=csv,noheader,nounits)
  python3 - "$rows" <<'PY_HARDWARE'
import sys

rows = [line.strip() for line in sys.argv[1].splitlines() if line.strip()]
assert len(rows) == 2, rows
for expected_index, row in enumerate(rows):
    index, name, memory, mig = (part.strip() for part in row.split(",", 3))
    assert int(index) == expected_index, row
    assert "H100" in name, row
    assert int(memory) >= 80000, row
    assert mig.lower() == "disabled", row
print("H100 preflight passed: 2 x >=80GB, MIG disabled")
PY_HARDWARE
}

set_common_h100_env() {
  export GPU_IDS=0,1
  export NUM_PROCESSES=2
  export BATCH_SIZE=16
  export NUM_WORKERS=$H100_NUM_WORKERS
}

set_canary_env() {
  set_common_h100_env
  export FORMAL_OUTPUT=$CANARY_OUTPUT
  export FORMAL_LOG=$CANARY_LOG
  export FORMAL_JOB_NAME=pi05_sorting_roll_v13_h100x2_canary
  export TARGET_STEPS=${1:-200}
  export WARMUP_STEPS=20
  export SAVE_FREQ=200
  export LOG_FREQ=10
  export MAIN_PROCESS_PORT=29525
}

set_formal_env() {
  set_common_h100_env
  export FORMAL_OUTPUT=$FORMAL_OUTPUT_H100
  export FORMAL_LOG=$FORMAL_LOG_H100
  export FORMAL_JOB_NAME=pi05_sorting_roll_v13_h100x2_expert20k
  export TARGET_STEPS=20000
  export WARMUP_STEPS=1000
  export SAVE_FREQ=1000
  export LOG_FREQ=10
  export MAIN_PROCESS_PORT=29526
}

action=${1:-dry-run}
[[ $# -eq 0 ]] || shift
[[ $# -eq 0 ]] || die "unexpected arguments: $*"

case "$action" in
  canary-dry-run)
    hardware_preflight
    set_canary_env 200
    exec bash "$V13_LAUNCHER" dry-run
    ;;
  canary)
    hardware_preflight
    set_canary_env 200
    exec bash "$V13_LAUNCHER" start
    ;;
  canary-resume)
    hardware_preflight
    set_canary_env 250
    exec bash "$V13_LAUNCHER" resume
    ;;
  canary-status)
    set_canary_env 200
    exec bash "$V13_LAUNCHER" status
    ;;
  dry-run)
    hardware_preflight
    set_formal_env
    exec bash "$V13_LAUNCHER" dry-run
    ;;
  start)
    hardware_preflight
    set_formal_env
    exec bash "$V13_LAUNCHER" start
    ;;
  resume)
    hardware_preflight
    set_formal_env
    exec bash "$V13_LAUNCHER" resume
    ;;
  status)
    set_formal_env
    exec bash "$V13_LAUNCHER" status
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    die "unknown action: $action"
    ;;
esac
