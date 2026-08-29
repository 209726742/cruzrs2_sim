#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
BASE_LAUNCHER=$PROJECT_ROOT/pi05_train.sh
ISAAC_PY=${ISAAC_PY:-/isaac-sim/python.sh}

CAMPAIGN=sorting_roll_v16_stage80_mixed320_20260829
DATASET_ROOT=$SIM_ROOT/out/datasets/${CAMPAIGN}_lerobot_v30
LOG_ROOT=$PROJECT_ROOT/log/$CAMPAIGN
DATASET_AUDIT=$LOG_ROOT/dataset_v30_audit.json
DATA_READINESS=$LOG_ROOT/data_training_readiness.json
SAMPLING_REPORT=$LOG_ROOT/sampling_weights_old50_h15_t15_r15_c5.json
SAMPLING_WEIGHTS=$DATASET_ROOT/meta/sampling_weights_old50_h15_t15_r15_c5.npy
BASE_POLICY=$SIM_ROOT/out/training/pi05_sorting_roll_v15_h100x4_fullft28k_seed1000/checkpoints/036000/pretrained_model
REPO_ID=local/$CAMPAIGN

CANARY_OUTPUT=$SIM_ROOT/out/training/pi05_sorting_roll_v16_h100x4_fullft_canary_seed1000
CANARY_LOG=$PROJECT_ROOT/log/pi05_sorting_roll_v16_h100x4_fullft_canary_seed1000.log
CANARY_AUDIT=$LOG_ROOT/pi05_h100x4_fullft_canary_audit.json
FORMAL_OUTPUT=${FORMAL_OUTPUT:-$SIM_ROOT/out/training/pi05_sorting_roll_v16_h100x4_fullft28k_seed1000}
FORMAL_LOG=${FORMAL_LOG:-$PROJECT_ROOT/log/pi05_sorting_roll_v16_h100x4_fullft28k_seed1000.log}

GPU_IDS=0,1,2,3
NUM_PROCESSES=4
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-8}
TARGET_STEPS=${TARGET_STEPS:-28000}
SAVE_FREQ=${SAVE_FREQ:-1000}
LOG_FREQ=${LOG_FREQ:-10}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
SEED=1000

usage() {
  printf '%s\n' \
    'Usage:' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v16_h100x4_fullft20h.sh data-check' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v16_h100x4_fullft20h.sh config-dry-run' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v16_h100x4_fullft20h.sh hardware-check' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v16_h100x4_fullft20h.sh canary-dry-run' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v16_h100x4_fullft20h.sh tmux-canary' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v16_h100x4_fullft20h.sh tmux-canary-resume' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v16_h100x4_fullft20h.sh canary-audit' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v16_h100x4_fullft20h.sh dry-run' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v16_h100x4_fullft20h.sh tmux-start' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v16_h100x4_fullft20h.sh tmux-resume' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v16_h100x4_fullft20h.sh status'
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

hardware_preflight() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  local rows
  rows=$(nvidia-smi -i "$GPU_IDS" \
    --query-gpu=index,name,memory.total,mig.mode.current \
    --format=csv,noheader,nounits)
  "$PROJECT_ROOT/envs/mjx/bin/python" - "$rows" <<'PY'
import sys

rows = [line.strip() for line in sys.argv[1].splitlines() if line.strip()]
assert len(rows) == 4, rows
for expected, row in enumerate(rows):
    index, name, memory, mig = (
        part.strip() for part in row.split(",", 3)
    )
    assert int(index) == expected, row
    assert "H100" in name, row
    assert int(memory) >= 80000, row
    assert mig.lower() == "disabled", row
print("hardware gate passed: 4xH100 >=80GB, MIG disabled")
PY
}

data_preflight() {
  [[ -x "$ISAAC_PY" ]] || die "Isaac Python is unavailable: $ISAAC_PY"
  [[ -f "$DATASET_ROOT/meta/info.json" ]] || die "dataset is not ready"
  [[ -f "$DATASET_AUDIT" ]] || die "dataset audit is missing"
  [[ -f "$DATA_READINESS" ]] || die "data readiness report is missing"
  [[ -f "$SAMPLING_REPORT" ]] || die "sampling audit is missing"
  [[ -f "$SAMPLING_WEIGHTS" ]] || die "sampling weights are missing"
  [[ -d "$BASE_POLICY" ]] || die "36k base policy is missing"
  "$ISAAC_PY" - "$DATASET_ROOT/meta/info.json" "$DATASET_AUDIT" \
    "$DATA_READINESS" "$SAMPLING_REPORT" <<'PY'
import json
from pathlib import Path
import sys

info, audit, readiness, sampling = (
    json.loads(Path(path).read_text(encoding="utf-8"))
    for path in sys.argv[1:]
)
expected_cameras = {
    "observation.images.stereo_left",
    "observation.images.left_wrist_realsense",
    "observation.images.right_wrist_realsense",
}
assert info["codebase_version"] == "v3.0"
assert info["source_task_version"] == "mixed"
assert info["total_episodes"] == info["total_source_episodes"] == 320
assert info["splits"] == {"train": "0:304", "val": "304:312", "test": "312:320"}
features = info["features"]
actual_cameras = {
    key for key, value in features.items() if value.get("dtype") == "video"
}
assert actual_cameras == expected_cameras, actual_cameras
for camera in expected_cameras:
    assert features[camera]["shape"] == [224, 224, 3]
    assert features[camera]["info"]["video.fps"] == 30
for key in ("observation.state", "action"):
    assert features[key]["dtype"] == "float32"
    assert features[key]["shape"] == [18]
assert audit["passed"] is True and audit["errors"] == []
assert audit["v16_family_counts"] == {"C": 12, "H": 20, "R": 28, "T": 20}
assert readiness["ready_for_full_parameter_canary"] is True
assert readiness["episodes"] == 320
assert sampling["passed"] is True
assert sampling["sampling_profile"] == "stage80_old50"
assert sampling["episodes"] == "0:304"
assert sampling["target_fractions"] == {
    "old": 0.50,
    "H": 0.15,
    "T": 0.15,
    "R": 0.15,
    "C": 0.05,
}
print("v16 data gate passed: 320 episodes, H/T/R/C=20/20/28/12, 3 cameras, 30 FPS, 18D")
PY
}

set_train_args() {
  local output=$1
  local log=$2
  local job=$3
  local steps=$4
  local warmup=$5
  local decay=$6
  local save_freq=$7
  local port=$8
  TRAIN_ARGS=(
    --dataset-root "$DATASET_ROOT"
    --repo-id "$REPO_ID"
    --episodes train
    --frame-sampling-weights "$SAMPLING_WEIGHTS"
    --use-pretrained-stats true
    --base-policy "$BASE_POLICY"
    --output-dir "$output"
    --job-name "$job"
    --log-file "$log"
    --gpu-ids "$GPU_IDS"
    --num-processes "$NUM_PROCESSES"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --port "$port"
    --steps "$steps"
    --save-freq "$save_freq"
    --log-freq "$LOG_FREQ"
    --seed "$SEED"
    --dtype bfloat16
    --gradient-checkpointing true
    --train-expert-only false
    --min-effective-batch 64
    --learning-rate "$LEARNING_RATE"
    --weight-decay 0.01
    --grad-clip-norm 1.0
    --warmup-steps "$warmup"
    --decay-steps "$decay"
    --n-action-steps 50
    --num-inference-steps 10
    --video-backend pyav
    --image-transforms false
    --use-imagenet-stats true
    --wandb false
    --wandb-mode offline
    --offline true
    --isaac-python "$ISAAC_PY"
  )
}

canary_args() {
  local steps=$1
  local save_freq=$2
  set_train_args \
    "$CANARY_OUTPUT" "$CANARY_LOG" \
    pi05_sorting_roll_v16_h100x4_fullft_canary \
    "$steps" 20 200 "$save_freq" 29545
}

formal_args() {
  set_train_args \
    "$FORMAL_OUTPUT" "$FORMAL_LOG" \
    pi05_sorting_roll_v16_h100x4_fullft28k \
    "$TARGET_STEPS" "$WARMUP_STEPS" "$TARGET_STEPS" "$SAVE_FREQ" 29546
}

formal_preflight() {
  [[ -f "$CANARY_AUDIT" ]] || die "run canary-audit before formal training"
  "$PROJECT_ROOT/envs/mjx/bin/python" - "$CANARY_AUDIT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["passed"] is True and report["errors"] == []
assert report["task_version"] == "sorting_roll_v16_expansion_stage_sim"
assert report["fresh_and_resume_exit_zero"] is True
assert report["full_parameter_count_verified"] is True
assert report["train_expert_only"] is False
assert report["effective_batch_size"] == 64
assert report["learning_rate"] == 1e-5
print("v16 full-parameter canary gate passed")
PY
}

launch_tmux() {
  local session=$1
  local action=$2
  local log=$3
  tmux has-session -t "$session" 2>/dev/null && die "tmux session exists: $session"
  tmux new-session -d -s "$session" \
    "cd '$PROJECT_ROOT' && bash '$0' '$action'; pid=\$(cat '$log.pid'); tail --pid=\$pid -F '$log'"
  printf 'tmux session: %s\n' "$session"
}

action=${1:-config-dry-run}
[[ $# -eq 0 ]] || shift
[[ $# -eq 0 ]] || die "unexpected arguments: $*"

case "$action" in
  data-check)
    data_preflight
    ;;
  config-dry-run)
    data_preflight
    formal_args
    exec bash "$BASE_LAUNCHER" dry-run "${TRAIN_ARGS[@]}"
    ;;
  hardware-check)
    hardware_preflight
    ;;
  canary-dry-run)
    hardware_preflight
    data_preflight
    canary_args 200 200
    exec bash "$BASE_LAUNCHER" dry-run "${TRAIN_ARGS[@]}"
    ;;
  canary)
    hardware_preflight
    data_preflight
    canary_args 200 200
    exec bash "$BASE_LAUNCHER" start "${TRAIN_ARGS[@]}"
    ;;
  canary-resume)
    hardware_preflight
    data_preflight
    canary_args 250 50
    exec bash "$BASE_LAUNCHER" resume "${TRAIN_ARGS[@]}"
    ;;
  tmux-canary)
    launch_tmux sorting_roll_v16_h100x4_fullft_canary canary "$CANARY_LOG"
    ;;
  tmux-canary-resume)
    launch_tmux sorting_roll_v16_h100x4_fullft_canary_resume canary-resume "$CANARY_LOG"
    ;;
  canary-audit)
    data_preflight
    exec "$ISAAC_PY" \
      "$SCRIPT_DIR/sorting_roll_v15_fullft_canary_audit.py" \
      --output "$CANARY_OUTPUT" \
      --log "$CANARY_LOG" \
      --dataset "$DATASET_ROOT" \
      --report "$CANARY_AUDIT" \
      --task-version sorting_roll_v16_expansion_stage_sim \
      --expected-learning-rate "$LEARNING_RATE"
    ;;
  dry-run)
    hardware_preflight
    data_preflight
    formal_preflight
    formal_args
    exec bash "$BASE_LAUNCHER" dry-run "${TRAIN_ARGS[@]}"
    ;;
  start)
    hardware_preflight
    data_preflight
    formal_preflight
    formal_args
    exec bash "$BASE_LAUNCHER" start "${TRAIN_ARGS[@]}"
    ;;
  resume)
    hardware_preflight
    data_preflight
    formal_preflight
    formal_args
    exec bash "$BASE_LAUNCHER" resume "${TRAIN_ARGS[@]}"
    ;;
  tmux-start)
    launch_tmux sorting_roll_v16_h100x4_fullft28k start "$FORMAL_LOG"
    ;;
  tmux-resume)
    launch_tmux sorting_roll_v16_h100x4_fullft28k_resume resume "$FORMAL_LOG"
    ;;
  status)
    formal_args
    bash "$BASE_LAUNCHER" status \
      --output-dir "$FORMAL_OUTPUT" \
      --job-name pi05_sorting_roll_v16_h100x4_fullft28k \
      --log-file "$FORMAL_LOG" \
      --isaac-python "$ISAAC_PY"
    tmux list-sessions 2>/dev/null || true
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    die "unknown action: $action"
    ;;
esac
