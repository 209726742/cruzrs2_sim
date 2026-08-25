#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)

DATASET_ROOT=$SIM_ROOT/out/datasets/sorting_roll_v13_diverse300_lerobot_v30_20260825
DATASET_AUDIT=$PROJECT_ROOT/log/sorting_roll_v13_diverse300_20260825_4gpu/dataset_v30_audit.json
CANARY_AUDIT=$PROJECT_ROOT/log/sorting_roll_v13_diverse300_20260825_4gpu/pi05_canary_audit.json
READINESS_REPORT=$SIM_ROOT/output/sorting_roll_expert/sorting_roll_v13_diverse300_20260825_4gpu/formal_training_readiness.json
BASE_POLICY=$PROJECT_ROOT/pretrained/pi05_base_remapped
FORMAL_OUTPUT=${FORMAL_OUTPUT:-$SIM_ROOT/out/training/pi05_sorting_roll_v13_formal10k_20260825}
FORMAL_LOG=${FORMAL_LOG:-$PROJECT_ROOT/log/pi05_sorting_roll_v13_formal10k_20260825.log}
FORMAL_JOB_NAME=${FORMAL_JOB_NAME:-pi05_sorting_roll_v13_formal10k}
ISAAC_PY=${ISAAC_PY:-/isaac-sim/python.sh}

GPU_IDS=${GPU_IDS:-0,1,2,3}
NUM_PROCESSES=${NUM_PROCESSES:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29514}
TARGET_STEPS=${TARGET_STEPS:-10000}
WARMUP_STEPS=${WARMUP_STEPS:-500}
SAVE_FREQ=${SAVE_FREQ:-500}
LOG_FREQ=${LOG_FREQ:-10}

usage() {
  printf '%s\n' \
    'Usage:' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_train.sh dry-run' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_train.sh start' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_train.sh resume' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_train.sh dry-run-resume' \
    '  bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_train.sh status' \
    '' \
    'The default is dry-run. start/resume are detached from SSH by pi05_train.sh.' \
    'This entry is pinned to the audited Sorting Roll v13 LeRobot v3.0 dataset.'
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

preflight() {
  [[ -x $ISAAC_PY ]] || die "Isaac Python is not executable: $ISAAC_PY"
  [[ -f $DATASET_ROOT/meta/info.json ]] || die "missing dataset info: $DATASET_ROOT/meta/info.json"
  [[ -f $DATASET_AUDIT ]] || die "missing dataset audit: $DATASET_AUDIT"
  [[ -f $CANARY_AUDIT ]] || die "missing canary audit: $CANARY_AUDIT"
  [[ -f $READINESS_REPORT ]] || die "missing readiness report: $READINESS_REPORT"

  "$ISAAC_PY" - "$DATASET_ROOT/meta/info.json" "$DATASET_AUDIT" \
    "$CANARY_AUDIT" "$READINESS_REPORT" <<'PY'
import json
import sys
from pathlib import Path

info, dataset_audit, canary_audit, readiness = (
    json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]
)

expected_cameras = {
    "observation.images.stereo_left",
    "observation.images.left_wrist_realsense",
    "observation.images.right_wrist_realsense",
}
assert info["codebase_version"] == "v3.0"
assert info["source_task_version"] == "sorting_roll_v13_diverse_sim"
assert info["source_campaign"] == "sorting_roll_v13_diverse300_20260825_4gpu"
assert info["collection_profile"] == "sorting_roll_d405_candidate_v4"
assert info["total_episodes"] == info["total_source_episodes"] == 300
assert info["total_frames"] == 519776
assert info["splits"] == {"train": "0:240", "val": "240:270", "test": "270:300"}

features = info["features"]
cameras = {key for key, value in features.items() if value.get("dtype") == "video"}
assert cameras == expected_cameras, cameras
for camera in expected_cameras:
    feature = features[camera]
    assert feature["shape"] == [224, 224, 3]
    assert feature["info"]["video.fps"] == 30
    assert feature["info"]["video.codec"] == "h264"
    assert feature["info"]["video.pix_fmt"] == "yuv420p"
for key in ("observation.state", "action"):
    assert features[key]["dtype"] == "float32"
    assert features[key]["shape"] == [18]

assert dataset_audit["passed"] is True and dataset_audit["errors"] == []
assert dataset_audit["episodes"] == 300 and dataset_audit["frames"] == 519776
assert dataset_audit["sampled_episode_count"] == 8
assert dataset_audit["decoded_sample_count"] == 24
assert canary_audit["passed"] is True and canary_audit["errors"] == []
assert canary_audit["fresh_and_resume_exit_zero"] is True
assert canary_audit["fresh_target_step"] == 20
assert canary_audit["resume_target_step"] == 40
assert readiness["ready_for_formal_training"] is True
assert all(readiness["checks"].values())
print("v13 preflight passed: dataset/schema/audit/canary/readiness")
PY
}

common_args=(
  --dataset-root "$DATASET_ROOT"
  --repo-id local/sorting_roll_v13_diverse300
  --episodes train
  --base-policy "$BASE_POLICY"
  --output-dir "$FORMAL_OUTPUT"
  --job-name "$FORMAL_JOB_NAME"
  --log-file "$FORMAL_LOG"
  --gpu-ids "$GPU_IDS"
  --num-processes "$NUM_PROCESSES"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --port "$MAIN_PROCESS_PORT"
  --steps "$TARGET_STEPS"
  --save-freq "$SAVE_FREQ"
  --log-freq "$LOG_FREQ"
  --dtype bfloat16
  --gradient-checkpointing true
  --train-expert-only true
  --allow-small-batch true
  --learning-rate 2.5e-5
  --weight-decay 0.01
  --grad-clip-norm 1.0
  --warmup-steps "$WARMUP_STEPS"
  --decay-steps "$TARGET_STEPS"
  --image-transforms false
  --use-imagenet-stats true
  --wandb false
  --wandb-mode offline
  --offline true
  --isaac-python "$ISAAC_PY"
)

action=${1:-dry-run}
[[ $# -eq 0 ]] || shift
[[ $# -eq 0 ]] || die "unexpected arguments: $*"

case "$action" in
  dry-run|start)
    preflight
    exec bash "$PROJECT_ROOT/pi05_train.sh" "$action" "${common_args[@]}"
    ;;
  resume|dry-run-resume)
    preflight
    exec bash "$PROJECT_ROOT/pi05_train.sh" "$action" "${common_args[@]}"
    ;;
  status)
    exec bash "$PROJECT_ROOT/pi05_train.sh" status \
      --output-dir "$FORMAL_OUTPUT" --job-name "$FORMAL_JOB_NAME" \
      --log-file "$FORMAL_LOG" --isaac-python "$ISAAC_PY"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    die "unknown action: $action"
    ;;
esac
