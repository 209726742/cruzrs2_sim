#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)

DATASET_ROOT=${DATASET_ROOT:-$SIM_ROOT/out/datasets/formal300_v24_lerobot_v30_20260817}
BASE_POLICY=${BASE_POLICY:-$PROJECT_ROOT/pretrained/pi05_base_remapped}
CANARY_OUTPUT=${CANARY_OUTPUT:-$SIM_ROOT/out/training/pi05_formal300_v24_ddp_canary_20260817}
FORMAL_OUTPUT=${FORMAL_OUTPUT:-$SIM_ROOT/out/training/pi05_formal300_v24_10k_20260817}
LOG_ROOT=${LOG_ROOT:-$PROJECT_ROOT/log}
ISAAC_PY=${ISAAC_PY:-/isaac-sim/python.sh}

GPU_IDS=${GPU_IDS:-0,1,2,3}
NUM_PROCESSES=${NUM_PROCESSES:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}
TARGET_STEPS=${TARGET_STEPS:-10000}
SAVE_FREQ=${SAVE_FREQ:-500}
CANARY_STEPS=${CANARY_STEPS:-20}
CANARY_RESUME_STEPS=${CANARY_RESUME_STEPS:-40}
SEED=${SEED:-1000}
REPO_ID=${REPO_ID:-formal/cruzr_shelf_v24_300source}
FORMAL_JOB_NAME=${FORMAL_JOB_NAME:-pi05_formal300_v24_10k}
CANARY_JOB_NAME=${CANARY_JOB_NAME:-pi05_formal300_v24_ddp_canary}

FORMAL_LOG=$LOG_ROOT/pi05_formal300_v24_10k.log
CANARY_LOG=$LOG_ROOT/pi05_formal300_v24_ddp_canary.log
FORMAL_PID_FILE=$LOG_ROOT/pi05_formal300_v24_10k.pid
CANARY_PID_FILE=$LOG_ROOT/pi05_formal300_v24_ddp_canary.pid

usage() {
  cat <<'EOF'
Usage:
  bash scripts/training/pi05_formal300_train.sh canary
  bash scripts/training/pi05_formal300_train.sh canary-resume
  bash scripts/training/pi05_formal300_train.sh start
  bash scripts/training/pi05_formal300_train.sh resume
  bash scripts/training/pi05_formal300_train.sh status

Actions:
  canary         Start a fresh detached DDP canary (default: 20 steps).
  canary-resume  Resume the latest complete canary checkpoint (default: to step 40).
  start          Start a fresh detached formal run from pi05_base_remapped.
  resume         Resume the latest complete formal checkpoint. If no complete
                 checkpoint exists after a reboot, archive the partial output
                 and restart the formal run from the base policy.
  status         Show active processes, latest valid checkpoints, and log tails.

Hardware overrides (defaults target the current 4-GPU host):
  GPU_IDS=0,1,2,3 NUM_PROCESSES=4 BATCH_SIZE=1 NUM_WORKERS=2

The GPU model is not hard-coded. The same command can use four H100s or other
CUDA GPUs by setting GPU_IDS and the per-GPU batch/worker values after a canary.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing directory: $1"
}

validate_hardware_args() {
  local -a ids
  IFS=',' read -r -a ids <<< "$GPU_IDS"
  [[ ${#ids[@]} -eq $NUM_PROCESSES ]] ||
    die "GPU_IDS has ${#ids[@]} entries but NUM_PROCESSES=$NUM_PROCESSES"
  [[ $NUM_PROCESSES -gt 0 ]] || die "NUM_PROCESSES must be positive"
  [[ $BATCH_SIZE -gt 0 ]] || die "BATCH_SIZE must be positive"
  [[ $NUM_WORKERS -ge 0 ]] || die "NUM_WORKERS cannot be negative"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  local available
  available=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',')
  local id
  for id in "${ids[@]}"; do
    [[ $id =~ ^[0-9]+$ ]] || die "invalid GPU id: $id"
    [[ ,$available == *,$id,* ]] || die "GPU id $id is not visible (visible: $available)"
  done
}

validate_dataset() {
  require_dir "$DATASET_ROOT"
  require_file "$DATASET_ROOT/meta/info.json"
  require_file "$DATASET_ROOT/meta/stats.json"
  "$ISAAC_PY" -c '
import json, sys
info = json.load(open(sys.argv[1]))
assert info.get("total_source_episodes") == 300, info.get("total_source_episodes")
assert info.get("source_task_version") == "dual_two_trip_v1"
assert info.get("collection_profile") == "sdk_recovery_v1"
train = info.get("splits", {}).get("train")
assert isinstance(train, str) and train.startswith("0:"), train
' "$DATASET_ROOT/meta/info.json" || die "dataset readiness check failed"
}

train_episodes_json() {
  "$ISAAC_PY" -c '
import json, sys
info = json.load(open(sys.argv[1]))
start, stop = map(int, info["splits"]["train"].split(":"))
assert start == 0 and stop > start
print(json.dumps(list(range(start, stop)), separators=(",", ":")))
' "$DATASET_ROOT/meta/info.json"
}

checkpoint_is_complete() {
  local checkpoint=$1
  local expected_step=$2
  "$ISAAC_PY" -c '
import json, os, sys
from safetensors import safe_open
root, expected = sys.argv[1], int(sys.argv[2])
required_json = [
    "pretrained_model/config.json",
    "pretrained_model/policy_preprocessor.json",
    "pretrained_model/policy_postprocessor.json",
    "pretrained_model/train_config.json",
    "training_state/optimizer_param_groups.json",
    "training_state/scheduler_state.json",
    "training_state/training_step.json",
]
for rel in required_json:
    with open(os.path.join(root, rel)) as handle:
        json.load(handle)
step = json.load(open(os.path.join(root, "training_state/training_step.json")))["step"]
assert step == expected, (step, expected)
for rel in (
    "pretrained_model/model.safetensors",
    "training_state/optimizer_state.safetensors",
    "training_state/rng_state.safetensors",
):
    with safe_open(os.path.join(root, rel), framework="pt", device="cpu") as handle:
        assert len(list(handle.keys())) > 0, rel
' "$checkpoint" "$expected_step" >/dev/null 2>&1
}

latest_valid_checkpoint() {
  local output=$1
  local latest=''
  local checkpoint name step
  shopt -s nullglob
  for checkpoint in "$output"/checkpoints/[0-9][0-9][0-9][0-9][0-9][0-9]; do
    [[ -d "$checkpoint" ]] || continue
    name=${checkpoint##*/}
    step=$((10#$name))
    if checkpoint_is_complete "$checkpoint" "$step"; then
      latest=$checkpoint
    fi
  done
  shopt -u nullglob
  [[ -n $latest ]] && printf '%s\n' "$latest"
}

checkpoint_step() {
  local name=${1##*/}
  printf '%d\n' "$((10#$name))"
}

load_checkpoint_loader_args() {
  local config=$1/pretrained_model/train_config.json
  local values
  values=$("$ISAAC_PY" -c '
import json, sys
cfg = json.load(open(sys.argv[1]))
print(int(cfg["batch_size"]), int(cfg["num_workers"]))
' "$config")
  read -r BATCH_SIZE NUM_WORKERS <<< "$values"
  printf 'restored loader config: batch_per_gpu=%s workers_per_gpu=%s\n' "$BATCH_SIZE" "$NUM_WORKERS"
}

archive_checkpoints_after() {
  local output=$1
  local keep_step=$2
  local archive_root=''
  local checkpoint name step
  shopt -s nullglob
  for checkpoint in "$output"/checkpoints/[0-9][0-9][0-9][0-9][0-9][0-9]; do
    [[ -d $checkpoint ]] || continue
    name=${checkpoint##*/}
    step=$((10#$name))
    if (( step > keep_step )); then
      if [[ -z $archive_root ]]; then
        archive_root="$output/interrupted_checkpoints_$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$archive_root"
      fi
      mv "$checkpoint" "$archive_root/"
    fi
  done
  shopt -u nullglob
  if [[ -n $archive_root ]]; then
    printf 'archived checkpoints newer than step %s to %s\n' "$keep_step" "$archive_root"
  fi
}

job_is_running() {
  local job_name=$1
  pgrep -f "[l]erobot.scripts.lerobot_train.*$job_name" >/dev/null
}

accelerate_prefix() {
  ACCELERATE_PREFIX=(
    "$ISAAC_PY" -m accelerate.commands.launch
    --multi_gpu
    --num_processes "$NUM_PROCESSES"
    --gpu_ids "$GPU_IDS"
    --main_process_port "$MAIN_PROCESS_PORT"
    -m lerobot.scripts.lerobot_train
  )
}

fresh_command() {
  local output=$1
  local job_name=$2
  local steps=$3
  local save_freq=$4
  local episodes
  episodes=$(train_episodes_json)
  accelerate_prefix
  TRAIN_COMMAND=(
    "${ACCELERATE_PREFIX[@]}"
    --dataset.repo_id="$REPO_ID"
    --dataset.root="$DATASET_ROOT"
    --dataset.episodes="$episodes"
    --dataset.video_backend=pyav
    --dataset.image_transforms.enable=false
    --policy.path="$BASE_POLICY"
    --policy.device=cuda
    --policy.dtype=bfloat16
    --policy.gradient_checkpointing=true
    --policy.train_expert_only=true
    --policy.push_to_hub=false
    --output_dir="$output"
    --job_name="$job_name"
    --resume=false
    --batch_size="$BATCH_SIZE"
    --num_workers="$NUM_WORKERS"
    --steps="$steps"
    --save_checkpoint=true
    --save_freq="$save_freq"
    --eval_freq=0
    --log_freq=10
    --seed="$SEED"
    --wandb.enable=false
  )
}

resume_command() {
  local checkpoint=$1
  local output=$2
  local job_name=$3
  local steps=$4
  local save_freq=$5
  accelerate_prefix
  TRAIN_COMMAND=(
    "${ACCELERATE_PREFIX[@]}"
    --config_path="$checkpoint/pretrained_model/train_config.json"
    --resume=true
    --output_dir="$output"
    --job_name="$job_name"
    --steps="$steps"
    --save_freq="$save_freq"
    --eval_freq=0
    --log_freq=10
    --wandb.enable=false
  )
}

launch_detached() {
  local log_path=$1
  local pid_file=$2
  local mode=$3
  install -d "$LOG_ROOT"
  if [[ $mode == fresh && -e $log_path ]]; then
    mv "$log_path" "$log_path.previous_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  printf '[%s] launch mode=%s gpu_ids=%s processes=%s batch_per_gpu=%s workers_per_gpu=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$mode" "$GPU_IDS" "$NUM_PROCESSES" \
    "$BATCH_SIZE" "$NUM_WORKERS" >> "$log_path"
  cd "$PROJECT_ROOT"
  nohup setsid bash -c '
"$@"
rc=$?
printf "[%s] train command exited rc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc"
exit "$rc"
' _ env \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    "${TRAIN_COMMAND[@]}" >> "$log_path" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$pid_file"
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    printf 'training command exited immediately; log tail:\n' >&2
    tail -n 20 "$log_path" >&2
    return 1
  fi
  printf 'launched pid=%s\nlog=%s\n' "$pid" "$log_path"
}

start_canary() {
  validate_dataset
  validate_hardware_args
  require_dir "$BASE_POLICY"
  job_is_running "$CANARY_JOB_NAME" && die "canary is already running"
  local latest
  latest=$(latest_valid_checkpoint "$CANARY_OUTPUT" || true)
  if [[ -n $latest ]] && [[ $(checkpoint_step "$latest") -ge $CANARY_STEPS ]]; then
    printf 'canary already complete at %s\n' "$latest"
    return
  fi
  [[ ! -e $CANARY_OUTPUT ]] || die "canary output exists without a complete target checkpoint: $CANARY_OUTPUT"
  fresh_command "$CANARY_OUTPUT" "$CANARY_JOB_NAME" "$CANARY_STEPS" "$CANARY_STEPS"
  launch_detached "$CANARY_LOG" "$CANARY_PID_FILE" fresh
}

resume_canary() {
  validate_dataset
  job_is_running "$CANARY_JOB_NAME" && die "canary is already running"
  local latest step
  latest=$(latest_valid_checkpoint "$CANARY_OUTPUT" || true)
  [[ -n $latest ]] || die "no complete canary checkpoint found"
  step=$(checkpoint_step "$latest")
  if [[ $step -ge $CANARY_RESUME_STEPS ]]; then
    printf 'canary resume target already reached: step=%s\n' "$step"
    return
  fi
  load_checkpoint_loader_args "$latest"
  validate_hardware_args
  archive_checkpoints_after "$CANARY_OUTPUT" "$step"
  resume_command "$latest" "$CANARY_OUTPUT" "$CANARY_JOB_NAME" "$CANARY_RESUME_STEPS" "$CANARY_STEPS"
  launch_detached "$CANARY_LOG" "$CANARY_PID_FILE" resume
}

start_formal() {
  validate_dataset
  validate_hardware_args
  require_dir "$BASE_POLICY"
  job_is_running "$FORMAL_JOB_NAME" && die "formal training is already running"
  [[ ! -e $FORMAL_OUTPUT ]] || die "formal output already exists; use resume: $FORMAL_OUTPUT"
  fresh_command "$FORMAL_OUTPUT" "$FORMAL_JOB_NAME" "$TARGET_STEPS" "$SAVE_FREQ"
  launch_detached "$FORMAL_LOG" "$FORMAL_PID_FILE" fresh
}

resume_formal() {
  validate_dataset
  job_is_running "$FORMAL_JOB_NAME" && die "formal training is already running"
  local latest step archive
  latest=$(latest_valid_checkpoint "$FORMAL_OUTPUT" || true)
  if [[ -z $latest ]]; then
    if [[ -e $FORMAL_OUTPUT ]]; then
      archive=$FORMAL_OUTPUT.interrupted_$(date -u +%Y%m%dT%H%M%SZ)
      mv "$FORMAL_OUTPUT" "$archive"
      printf 'no complete checkpoint; archived partial output to %s\n' "$archive"
    fi
    validate_hardware_args
    require_dir "$BASE_POLICY"
    fresh_command "$FORMAL_OUTPUT" "$FORMAL_JOB_NAME" "$TARGET_STEPS" "$SAVE_FREQ"
    launch_detached "$FORMAL_LOG" "$FORMAL_PID_FILE" fresh
    return
  fi

  step=$(checkpoint_step "$latest")
  if [[ $step -ge $TARGET_STEPS ]]; then
    printf 'formal training already reached target: step=%s\n' "$step"
    return
  fi
  load_checkpoint_loader_args "$latest"
  validate_hardware_args
  archive_checkpoints_after "$FORMAL_OUTPUT" "$step"
  printf 'resuming formal training from %s\n' "$latest"
  resume_command "$latest" "$FORMAL_OUTPUT" "$FORMAL_JOB_NAME" "$TARGET_STEPS" "$SAVE_FREQ"
  launch_detached "$FORMAL_LOG" "$FORMAL_PID_FILE" resume
}

show_job_status() {
  local label=$1
  local job_name=$2
  local output=$3
  local log_path=$4
  local latest
  printf '%s:\n' "$label"
  if job_is_running "$job_name"; then
    pgrep -af "[l]erobot.scripts.lerobot_train.*$job_name"
  else
    printf '  process: not running\n'
  fi
  latest=$(latest_valid_checkpoint "$output" || true)
  if [[ -n $latest ]]; then
    printf '  latest complete checkpoint: %s\n' "$latest"
  else
    printf '  latest complete checkpoint: none\n'
  fi
  if [[ -f $log_path ]]; then
    printf '  log tail (%s):\n' "$log_path"
    tail -n 8 "$log_path" | sed 's/^/    /'
  fi
}

show_status() {
  show_job_status canary "$CANARY_JOB_NAME" "$CANARY_OUTPUT" "$CANARY_LOG"
  show_job_status formal "$FORMAL_JOB_NAME" "$FORMAL_OUTPUT" "$FORMAL_LOG"
}

main() {
  local action=${1:-}
  case "$action" in
    canary)
      start_canary
      ;;
    canary-resume)
      resume_canary
      ;;
    start)
      start_formal
      ;;
    resume)
      resume_formal
      ;;
    status)
      show_status
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
