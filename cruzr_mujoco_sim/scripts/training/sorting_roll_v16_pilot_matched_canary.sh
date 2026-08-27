#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
BASE_LAUNCHER=$PROJECT_ROOT/pi05_train.sh
POLICY=$SIM_ROOT/out/training/pi05_sorting_roll_v15_h100x4_fullft28k_seed1000/checkpoints/036000/pretrained_model
CONTROL_DATASET=$SIM_ROOT/out/datasets/sorting_roll_v15_diverse300_20260826_8x4090_lerobot_v30
TREATMENT_DATASET=$SIM_ROOT/out/datasets/sorting_roll_v16_pilot_mixed256_20260828_lerobot_v30
TREATMENT_WEIGHTS=$TREATMENT_DATASET/meta/sampling_weights_old50_h15_t15_r20.npy
CANARY_STEPS=${CANARY_STEPS:-3000}
CANARY_BATCH_SIZE=${CANARY_BATCH_SIZE:-8}
PREFLIGHT_STEPS=${PREFLIGHT_STEPS:-5}
PREFLIGHT_BATCH_SIZE=${PREFLIGHT_BATCH_SIZE:-8}

usage() {
  echo "usage: $0 {dry-run|preflight-tmux|preflight-status|canary-tmux|status|resume-tmux}"
}

configure() {
  local group=$1 stage=$2
  local dataset repo weights gpu_ids port suffix
  if [[ $group == control ]]; then
    dataset=$CONTROL_DATASET
    repo=local/sorting_roll_v15_control_train240
    weights=
    gpu_ids=0,1
    port=29620
  elif [[ $group == treatment ]]; then
    dataset=$TREATMENT_DATASET
    repo=local/sorting_roll_v16_pilot_mixed256
    weights=$TREATMENT_WEIGHTS
    gpu_ids=2,3
    port=29621
  else
    echo "unknown group: $group" >&2
    exit 2
  fi

  if [[ $stage == preflight ]]; then
    suffix=preflight
    STEPS=$PREFLIGHT_STEPS
    BATCH_SIZE=$PREFLIGHT_BATCH_SIZE
    SAVE_FREQ=1
    SAVE_CHECKPOINT=false
    WARMUP_STEPS=0
    DECAY_STEPS=1
  elif [[ $stage == canary ]]; then
    suffix=expert3k
    STEPS=$CANARY_STEPS
    BATCH_SIZE=$CANARY_BATCH_SIZE
    SAVE_FREQ=500
    SAVE_CHECKPOINT=true
    WARMUP_STEPS=200
    DECAY_STEPS=$CANARY_STEPS
  else
    echo "unknown stage: $stage" >&2
    exit 2
  fi

  OUTPUT=$SIM_ROOT/out/training/pi05_sorting_roll_v16_pilot_${group}_2x4090_${suffix}
  LOG=$PROJECT_ROOT/log/pi05_sorting_roll_v16_pilot_${group}_2x4090_${suffix}.log
  JOB=sorting_roll_v16_pilot_${group}_2x4090_${suffix}
  SESSION=sorting_roll_v16_${group}_${suffix}
  ARGS=(
    --dataset-root "$dataset"
    --repo-id "$repo"
    --episodes train
    --use-pretrained-stats true
    --base-policy "$POLICY"
    --output-dir "$OUTPUT"
    --log-file "$LOG"
    --job-name "$JOB"
    --gpu-ids "$gpu_ids"
    --num-processes 2
    --batch-size "$BATCH_SIZE"
    --num-workers 2
    --steps "$STEPS"
    --save-freq "$SAVE_FREQ"
    --save-checkpoint "$SAVE_CHECKPOINT"
    --log-freq 10
    --warmup-steps "$WARMUP_STEPS"
    --decay-steps "$DECAY_STEPS"
    --learning-rate 5e-6
    --gradient-checkpointing true
    --train-expert-only true
    --min-effective-batch 2
    --allow-small-batch true
    --port "$port"
  )
  [[ -z $weights ]] || ARGS+=(--frame-sampling-weights "$weights")
}

run_detached() {
  local group=$1 stage=$2 action=$3
  configure "$group" "$stage"
  bash "$BASE_LAUNCHER" "$action" "${ARGS[@]}"
  local supervisor
  supervisor=$(<"$LOG.pid")
  exec tail --pid="$supervisor" -n +1 -F "$LOG"
}

launch_pair() {
  local stage=$1 action=$2 group
  for group in control treatment; do
    configure "$group" "$stage"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux session already exists: $SESSION" >&2
      exit 1
    fi
  done
  for group in control treatment; do
    configure "$group" "$stage"
    tmux new-session -d -s "$SESSION" \
      "cd '$PROJECT_ROOT' && exec bash '$SCRIPT_DIR/$(basename "$0")' _run '$group' '$stage' '$action'"
    echo "started: $SESSION"
  done
}

status_pair() {
  local stage=$1 group
  for group in control treatment; do
    configure "$group" "$stage"
    echo "[$group]"
    bash "$BASE_LAUNCHER" status --output-dir "$OUTPUT" --log-file "$LOG" --job-name "$JOB"
    tmux has-session -t "$SESSION" 2>/dev/null \
      && echo "tmux: running ($SESSION)" \
      || echo "tmux: not running ($SESSION)"
  done
}

case "${1:-dry-run}" in
  dry-run)
    for group in control treatment; do
      configure "$group" canary
      echo "[$group]"
      bash "$BASE_LAUNCHER" dry-run "${ARGS[@]}"
    done
    ;;
  preflight-tmux) launch_pair preflight start ;;
  preflight-status) status_pair preflight ;;
  canary-tmux) launch_pair canary start ;;
  status) status_pair canary ;;
  resume-tmux) launch_pair canary resume ;;
  _run) run_detached "$2" "$3" "$4" ;;
  help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
