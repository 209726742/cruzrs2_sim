#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
ADMISSION=$SCRIPT_DIR/sorting_roll_v15_admission.sh
OUTPUT_ROOT=
GPUS_CSV=0,1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --gpus)
      GPUS_CSV=$2
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "usage: $0 --out-root PATH [--gpus CSV]" >&2
  exit 2
fi

IFS=',' read -r -a gpus <<<"$GPUS_CSV"
if (( ${#gpus[@]} < 1 )); then
  echo "--gpus must contain at least one GPU ID" >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${seen_gpus[$gpu]+x}" ]]; then
    echo "--gpus must contain distinct non-negative GPU IDs" >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done

OUTPUT_ROOT=$(realpath -m "$OUTPUT_ROOT")
mkdir -p "$OUTPUT_ROOT"
bash "$PROJECT_ROOT/Sorting_Roll/run_scene.sh" check \
  >"$OUTPUT_ROOT/scene_check.log" 2>&1

run_worker() {
  local gpu=$1
  local groups=$2
  bash "$ADMISSION" \
    --out-root "$OUTPUT_ROOT" \
    --gpu "$gpu" \
    --groups "$groups" \
    --no-final-report \
    --skip-scene-check \
    >"$OUTPUT_ROOT/gpu${gpu}_admission.log" 2>&1
}

groups=(
  dynamics_heavy_low_friction
  dynamics_light_high_friction
  geometry_long
  geometry_medium
  geometry_short
)
worker_groups=()
for ((index = 0; index < ${#gpus[@]}; index++)); do
  worker_groups+=("")
done
for ((index = 0; index < ${#groups[@]}; index++)); do
  slot=$((index % ${#gpus[@]}))
  separator=
  if [[ -n "${worker_groups[$slot]}" ]]; then
    separator=,
  fi
  worker_groups[$slot]+=$separator${groups[$index]}
done

pids=()
for ((index = 0; index < ${#gpus[@]}; index++)); do
  [[ -n "${worker_groups[$index]}" ]] || continue
  run_worker "${gpus[$index]}" "${worker_groups[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if (( status != 0 )); then
  echo "v15 admission worker failed; inspect $OUTPUT_ROOT/gpu*_admission.log" >&2
  exit "$status"
fi

bash "$ADMISSION" \
  --out-root "$OUTPUT_ROOT" \
  --finalize-only \
  --skip-scene-check
