#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
ADMISSION=$SCRIPT_DIR/sorting_roll_v15_admission.sh
OUTPUT_ROOT=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      OUTPUT_ROOT=$2
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "usage: $0 --out-root PATH" >&2
  exit 2
fi

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

run_worker 0 \
  dynamics_heavy_low_friction,geometry_long,geometry_short &
pid0=$!
run_worker 1 \
  dynamics_light_high_friction,geometry_medium &
pid1=$!

status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if (( status != 0 )); then
  echo "v15 admission worker failed; inspect $OUTPUT_ROOT/gpu*_admission.log" >&2
  exit "$status"
fi

bash "$ADMISSION" \
  --out-root "$OUTPUT_ROOT" \
  --finalize-only \
  --skip-scene-check
