#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
ISAAC_PY=${ISAAC_PY:-/isaac-sim/python.sh}
SOURCE_CAMPAIGN=${SORTING_ROLL_V16_EXPANSION_CAMPAIGN:-sorting_roll_v16_stage80_20260829_v51_safe_fast_nav}
DATASET_CAMPAIGN=${SORTING_ROLL_V16_DATASET_CAMPAIGN:-sorting_roll_v16_stage80_mixed320_20260829}
SOURCE_ROOT=${SORTING_ROLL_V16_EXPANSION_OUTPUT_ROOT:-$SIM_ROOT/out/collection/sorting_roll_v16_stage80_20260829_v93_final_v51_manifest}
MANIFEST=${SORTING_ROLL_V16_EXPANSION_MANIFEST:-$SIM_ROOT/out/collection/$SOURCE_CAMPAIGN/campaign_manifest.json}
V15_ROOT=$SIM_ROOT/output/sorting_roll_expert/sorting_roll_v15_diverse300_20260826_8x4090
V15_VALIDATION=${SORTING_ROLL_V15_VALIDATION:-$V15_ROOT/validation_report.json}
V15_V21=${SORTING_ROLL_V15_V21:-$SIM_ROOT/out/datasets/sorting_roll_v15_diverse300_20260826_8x4090_lerobot_v21}
DATASET_V21=$SIM_ROOT/out/datasets/${DATASET_CAMPAIGN}_lerobot_v21
LOG_ROOT=$PROJECT_ROOT/log/$DATASET_CAMPAIGN
BUILD_REPORT=$LOG_ROOT/build_v21_report.json
READINESS=$LOG_ROOT/data_training_readiness.json
SESSION=sorting_roll_v16_${DATASET_CAMPAIGN}_dataset

require_inputs() {
  local path
  for path in "$MANIFEST" "$V15_VALIDATION" "$V15_V21/meta/info.json"; do
    if [[ ! -f $path ]]; then
      echo "missing required input: $path" >&2
      exit 1
    fi
  done
}

run_pipeline() {
  require_inputs
  mkdir -p "$LOG_ROOT"
  if [[ -e $DATASET_V21 || -e $BUILD_REPORT ]]; then
    if [[ ! -f $DATASET_V21/meta/info.json || ! -f $BUILD_REPORT ]]; then
      echo "refusing inconsistent existing v2.1 output" >&2
      exit 1
    fi
    echo "reuse complete v2.1 dataset: $DATASET_V21"
  else
    cd "$PROJECT_ROOT"
    PYTHONPATH=. "$ISAAC_PY" \
      "$SIM_ROOT/scripts/collection/sorting_roll_v16_build_mixed_v21.py" \
      --v15-validation "$V15_VALIDATION" \
      --v15-v21 "$V15_V21" \
      --v16-root "$SOURCE_ROOT" \
      --v16-manifest "$MANIFEST" \
      --manifest-kind expansion \
      --out "$DATASET_V21" \
      --report "$BUILD_REPORT" \
      --encode-workers 4 \
      >"$LOG_ROOT/build_v21.log" 2>&1
  fi

  env \
    ISAAC_PY="$ISAAC_PY" \
    SORTING_ROLL_V16_CAMPAIGN="$DATASET_CAMPAIGN" \
    SORTING_ROLL_V16_SOURCE_CAMPAIGN="$SOURCE_CAMPAIGN" \
    SORTING_ROLL_V16_TASK_VERSION=sorting_roll_v16_expansion_stage_sim \
    SORTING_ROLL_V16_SAMPLING_PROFILE=stage80_old50 \
    SORTING_ROLL_V16_TRAIN_EPISODES=0:304 \
    SORTING_ROLL_V16_CANDIDATE_STAGE=v16_expansion_80 \
    bash "$SCRIPT_DIR/sorting_roll_v16_pilot_dataset_pipeline.sh"
}

status() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux: running ($SESSION)"
  else
    echo "tmux: not running ($SESSION)"
  fi
  if [[ -f $READINESS ]]; then
    "$ISAAC_PY" -c \
      'import json,sys; p=json.load(open(sys.argv[1])); print(json.dumps({"candidate_stage":p.get("candidate_stage"),"episodes":p.get("episodes"),"ready":p.get("ready_for_full_parameter_canary")},indent=2))' \
      "$READINESS"
  else
    echo "readiness: pending ($READINESS)"
  fi
}

case "${1:-dry-run}" in
  dry-run)
    require_inputs
    printf 'source=%s\nmanifest=%s\ndataset=%s\nreadiness=%s\n' \
      "$SOURCE_ROOT" "$MANIFEST" "$DATASET_V21" "$READINESS"
    ;;
  run-tmux)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux session already exists: $SESSION" >&2
      exit 1
    fi
    tmux new-session -d -s "$SESSION" \
      "cd '$PROJECT_ROOT' && exec bash '$SCRIPT_DIR/$(basename "$0")' _run"
    echo "started: $SESSION"
    ;;
  status) status ;;
  _run) run_pipeline ;;
  *) echo "usage: $0 {dry-run|run-tmux|status}" >&2; exit 2 ;;
esac
