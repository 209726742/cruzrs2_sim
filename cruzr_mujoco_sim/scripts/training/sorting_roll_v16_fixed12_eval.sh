#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
MJX_PY=${SORTING_ROLL_PYTHON:-$PROJECT_ROOT/envs/mjx/bin/python}
ISAAC_PY=${ISAAC_PY:-/isaac-sim/python.sh}
SERVER=$SCRIPT_DIR/sorting_roll_pi05_official_server.py
ROLLOUT=$SCRIPT_DIR/sorting_roll_pi05_fixed_rollout.py
AUDIT=$SCRIPT_DIR/sorting_roll_v16_fixed12_audit.py
V15_MANIFEST=$SIM_ROOT/output/sorting_roll_expert/sorting_roll_v15_diverse300_20260826_8x4090/campaign_manifest.json
EVAL_ROOT=${EVAL_ROOT:-$SIM_ROOT/out/rollout/sorting_roll_v16_fixed12_matched_20260828}
V16_MANIFEST=$EVAL_ROOT/fixed_v16_manifest.json
REPORT=$EVAL_ROOT/fixed12_audit.json
LOG=$PROJECT_ROOT/log/sorting_roll_v16_fixed12_matched_20260828.log
SESSION=sorting_roll_v16_fixed12
POLICY_SEED=${POLICY_SEED:-28000}
REPLAN=${REPLAN:-20}

LABELS=(original36 control3k treatment3k)
CHECKPOINTS=(
  "$SIM_ROOT/out/training/pi05_sorting_roll_v15_h100x4_fullft28k_seed1000/checkpoints/036000/pretrained_model"
  "$SIM_ROOT/out/training/pi05_sorting_roll_v16_pilot_control_2x4090_expert3k/checkpoints/003000/pretrained_model"
  "$SIM_ROOT/out/training/pi05_sorting_roll_v16_pilot_treatment_2x4090_expert3k/checkpoints/003000/pretrained_model"
)
PORTS=(8742 8743 8744)
GPUS=(0 1 2)
CASES=(
  v15:3010 v15:3020 v15:3040
  v16:6000 v16:6001 v16:6004
  v16:6003 v16:6007 v16:6011
  v16:6002 v16:6005 v16:6006
)

usage() {
  echo "usage: $0 {dry-run|start-tmux|status|_run}"
}

checkpoint_ready() {
  [[ -f $1/model.safetensors && -f $1/config.json ]]
}

generate_manifest() {
  [[ -f $V16_MANIFEST ]] && return
  mkdir -p "$EVAL_ROOT"
  "$MJX_PY" "$SIM_ROOT/scripts/core/sorting_roll_v16_pilot_contract.py" \
    generate --out "$V16_MANIFEST" \
    --campaign sorting_roll_v16_fixed12_eval_20260828 --seed-start 6000
}

wait_for_server() {
  local pid=$1 log=$2
  for _ in $(seq 1 240); do
    grep -q "serving official" "$log" 2>/dev/null && return
    kill -0 "$pid" 2>/dev/null || {
      tail -n 80 "$log" >&2
      return 1
    }
    sleep 1
  done
  echo "policy server did not become ready: $log" >&2
  return 1
}

run_label() {
  local label=$1 checkpoint=$2 port=$3 case_spec kind seed manifest output case_log
  for case_spec in "${CASES[@]}"; do
    kind=${case_spec%%:*}
    seed=${case_spec##*:}
    manifest=$V16_MANIFEST
    [[ $kind == v15 ]] && manifest=$V15_MANIFEST
    output=$EVAL_ROOT/$label/${kind}_seed${seed}
    case_log=$EVAL_ROOT/logs/${label}_${kind}_seed${seed}.log
    if [[ -f $output/result.json ]]; then
      echo "[fixed12] skip completed $label $kind seed=$seed"
      continue
    fi
    if [[ -e $output ]]; then
      echo "[fixed12] incomplete immutable output: $output" >&2
      return 1
    fi
    echo "[fixed12] start $label $kind seed=$seed"
    env \
      CUDA_VISIBLE_DEVICES= \
      ROLLOUT_MANIFEST="$manifest" \
      ROLLOUT_MANIFEST_KIND="$kind" \
      ROLLOUT_SEED="$seed" \
      ROLLOUT_REPLAN="$REPLAN" \
      POLICY_SAMPLE_SEED="$POLICY_SEED" \
      CHECKPOINT_LABEL="$label" \
      EXPECTED_CHECKPOINT="$checkpoint" \
      ROLLOUT_OUTPUT="$output" \
      POLICY_PORT="$port" \
      "$MJX_PY" "$ROLLOUT" >"$case_log" 2>&1
  done
}

run_all() {
  local index label checkpoint port gpu server_log
  local -a server_pids=() worker_pids=()
  mkdir -p "$EVAL_ROOT/logs" "$(dirname "$LOG")"
  exec > >(tee -a "$LOG") 2>&1
  for checkpoint in "${CHECKPOINTS[@]}"; do
    checkpoint_ready "$checkpoint" || {
      echo "checkpoint is incomplete: $checkpoint" >&2
      return 1
    }
  done
  generate_manifest

  cleanup() {
    local pid
    for pid in "${server_pids[@]:-}"; do
      kill "$pid" 2>/dev/null || true
    done
  }
  trap cleanup EXIT INT TERM

  for index in "${!LABELS[@]}"; do
    label=${LABELS[$index]}
    checkpoint=${CHECKPOINTS[$index]}
    port=${PORTS[$index]}
    gpu=${GPUS[$index]}
    server_log=$EVAL_ROOT/logs/${label}_server.log
    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      "$ISAAC_PY" "$SERVER" \
      --checkpoint "$checkpoint" --device cuda:0 --port "$port" \
      --cuda-memory-fraction 0.5 --default-policy-seed "$POLICY_SEED" \
      >"$server_log" 2>&1 &
    server_pids+=("$!")
  done
  for index in "${!LABELS[@]}"; do
    wait_for_server "${server_pids[$index]}" \
      "$EVAL_ROOT/logs/${LABELS[$index]}_server.log"
  done

  for index in "${!LABELS[@]}"; do
    run_label "${LABELS[$index]}" "${CHECKPOINTS[$index]}" \
      "${PORTS[$index]}" &
    worker_pids+=("$!")
  done
  failures=0
  for index in "${!worker_pids[@]}"; do
    wait "${worker_pids[$index]}" || failures=$((failures + 1))
  done
  (( failures == 0 )) || {
    echo "fixed12 rollout workers failed: $failures" >&2
    return 1
  }
  "$MJX_PY" "$AUDIT" "$EVAL_ROOT" --out "$REPORT"
}

status() {
  tmux has-session -t "$SESSION" 2>/dev/null \
    && echo "tmux: running ($SESSION)" \
    || echo "tmux: not running ($SESSION)"
  local label count
  for label in "${LABELS[@]}"; do
    if [[ -d $EVAL_ROOT/$label ]]; then
      count=$(find "$EVAL_ROOT/$label" -mindepth 2 -maxdepth 2 \
        -name result.json | wc -l)
    else
      count=0
    fi
    echo "$label: $count/12"
  done
  if [[ -f $REPORT ]]; then
    "$MJX_PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("passed=",r["passed"],"ready_to_expand=",r["ready_to_expand"]); print(r["summaries"])' "$REPORT"
  fi
  [[ ! -f $LOG ]] || tail -n 30 "$LOG"
}

case "${1:-dry-run}" in
  dry-run)
    printf 'evaluation root: %s\n' "$EVAL_ROOT"
    for index in "${!LABELS[@]}"; do
      checkpoint_ready "${CHECKPOINTS[$index]}" \
        && state=ready || state=pending
      printf '%s gpu=%s port=%s checkpoint=%s [%s]\n' \
        "${LABELS[$index]}" "${GPUS[$index]}" "${PORTS[$index]}" \
        "${CHECKPOINTS[$index]}" "$state"
    done
    printf 'cases per checkpoint: %s\n' "${CASES[*]}"
    ;;
  start-tmux)
    tmux has-session -t "$SESSION" 2>/dev/null && {
      echo "tmux session already exists: $SESSION" >&2
      exit 1
    }
    tmux new-session -d -s "$SESSION" \
      "cd '$PROJECT_ROOT' && exec bash '$0' _run"
    echo "started: $SESSION"
    ;;
  status) status ;;
  _run) run_all ;;
  help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
