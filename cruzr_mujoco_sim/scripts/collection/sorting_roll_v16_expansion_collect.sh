#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PACKAGE_ROOT="$PROJECT_ROOT/cruzr_mujoco_sim"
PYTHON_BIN=${SORTING_ROLL_PYTHON:-$PROJECT_ROOT/envs/mjx/bin/python}
CAMPAIGN=${SORTING_ROLL_V16_EXPANSION_CAMPAIGN:-sorting_roll_v16_stage80_20260828}
OUTPUT_ROOT=${SORTING_ROLL_V16_EXPANSION_OUTPUT_ROOT:-$PACKAGE_ROOT/output/sorting_roll_expert/$CAMPAIGN}
MANIFEST=${SORTING_ROLL_V16_EXPANSION_MANIFEST:-$OUTPUT_ROOT/campaign_manifest.json}
STAGE=${SORTING_ROLL_V16_EXPANSION_STAGE:-80}
SEED_START=${SORTING_ROLL_V16_EXPANSION_SEED_START:-6000}
GPU_CSV=${SORTING_ROLL_V16_GPUS:-0,1,2,3}
MODE=${1:-representative}
CONTRACT="$PACKAGE_ROOT/scripts/core/sorting_roll_v16_expansion_contract.py"
EXPERT="$PACKAGE_ROOT/scripts/collection/sorting_roll_v16_expansion_expert.py"
VALIDATOR="$PACKAGE_ROOT/scripts/collection/sorting_roll_v16_expansion_validate.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing Python environment: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs"
if [[ "$MODE" == "prepare" ]]; then
  if [[ -f "$MANIFEST" ]]; then
    "$PYTHON_BIN" "$CONTRACT" check "$MANIFEST"
  else
    "$PYTHON_BIN" "$CONTRACT" generate \
      --out "$MANIFEST" \
      --campaign "$CAMPAIGN" \
      --stage "$STAGE" \
      --seed-start "$SEED_START"
  fi
  exit 0
fi

case "$MODE" in
  representative|all|validate)
    ;;
  *)
    echo "usage: $0 {prepare|representative|all|validate}" >&2
    exit 2
    ;;
esac

if [[ ! -f "$MANIFEST" ]]; then
  echo "missing expansion manifest; run '$0 prepare' first: $MANIFEST" >&2
  exit 1
fi
"$PYTHON_BIN" "$CONTRACT" check "$MANIFEST" >/dev/null

IFS=',' read -r -a GPUS <<<"$GPU_CSV"
if [[ ${#GPUS[@]} -lt 1 ]]; then
  echo "SORTING_ROLL_V16_GPUS must contain at least one GPU index" >&2
  exit 1
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU index: $gpu" >&2
    exit 1
  fi
done

if [[ "$MODE" != "validate" ]]; then
  mapfile -t SEEDS < <(
    "$PYTHON_BIN" "$CONTRACT" select "$MANIFEST" --mode "$MODE"
  )
  FAILURES=0
  PIDS=()
  for seed in "${SEEDS[@]}"; do
    episode="$OUTPUT_ROOT/seed_$seed"
    if [[ -f "$episode/result.json" ]]; then
      if "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["success"] is True else 1)' "$episode/result.json"; then
        echo "[v16 expansion collect] seed=$seed already successful; skip"
        continue
      fi
      echo "[v16 expansion collect] failed immutable result exists: $episode" >&2
      FAILURES=$((FAILURES + 1))
      continue
    fi
    if [[ -e "$episode" ]]; then
      echo "[v16 expansion collect] incomplete immutable output exists: $episode" >&2
      FAILURES=$((FAILURES + 1))
      continue
    fi
    worker=${#PIDS[@]}
    gpu=${GPUS[$worker]}
    log="$OUTPUT_ROOT/logs/seed_${seed}.log"
    echo "[v16 expansion collect] start seed=$seed gpu=$gpu log=$log"
    "$PYTHON_BIN" "$EXPERT" \
      --out "$episode" \
      --seed "$seed" \
      --gpu "$gpu" \
      --randomize \
      --review-videos \
      --manifest "$MANIFEST" >"$log" 2>&1 &
    PIDS+=("$!")
    if [[ ${#PIDS[@]} -eq ${#GPUS[@]} ]]; then
      for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
          FAILURES=$((FAILURES + 1))
        fi
      done
      PIDS=()
    fi
  done
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      FAILURES=$((FAILURES + 1))
    fi
  done
else
  FAILURES=0
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
report="$OUTPUT_ROOT/validation_report_${MODE}_${timestamp}.json"
validate_args=(
  "$PYTHON_BIN"
  "$VALIDATOR"
  "$OUTPUT_ROOT"
  --manifest "$MANIFEST"
  --report "$report"
)
if [[ "$MODE" == "all" || "$MODE" == "validate" ]]; then
  validate_args+=(--require-complete)
fi
if ! "${validate_args[@]}"; then
  FAILURES=$((FAILURES + 1))
fi

echo "[v16 expansion collect] mode=$MODE failures=$FAILURES report=$report"
exit "$FAILURES"
