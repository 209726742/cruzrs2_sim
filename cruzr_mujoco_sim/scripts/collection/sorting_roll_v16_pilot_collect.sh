#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PACKAGE_ROOT="$PROJECT_ROOT/cruzr_mujoco_sim"
PYTHON_BIN=${SORTING_ROLL_PYTHON:-$PROJECT_ROOT/envs/mjx/bin/python}
CAMPAIGN=${SORTING_ROLL_V16_CAMPAIGN:-sorting_roll_v16_pilot_admission_20260828}
OUTPUT_ROOT=${SORTING_ROLL_V16_OUTPUT_ROOT:-$PACKAGE_ROOT/output/sorting_roll_expert/$CAMPAIGN}
MANIFEST=${SORTING_ROLL_V16_MANIFEST:-$OUTPUT_ROOT/campaign_manifest.json}
MODE=${1:-representative}

case "$MODE" in
  representative)
    SEEDS=(5000 5003 5005 5010)
    ;;
  all)
    SEEDS=(5000 5001 5002 5003 5004 5005 5006 5007 5008 5009 5010 5011 5012 5013 5014 5015)
    ;;
  *)
    echo "usage: $0 {representative|all}" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing Python environment: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "missing pilot manifest: $MANIFEST" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs"
PIDS=()
FAILURES=0
worker_index=0
for seed in "${SEEDS[@]}"; do
  episode="$OUTPUT_ROOT/seed_$seed"
  if [[ -f "$episode/result.json" ]]; then
    if "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["success"] is True else 1)' "$episode/result.json"; then
      echo "[v16 collect] seed=$seed already successful; skip"
      continue
    fi
    echo "[v16 collect] seed=$seed has a failed immutable result: $episode" >&2
    FAILURES=$((FAILURES + 1))
    continue
  fi
  if [[ -e "$episode" ]]; then
    echo "[v16 collect] seed=$seed output exists without result: $episode" >&2
    FAILURES=$((FAILURES + 1))
    continue
  fi
  gpu=$((worker_index % 4))
  worker_index=$((worker_index + 1))
  log="$OUTPUT_ROOT/logs/seed_${seed}.log"
  echo "[v16 collect] start seed=$seed gpu=$gpu log=$log"
  "$PYTHON_BIN" "$PACKAGE_ROOT/scripts/collection/sorting_roll_v16_pilot_expert.py" \
    --out "$episode" \
    --seed "$seed" \
    --gpu "$gpu" \
    --randomize \
    --review-videos \
    --manifest "$MANIFEST" >"$log" 2>&1 &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILURES=$((FAILURES + 1))
  fi
done

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
report="$OUTPUT_ROOT/validation_report_${MODE}_${timestamp}.json"
validate_args=(
  "$PYTHON_BIN"
  "$PACKAGE_ROOT/scripts/collection/sorting_roll_v16_validate.py"
  "$OUTPUT_ROOT"
  --manifest "$MANIFEST"
  --report "$report"
)
if [[ "$MODE" == "all" ]]; then
  validate_args+=(--require-complete)
fi
if ! "${validate_args[@]}"; then
  FAILURES=$((FAILURES + 1))
fi

echo "[v16 collect] mode=$MODE failures=$FAILURES report=$report"
exit "$FAILURES"
