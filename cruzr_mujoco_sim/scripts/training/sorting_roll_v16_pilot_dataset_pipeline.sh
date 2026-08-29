#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
ISAAC_PY=${ISAAC_PY:-/isaac-sim/python.sh}
CAMPAIGN=${SORTING_ROLL_V16_CAMPAIGN:-sorting_roll_v16_pilot_mixed256_20260828}
V15_CAMPAIGN=sorting_roll_v15_diverse300_20260826_8x4090
V16_CAMPAIGN=${SORTING_ROLL_V16_SOURCE_CAMPAIGN:-sorting_roll_v16_pilot_20260828_v3}
V16_TASK_VERSION=${SORTING_ROLL_V16_TASK_VERSION:-sorting_roll_v16_expansion_pilot_sim}
SAMPLING_PROFILE=${SORTING_ROLL_V16_SAMPLING_PROFILE:-pilot_old50}
TRAIN_EPISODES=${SORTING_ROLL_V16_TRAIN_EPISODES:-0:252}
CANDIDATE_STAGE=${SORTING_ROLL_V16_CANDIDATE_STAGE:-v16_pilot_16}
DATASET_V21=$SIM_ROOT/out/datasets/${CAMPAIGN}_lerobot_v21
DATASET_V30=$SIM_ROOT/out/datasets/${CAMPAIGN}_lerobot_v30
LOG_ROOT=$PROJECT_ROOT/log/$CAMPAIGN
BUILD_REPORT=$LOG_ROOT/build_v21_report.json
AUDIT_REPORT=$LOG_ROOT/dataset_v30_audit.json
READINESS=$LOG_ROOT/data_training_readiness.json
if [[ $SAMPLING_PROFILE == pilot_old50 ]]; then
  SAMPLING_TAG=old50_h15_t15_r20
elif [[ $SAMPLING_PROFILE == full_v2_old70 ]]; then
  SAMPLING_TAG=old70_h10_t10_r10
elif [[ $SAMPLING_PROFILE == stage80_old50 ]]; then
  SAMPLING_TAG=old50_h15_t15_r15_c5
else
  echo "unsupported sampling profile: $SAMPLING_PROFILE" >&2
  exit 2
fi
SAMPLING_WEIGHTS=$DATASET_V30/meta/sampling_weights_${SAMPLING_TAG}.npy
SAMPLING_REPORT=$LOG_ROOT/sampling_weights_${SAMPLING_TAG}.json
REPO_ID=local/$CAMPAIGN

if [[ ! -x "$ISAAC_PY" ]]; then
  echo "missing Isaac Python: $ISAAC_PY" >&2
  exit 1
fi
if [[ ! -f "$DATASET_V21/meta/info.json" || ! -f "$BUILD_REPORT" ]]; then
  echo "v2.1 dataset or build report is incomplete" >&2
  exit 1
fi
mkdir -p "$LOG_ROOT"

if [[ ! -e "$DATASET_V30" ]]; then
  cp -a "$DATASET_V21" "$DATASET_V30"
  cd "$PROJECT_ROOT"
  PYTHONPATH=. "$ISAAC_PY" \
    src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id "$REPO_ID" \
    --root "$DATASET_V30" \
    --push-to-hub false \
    >"$LOG_ROOT/dataset_v30_convert.log" 2>&1
elif ! "$ISAAC_PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["codebase_version"] == "v3.0"' "$DATASET_V30/meta/info.json"; then
  echo "refusing to reuse incomplete v3.0 dataset: $DATASET_V30" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
HF_HUB_OFFLINE=1 PYTHONPATH=. "$ISAAC_PY" \
  "$SIM_ROOT/scripts/training/sorting_roll_v16_dataset_audit.py" \
  --dataset "$DATASET_V30" \
  --repo-id "$REPO_ID" \
  --v15-campaign "$V15_CAMPAIGN" \
  --v16-campaign "$V16_CAMPAIGN" \
  --v16-task-version "$V16_TASK_VERSION" \
  --build-report "$BUILD_REPORT" \
  --out "$AUDIT_REPORT" \
  >"$LOG_ROOT/dataset_v30_audit.log" 2>&1

PYTHONPATH=. "$ISAAC_PY" \
  "$SIM_ROOT/scripts/training/sorting_roll_v16_sampling_weights.py" \
  --dataset "$DATASET_V30" \
  --repo-id "$REPO_ID" \
  --episodes "$TRAIN_EPISODES" \
  --output "$SAMPLING_WEIGHTS" \
  --report "$SAMPLING_REPORT" \
  --profile "$SAMPLING_PROFILE" \
  >"$LOG_ROOT/sampling_weights.log" 2>&1

"$ISAAC_PY" - "$DATASET_V21" "$DATASET_V30" "$REPO_ID" \
  "$BUILD_REPORT" "$AUDIT_REPORT" "$SAMPLING_REPORT" "$READINESS" \
  "$CANDIDATE_STAGE" <<'PY'
import json
from pathlib import Path
import sys

v21, v30, repo_id, build_path, audit_path, sampling_path, output_path, stage = sys.argv[1:]
build = json.loads(Path(build_path).read_text(encoding="utf-8"))
audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
sampling = json.loads(Path(sampling_path).read_text(encoding="utf-8"))
payload = {
    "schema_version": 1,
    "candidate_stage": stage,
    "dataset_v21": v21,
    "dataset_v30": v30,
    "repo_id": repo_id,
    "build_report": build_path,
    "audit_report": audit_path,
    "sampling_report": sampling_path,
    "sampling_weights": sampling.get("weights_path"),
    "episodes": audit.get("episodes"),
    "frames": audit.get("frames"),
    "splits": audit.get("splits"),
    "v16_family_counts": audit.get("v16_family_counts"),
    "target_sampling_fractions": sampling.get("target_fractions"),
    "ready_for_full_parameter_canary": (
        build.get("passed") is True
        and audit.get("passed") is True
        and sampling.get("passed") is True
        and audit.get("episodes") == build.get("total_count")
    ),
}
output = Path(output_path)
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps(payload, indent=2))
if not payload["ready_for_full_parameter_canary"]:
    raise SystemExit(1)
PY
