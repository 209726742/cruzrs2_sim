#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
MJX_PY=${SORTING_ROLL_PYTHON:-$PROJECT_ROOT/envs/mjx/bin/python}
ISAAC_PY=${ISAAC_PY:-/isaac-sim/python.sh}
SOURCE_ROOT=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      SOURCE_ROOT=$2
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SOURCE_ROOT" ]]; then
  echo "usage: $0 --source-root PATH" >&2
  exit 2
fi
if [[ ! -x "$MJX_PY" || ! -x "$ISAAC_PY" ]]; then
  echo "required Python environment is unavailable" >&2
  exit 2
fi

SOURCE_ROOT=$(realpath -m "$SOURCE_ROOT")
CAMPAIGN=$(basename "$SOURCE_ROOT")
DATASET_V21=$SIM_ROOT/out/datasets/${CAMPAIGN}_lerobot_v21
DATASET_V30=$SIM_ROOT/out/datasets/${CAMPAIGN}_lerobot_v30
REPO_ID=local/$CAMPAIGN
LOG_ROOT=$PROJECT_ROOT/log/$CAMPAIGN
MANIFEST=$SOURCE_ROOT/campaign_manifest_with_replacements.json
SELECTION=$SOURCE_ROOT/selection_report.json
VALIDATION=$SOURCE_ROOT/validation_report.json
mkdir -p "$LOG_ROOT"

"$MJX_PY" - "$SELECTION" "$VALIDATION" <<'PY'
import json
from pathlib import Path
import sys

selection, validation = (
    json.loads(Path(path).read_text(encoding="utf-8"))
    for path in sys.argv[1:]
)
assert selection.get("task_version") == "sorting_roll_v15_diverse_sim"
assert selection.get("selected_count") == 300
assert selection.get("passed") is True
assert validation.get("task_version") == "sorting_roll_v15_diverse_sim"
assert validation.get("episode_count") == validation.get("passed_count") == 300
assert validation.get("failed_count") == 0
assert validation.get("passed") is True
print("v15 source build gate passed")
PY

mapfile -t sources <"$SOURCE_ROOT/selected_sources.txt"
if (( ${#sources[@]} != 300 )); then
  echo "expected 300 source paths, got ${#sources[@]}" >&2
  exit 1
fi

if [[ ! -e "$DATASET_V21" ]]; then
  "$MJX_PY" "$SIM_ROOT/scripts/collection/sorting_roll_build_v21.py" \
    "${sources[@]}" \
    --out "$DATASET_V21" \
    --encode-workers 4 \
    --manifest "$MANIFEST" \
    >"$LOG_ROOT/dataset_v21_build.log" 2>&1
elif [[ ! -f "$DATASET_V21/meta/info.json" ]]; then
  echo "incomplete v2.1 dataset exists: $DATASET_V21" >&2
  exit 1
fi

if [[ ! -e "$DATASET_V30" ]]; then
  cp -a "$DATASET_V21" "$DATASET_V30"
  cd "$PROJECT_ROOT"
  PYTHONPATH=. "$ISAAC_PY" \
    src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id "$REPO_ID" \
    --root "$DATASET_V30" \
    --push-to-hub false \
    >"$LOG_ROOT/dataset_v30_convert.log" 2>&1
elif ! "$ISAAC_PY" - "$DATASET_V30/meta/info.json" <<'PY'
import json
import sys
assert json.load(open(sys.argv[1], encoding="utf-8")).get("codebase_version") == "v3.0"
PY
then
  echo "incomplete v3.0 dataset exists: $DATASET_V30" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
HF_HUB_OFFLINE=1 PYTHONPATH=. "$ISAAC_PY" \
  "$SIM_ROOT/scripts/training/sorting_roll_v15_dataset_audit.py" \
  --dataset "$DATASET_V30" \
  --repo-id "$REPO_ID" \
  --campaign "$CAMPAIGN" \
  --selection-report "$SELECTION" \
  --out "$LOG_ROOT/dataset_v30_audit.json" \
  >"$LOG_ROOT/dataset_v30_audit.log" 2>&1

"$MJX_PY" - "$SOURCE_ROOT/dataset_paths.json" \
  "$DATASET_V21" "$DATASET_V30" "$REPO_ID" \
  "$LOG_ROOT/dataset_v30_audit.json" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "task_version": "sorting_roll_v15_diverse_sim",
    "dataset_v21": sys.argv[2],
    "dataset_v30": sys.argv[3],
    "repo_id": sys.argv[4],
    "audit_report": sys.argv[5],
    "passed": True,
}
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps(payload, indent=2))
PY
