#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$PACKAGE_ROOT/.." && pwd)
PYTHON_BIN=${SORTING_ROLL_PYTHON:-$PROJECT_ROOT/envs/mjx/bin/python}
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

OUTPUT_ROOT=$(realpath -m "$OUTPUT_ROOT")
"$PYTHON_BIN" "$SCRIPT_DIR/sorting_roll_v15_finalize.py" \
  --root "$OUTPUT_ROOT" --gpus "$GPUS_CSV"

mapfile -t sources <"$OUTPUT_ROOT/selected_sources.txt"
if (( ${#sources[@]} != 300 )); then
  echo "expected 300 selected sources, got ${#sources[@]}" >&2
  exit 1
fi
"$PYTHON_BIN" "$SCRIPT_DIR/sorting_roll_validate.py" \
  "${sources[@]}" \
  --manifest "$OUTPUT_ROOT/campaign_manifest_with_replacements.json" \
  --report "$OUTPUT_ROOT/validation_report.json"

"$PYTHON_BIN" - "$OUTPUT_ROOT/validation_report.json" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report.get("task_version") == "sorting_roll_v15_diverse_sim"
assert report.get("episode_count") == 300
assert report.get("passed_count") == 300
assert report.get("failed_count") == 0
assert report.get("passed") is True
print("v15 source validation gate passed: 300/300")
PY
