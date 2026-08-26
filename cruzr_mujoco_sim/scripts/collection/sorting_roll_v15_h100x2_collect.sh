#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$PACKAGE_ROOT/.." && pwd)
PYTHON_BIN=${SORTING_ROLL_PYTHON:-$PROJECT_ROOT/envs/mjx/bin/python}
OUTPUT_ROOT=
ADMISSION_REPORT=
GPUS_CSV=0,1
FIRST_SEED=3000
TOTAL=300

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --admission-report)
      ADMISSION_REPORT=$2
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

if [[ -z "$OUTPUT_ROOT" || -z "$ADMISSION_REPORT" ]]; then
  echo "usage: $0 --out-root PATH --admission-report PATH [--gpus CSV]" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing Python environment: $PYTHON_BIN" >&2
  exit 2
fi

IFS=',' read -r -a gpus <<<"$GPUS_CSV"
if (( ${#gpus[@]} < 1 || ${#gpus[@]} > TOTAL )); then
  echo "--gpus must contain between 1 and $TOTAL GPU IDs" >&2
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
ADMISSION_REPORT=$(realpath -m "$ADMISSION_REPORT")
mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" - "$ADMISSION_REPORT" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report.get("task_version") == "sorting_roll_v15_diverse_sim"
assert report.get("passed") is True
assert len(report.get("groups", [])) == 5
assert all(group.get("passed") is True for group in report["groups"])
print("v15 admission gate passed")
PY

bash "$PROJECT_ROOT/Sorting_Roll/run_scene.sh" check \
  >"$OUTPUT_ROOT/scene_check.log" 2>&1

MANIFEST=$OUTPUT_ROOT/campaign_manifest.json
CAMPAIGN=$(basename "$OUTPUT_ROOT")
if [[ -f "$MANIFEST" ]]; then
  "$PYTHON_BIN" "$PACKAGE_ROOT/scripts/core/sorting_roll_diversity.py" \
    check "$MANIFEST" >/dev/null
else
  "$PYTHON_BIN" "$PACKAGE_ROOT/scripts/core/sorting_roll_diversity.py" \
    generate \
    --out "$MANIFEST" \
    --campaign "$CAMPAIGN" \
    --seed-start "$FIRST_SEED" \
    --count "$TOTAL" \
    >"$OUTPUT_ROOT/manifest_counts.log"
fi

"$PYTHON_BIN" - "$MANIFEST" "$CAMPAIGN" "$FIRST_SEED" "$TOTAL" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
actual = (
    payload.get("task_version"),
    payload.get("campaign"),
    payload.get("seed_start"),
    payload.get("count"),
)
expected = (
    "sorting_roll_v15_diverse_sim",
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]),
)
assert actual == expected, (actual, expected)
assert len({item["seed"] for item in payload["assignments"]}) == expected[3]
print("v15 formal manifest gate passed")
PY

run_shard() {
  local gpu=$1
  local seed_start=$2
  local shard=$OUTPUT_ROOT/shard_$gpu
  "$PYTHON_BIN" "$PACKAGE_ROOT/scripts/collection/sorting_roll_batch.py" \
    --out-root "$shard" \
    --seed-start "$seed_start" \
    --count 150 \
    --min-success 135 \
    --gpu "$gpu" \
    --timeout 1800 \
    --render \
    --resume \
    --manifest "$MANIFEST" \
    >"$OUTPUT_ROOT/shard_${gpu}.log" 2>&1
}

run_shard 0 3000 &
pid0=$!
run_shard 1 3150 &
pid1=$!

status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
if (( status != 0 )); then
  echo "v15 formal collection shard failed; inspect $OUTPUT_ROOT/shard_*.log" >&2
  exit "$status"
fi

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summaries = [
    json.loads((root / f"shard_{gpu}" / "summary.json").read_text())
    for gpu in (0, 1)
]
records = [record for summary in summaries for record in summary["records"]]
seeds = [int(record["seed"]) for record in records]
successes = sum(bool(record.get("passed")) for record in records)
report = {
    "schema_version": 1,
    "task_version": "sorting_roll_v15_diverse_sim",
    "campaign": root.name,
    "attempted_count": len(records),
    "success_count": successes,
    "failed_count": len(records) - successes,
    "initial_success_rate": successes / len(records),
    "shard_count": len(summaries),
    "passed": (
        len(records) == 300
        and len(set(seeds)) == 300
        and successes >= 270
    ),
}
output = root / "initial_collection_report.json"
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise SystemExit(1)
PY
