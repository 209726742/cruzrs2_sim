#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$PACKAGE_ROOT/.." && pwd)
PYTHON_BIN=${SORTING_ROLL_PYTHON:-$PROJECT_ROOT/envs/mjx/bin/python}
OUTPUT_ROOT=
ADMISSION_REPORT=
FIRST_SEED=3000
TOTAL=300
GPU_IDS=(0 1 2 3 4 5 6 7)
SEED_STARTS=(3000 3038 3076 3114 3152 3189 3226 3263)
COUNTS=(38 38 38 38 37 37 37 37)

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
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" || -z "$ADMISSION_REPORT" ]]; then
  echo "usage: $0 --out-root PATH --admission-report PATH" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing Python environment: $PYTHON_BIN" >&2
  exit 2
fi

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

rows=$(nvidia-smi -i 0,1,2,3,4,5,6,7 \
  --query-gpu=index,name --format=csv,noheader,nounits)
"$PYTHON_BIN" - "$rows" <<'PY'
import sys

rows = [line.strip() for line in sys.argv[1].splitlines() if line.strip()]
assert len(rows) == 8, rows
for expected, row in enumerate(rows):
    index, name = (part.strip() for part in row.split(",", 1))
    assert int(index) == expected, row
    assert "RTX 4090" in name, row
print("8x4090 collection hardware gate passed")
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
  local shard_index=$1
  local gpu=$2
  local seed_start=$3
  local count=$4
  local shard=$OUTPUT_ROOT/shard_$shard_index
  "$PYTHON_BIN" "$PACKAGE_ROOT/scripts/collection/sorting_roll_batch.py" \
    --out-root "$shard" \
    --seed-start "$seed_start" \
    --count "$count" \
    --min-success 1 \
    --gpu "$gpu" \
    --timeout 1800 \
    --render \
    --resume \
    --manifest "$MANIFEST" \
    >"$OUTPUT_ROOT/shard_${shard_index}_gpu${gpu}.log" 2>&1
}

pids=()
for index in "${!GPU_IDS[@]}"; do
  run_shard \
    "$index" \
    "${GPU_IDS[$index]}" \
    "${SEED_STARTS[$index]}" \
    "${COUNTS[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if (( status != 0 )); then
  echo "v15 collection worker failed; inspect $OUTPUT_ROOT/shard_*_gpu*.log" >&2
  exit "$status"
fi

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
paths = sorted(root.glob("shard_*/summary.json"))
if len(paths) != 8:
    raise SystemExit(f"expected 8 shard summaries, got {len(paths)}")
summaries = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in paths
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
        and sorted(seeds) == list(range(3000, 3300))
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
