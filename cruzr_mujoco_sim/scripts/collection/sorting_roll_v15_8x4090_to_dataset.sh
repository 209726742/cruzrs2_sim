#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$PACKAGE_ROOT/.." && pwd)
PYTHON_BIN=${SORTING_ROLL_PYTHON:-$PROJECT_ROOT/envs/mjx/bin/python}
ADMISSION_SESSION=${ADMISSION_SESSION:-sorting_roll_v15_admission_8x4090}
ADMISSION_ROOT=${ADMISSION_ROOT:-$PACKAGE_ROOT/output/sorting_roll_expert/v15_diverse_admission_20260826_8x4090_fix}
SOURCE_ROOT=${SOURCE_ROOT:-$PACKAGE_ROOT/output/sorting_roll_expert/sorting_roll_v15_diverse300_20260826_8x4090}
POLL_SECONDS=${POLL_SECONDS:-30}
GPUS=0,1,2,3,4,5,6,7

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

log "waiting for admission session $ADMISSION_SESSION"
while tmux has-session -t "$ADMISSION_SESSION" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

ADMISSION_REPORT=$ADMISSION_ROOT/admission_report.json
"$PYTHON_BIN" - "$ADMISSION_REPORT" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report.get("task_version") == "sorting_roll_v15_diverse_sim"
assert report.get("passed") is True
assert len(report.get("groups", [])) == 5
assert all(group.get("passed") is True for group in report["groups"])
print("v15 admission accepted")
PY

log "starting 300-episode 8x4090 collection"
bash "$SCRIPT_DIR/sorting_roll_v15_8x4090_collect.sh" \
  --out-root "$SOURCE_ROOT" \
  --admission-report "$ADMISSION_REPORT"

log "collecting quota-preserving replacements and validating 300/300"
bash "$SCRIPT_DIR/sorting_roll_v15_h100x2_finalize.sh" \
  --out-root "$SOURCE_ROOT" \
  --gpus "$GPUS"

log "generating representative hard-case review videos"
bash "$SCRIPT_DIR/sorting_roll_v15_review_bundle.sh" \
  --out-root "$SOURCE_ROOT" \
  --gpu 0

log "building and auditing LeRobot v3.0 dataset"
bash "$PACKAGE_ROOT/scripts/training/sorting_roll_v15_dataset_pipeline.sh" \
  --source-root "$SOURCE_ROOT"

"$PYTHON_BIN" - "$SOURCE_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
selection = json.loads((root / "selection_report.json").read_text())
validation = json.loads((root / "validation_report.json").read_text())
dataset = json.loads((root / "dataset_paths.json").read_text())
review = json.loads((root / "review_bundle/review_bundle.json").read_text())
payload = {
    "schema_version": 1,
    "task_version": "sorting_roll_v15_diverse_sim",
    "selected_episodes": selection["selected_count"],
    "source_validation_passed": validation["passed"],
    "dataset_v30": dataset["dataset_v30"],
    "dataset_audit": dataset["audit_report"],
    "review_bundle": str(root / "review_bundle/review_bundle.json"),
    "ready_for_full_parameter_canary": (
        selection["passed"] is True
        and validation["passed"] is True
        and dataset["passed"] is True
        and review["passed"] is True
    ),
}
output = root / "data_training_readiness.json"
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps(payload, indent=2))
if not payload["ready_for_full_parameter_canary"]:
    raise SystemExit(1)
PY

log "v15 data pipeline complete"
