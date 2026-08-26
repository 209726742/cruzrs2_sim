#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$PACKAGE_ROOT/.." && pwd)
PYTHON_BIN=${SORTING_ROLL_PYTHON:-$PROJECT_ROOT/envs/mjx/bin/python}
OUTPUT_ROOT=
GPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --gpu)
      GPU=$2
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "usage: $0 --out-root PATH [--gpu ID]" >&2
  exit 2
fi

OUTPUT_ROOT=$(realpath -m "$OUTPUT_ROOT")
MANIFEST=$OUTPUT_ROOT/campaign_manifest_with_replacements.json
SOURCES=$OUTPUT_ROOT/selected_sources.txt
REVIEW_ROOT=$OUTPUT_ROOT/review_bundle
mkdir -p "$REVIEW_ROOT"

selection=$("$PYTHON_BIN" - "$SOURCES" <<'PY'
import json
from pathlib import Path
import sys

sources = [
    Path(line.strip())
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
candidates = []
for source in sources:
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    assignment = result["diversity"]["assignment"]
    score = (
        assignment["object_profile"]["name"] == "short_slim",
        assignment["pose_bin"] == "boundary",
        assignment["split"] == "test",
    )
    candidates.append((score, -int(result["seed"]), int(result["seed"]), source))
_, _, seed, source = max(candidates)
print(seed)
print(source)
PY
)
seed=${selection%%$'\n'*}
source_episode=${selection#*$'\n'}
episode=$REVIEW_ROOT/seed_$(printf '%04d' "$seed")

if [[ ! -f "$episode/result.json" ]]; then
  if [[ -e "$episode" ]]; then
    echo "incomplete review episode exists: $episode" >&2
    exit 1
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/sorting_roll_expert.py" \
    --out "$episode" \
    --seed "$seed" \
    --gpu "$GPU" \
    --randomize \
    --manifest "$MANIFEST" \
    --review-videos \
    >"$REVIEW_ROOT/review_seed_${seed}.log" 2>&1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/sorting_roll_validate.py" \
  "$episode" \
  --manifest "$MANIFEST" \
  --report "$REVIEW_ROOT/validator_report.json"

"$PYTHON_BIN" - "$REVIEW_ROOT/review_bundle.json" "$episode" "$source_episode" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
episode = Path(sys.argv[2]).resolve()
source = Path(sys.argv[3]).resolve()
result = json.loads((episode / "result.json").read_text(encoding="utf-8"))
payload = {
    "schema_version": 1,
    "task_version": "sorting_roll_v15_diverse_sim",
    "seed": int(result["seed"]),
    "source_episode": str(source),
    "review_episode": str(episode),
    "diversity_assignment": result["diversity"]["assignment"],
    "success": result["success"],
    "sim_seconds": result["sim_seconds"],
    "final_evidence": result["final_evidence"],
    "third_person_video": result["review_video"],
    "robot_multiview_video": result["robot_multiview_video"],
    "slot_visual_video": result["slot_visual_review_video"],
    "slot_physics_video": result["slot_physics_review_video"],
    "passed": result["success"] is True,
}
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps(payload, indent=2))
PY
