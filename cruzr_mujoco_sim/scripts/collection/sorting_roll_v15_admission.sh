#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$PACKAGE_ROOT/.." && pwd)
PYTHON_BIN=${SORTING_ROLL_PYTHON:-$PROJECT_ROOT/envs/mjx/bin/python}
GPU=0
OUTPUT_ROOT=
GROUPS_CSV=
FINALIZE=true
FINALIZE_ONLY=false
SKIP_SCENE_CHECK=false

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
    --groups)
      GROUPS_CSV=$2
      shift 2
      ;;
    --no-final-report)
      FINALIZE=false
      shift
      ;;
    --finalize-only)
      FINALIZE_ONLY=true
      shift
      ;;
    --skip-scene-check)
      SKIP_SCENE_CHECK=true
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "usage: $0 --out-root PATH [--gpu ID] [--groups CSV]" >&2
  exit 2
fi
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "--gpu must be a non-negative integer" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing Python environment: $PYTHON_BIN" >&2
  exit 2
fi

OUTPUT_ROOT=$(realpath -m "$OUTPUT_ROOT")
MANIFEST_ROOT=$OUTPUT_ROOT/manifests
GROUP_ROOT=$OUTPUT_ROOT/groups
mkdir -p "$MANIFEST_ROOT" "$GROUP_ROOT"
if [[ "$SKIP_SCENE_CHECK" != true && "$FINALIZE_ONLY" != true ]]; then
  bash "$PROJECT_ROOT/Sorting_Roll/run_scene.sh" check
fi

groups=(
  dynamics_heavy_low_friction
  dynamics_light_high_friction
  geometry_long
  geometry_medium
  geometry_short
)
declare -A seed_starts=(
  [geometry_short]=10000
  [geometry_medium]=10020
  [geometry_long]=10040
  [dynamics_light_high_friction]=10060
  [dynamics_heavy_low_friction]=10080
)

if [[ -n "$GROUPS_CSV" ]]; then
  IFS=',' read -r -a requested_groups <<<"$GROUPS_CSV"
  for group in "${requested_groups[@]}"; do
    if [[ -z "${seed_starts[$group]+x}" ]]; then
      echo "unknown admission group: $group" >&2
      exit 2
    fi
  done
  groups=("${requested_groups[@]}")
fi

if [[ "$FINALIZE_ONLY" != true ]]; then
for group in "${groups[@]}"; do
  seed_start=${seed_starts[$group]}
  manifest=$MANIFEST_ROOT/$group.json
  if [[ -f "$manifest" ]]; then
    "$PYTHON_BIN" "$PACKAGE_ROOT/scripts/core/sorting_roll_diversity.py" \
      check "$manifest" >/dev/null
  else
    "$PYTHON_BIN" "$PACKAGE_ROOT/scripts/core/sorting_roll_diversity.py" \
      generate \
      --out "$manifest" \
      --campaign "$(basename "$OUTPUT_ROOT")_$group" \
      --seed-start "$seed_start" \
      --count 20 \
      --admission-group "$group" \
      >"$MANIFEST_ROOT/$group.counts.log"
  fi

  "$PYTHON_BIN" -c '
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected = (sys.argv[2], int(sys.argv[3]), 20)
actual = (
    payload.get("admission_group"),
    payload.get("seed_start"),
    payload.get("count"),
)
if actual != expected:
    raise SystemExit(f"manifest contract {actual} != {expected}")
' "$manifest" "$group" "$seed_start"

  group_output=$GROUP_ROOT/$group
  echo "[admission] group=$group seed_start=$seed_start"
  "$PYTHON_BIN" "$PACKAGE_ROOT/scripts/collection/sorting_roll_batch.py" \
    --out-root "$group_output" \
    --seed-start "$seed_start" \
    --count 20 \
    --min-success 18 \
    --gpu "$GPU" \
    --timeout 1800 \
    --render \
    --resume \
    --manifest "$manifest"

  mapfile -t passed_sources < <(
    "$PYTHON_BIN" -c '
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
for record in summary["records"]:
    if record.get("passed"):
        print(record["episode"])
' "$group_output/summary.json"
  )
  if (( ${#passed_sources[@]} < 18 )); then
    echo "group $group has fewer than 18 successful episodes" >&2
    exit 1
  fi
  printf '%s\n' "${passed_sources[@]}" >"$group_output/passed_sources.txt"

  "$PYTHON_BIN" "$PACKAGE_ROOT/scripts/collection/sorting_roll_validate.py" \
    "${passed_sources[@]}" \
    --manifest "$manifest" \
    --report "$group_output/validator_report.json"

  mapfile -t audit_sources < <(
    "$PYTHON_BIN" -c '
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
records = [record for record in summary["records"] if record.get("passed")]
records.sort(key=lambda record: (
    (record.get("diversity") or {}).get("assignment", {}).get("pose_bin") != "boundary",
    record["seed"],
))
for record in records[:3]:
    print(record["episode"])
' "$group_output/summary.json"
  )
  if (( ${#audit_sources[@]} != 3 )); then
    echo "group $group does not have three successful audit sources" >&2
    exit 1
  fi
  for episode in "${audit_sources[@]}"; do
    audit_report=$episode/camera_observability.json
    "$PYTHON_BIN" \
      "$PACKAGE_ROOT/scripts/collection/sorting_roll_camera_audit.py" \
      --episode "$episode" \
      --gpu "$GPU" \
      --samples-per-phase 3 \
      --out "$audit_report" \
      >"$episode/camera_audit.log" 2>&1
    "$PYTHON_BIN" -c '
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
candidate = report["candidates"]["sorting_roll_d405_candidate_v6"]
if report.get("replay_diversity", {}).get("matches_recorded") is not True:
    raise SystemExit("audit diversity replay mismatch")
if candidate["coverage_fraction"] != 1.0:
    raise SystemExit("policy camera coverage is not 100%")
if candidate["required_role_coverage_fraction"] != 1.0:
    raise SystemExit("required camera-role coverage is not 100%")
' "$audit_report"
    for video in \
      sorting_roll_stereo_left.mp4 \
      sorting_roll_left_wrist_realsense.mp4 \
      sorting_roll_right_wrist_realsense.mp4 \
      sorting_roll_robot_multiview.mp4; do
      if [[ ! -s "$episode/$video" ]]; then
        echo "missing audit video: $episode/$video" >&2
        exit 1
      fi
    done
  done
  echo "[admission] group=$group PASS successes=${#passed_sources[@]}/20"
done
fi

if [[ "$FINALIZE" == true ]]; then
"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
groups = (
    "dynamics_heavy_low_friction",
    "dynamics_light_high_friction",
    "geometry_long",
    "geometry_medium",
    "geometry_short",
)
records = []
for group in groups:
    group_root = root / "groups" / group
    summary = json.loads((group_root / "summary.json").read_text())
    validator = json.loads((group_root / "validator_report.json").read_text())
    audits = sorted(group_root.glob("seed_*/camera_observability.json"))
    passed = (
        summary.get("completed_count") == 20
        and summary.get("success_count", 0) >= 18
        and summary.get("passed") is True
        and validator.get("passed") is True
        and validator.get("passed_count") == summary.get("success_count")
        and len(audits) >= 3
    )
    records.append({
        "group": group,
        "completed_count": summary.get("completed_count"),
        "success_count": summary.get("success_count"),
        "failed_count": summary.get("failed_count"),
        "validator_passed_count": validator.get("passed_count"),
        "camera_audit_count": len(audits),
        "passed": passed,
    })

report = {
    "schema_version": 1,
    "task_version": "sorting_roll_v15_diverse_sim",
    "required_successes_per_group": 18,
    "required_attempts_per_group": 20,
    "required_camera_audits_per_group": 3,
    "passed": all(record["passed"] for record in records),
    "groups": records,
}
output = root / "admission_report.json"
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise SystemExit(1)
PY
fi
