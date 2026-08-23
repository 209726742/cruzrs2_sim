#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MJX_PY=$PROJECT_ROOT/envs/mjx/bin/python
ISAAC_PY=${ISAAC_PY:-/isaac-sim/python.sh}
WAIT_SESSION=${WAIT_SESSION:-sorting_roll_v9_canary30_final}
SOURCE_ROOT=${SOURCE_ROOT:-$PROJECT_ROOT/cruzr_mujoco_sim/output/sorting_roll_expert/v9_d405_canary30_final_seed0200_0229}
DATASET_ROOT=${DATASET_ROOT:-$PROJECT_ROOT/cruzr_mujoco_sim/out/datasets/sorting_roll_d405_canary30_lerobot_v30_20260823}
DATASET_REPO_ID=${DATASET_REPO_ID:-local/sorting_roll_d405_canary30}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-$PROJECT_ROOT/cruzr_mujoco_sim/out/training/pi05_sorting_roll_d405_canary30_20step_20260823}
TRAIN_LOG=${TRAIN_LOG:-$PROJECT_ROOT/log/pi05_sorting_roll_d405_canary30_20step_20260823.log}
EXPECTED_EPISODES=${EXPECTED_EPISODES:-30}
MIN_INITIAL_SUCCESSES=${MIN_INITIAL_SUCCESSES:-27}
MAX_REPLACEMENT_ATTEMPTS=${MAX_REPLACEMENT_ATTEMPTS:-10}
REPLACEMENT_ROOT=${REPLACEMENT_ROOT:-$PROJECT_ROOT/cruzr_mujoco_sim/output/sorting_roll_expert/v9_d405_canary30_replacements}
POLL_SECONDS=${POLL_SECONDS:-30}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

log "waiting for tmux session $WAIT_SESSION"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

log "checking completed batch summary"
"$MJX_PY" - "$SOURCE_ROOT/summary.json" "$EXPECTED_EPISODES" \
  "$MIN_INITIAL_SUCCESSES" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected = int(sys.argv[2])
summary = json.loads(path.read_text(encoding="utf-8"))
actual = (
    summary.get("requested_count"),
    summary.get("completed_count"),
    summary.get("success_count"),
    summary.get("failed_count"),
    summary.get("passed"),
)
if actual[0] != expected or actual[1] != expected:
    raise SystemExit(f"batch completion gate failed: {actual}")
if actual[2] < int(sys.argv[3]):
    raise SystemExit(f"initial success-rate gate failed: {actual}")
if actual[3] != expected - actual[2]:
    raise SystemExit(f"batch success/failure counts are inconsistent: {actual}")
print(f"initial batch gate passed: {actual[2]}/{expected}")
PY

mapfile -t VALID_SOURCES < <(
  "$MJX_PY" - "$SOURCE_ROOT/summary.json" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for record in summary["records"]:
    if record.get("passed"):
        print(record["episode"])
PY
)

next_seed=230
replacement_attempts=0
while (( ${#VALID_SOURCES[@]} < EXPECTED_EPISODES )); do
  if (( replacement_attempts >= MAX_REPLACEMENT_ATTEMPTS )); then
    printf 'replacement gate failed: only %s/%s valid sources\n' \
      "${#VALID_SOURCES[@]}" "$EXPECTED_EPISODES" >&2
    exit 1
  fi
  attempt_root=$REPLACEMENT_ROOT/attempt_seed_$(printf '%04d' "$next_seed")
  test ! -e "$attempt_root" || {
    printf 'replacement output already exists: %s\n' "$attempt_root" >&2
    exit 1
  }
  log "collecting replacement seed $next_seed"
  if "$MJX_PY" \
    "$PROJECT_ROOT/cruzr_mujoco_sim/scripts/collection/sorting_roll_batch.py" \
    --out-root "$attempt_root" \
    --seed-start "$next_seed" \
    --count 1 \
    --min-success 1 \
    --gpu 0 \
    --timeout 1800 \
    --render; then
    VALID_SOURCES+=("$attempt_root/seed_$(printf '%04d' "$next_seed")")
  else
    log "replacement seed $next_seed failed and remains excluded"
  fi
  next_seed=$((next_seed + 1))
  replacement_attempts=$((replacement_attempts + 1))
done

"$MJX_PY" - "$SOURCE_ROOT/selected_sources.json" "${VALID_SOURCES[@]}" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
sources = [str(Path(value).resolve()) for value in sys.argv[2:]]
output.write_text(
    json.dumps({"count": len(sources), "sources": sources}, indent=2),
    encoding="utf-8",
)
print(f"selected source gate passed: {len(sources)} sources")
PY

log "validating all rendered source episodes"
"$MJX_PY" \
  "$PROJECT_ROOT/cruzr_mujoco_sim/scripts/collection/sorting_roll_validate.py" \
  "${VALID_SOURCES[@]}" \
  --report "$SOURCE_ROOT/validation_report.json"

log "building LeRobot v2.1 staging dataset"
"$MJX_PY" \
  "$PROJECT_ROOT/cruzr_mujoco_sim/scripts/collection/sorting_roll_build_v21.py" \
  "${VALID_SOURCES[@]}" \
  --out "$DATASET_ROOT" \
  --encode-workers 1

log "converting staging dataset to LeRobot v3.0"
cd "$PROJECT_ROOT"
PYTHONPATH=. "$ISAAC_PY" \
  src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
  --repo-id "$DATASET_REPO_ID" \
  --root "$DATASET_ROOT" \
  --push-to-hub false

log "loading v3.0 dataset and decoding boundary samples"
HF_HUB_OFFLINE=1 PYTHONPATH=. "$ISAAC_PY" - \
  "$DATASET_ROOT" "$DATASET_REPO_ID" "$EXPECTED_EPISODES" <<'PY'
import json
from pathlib import Path
import sys

from src.lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(sys.argv[1])
repo_id = sys.argv[2]
expected = int(sys.argv[3])
info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
stats = json.loads((root / "meta" / "stats.json").read_text(encoding="utf-8"))
if info.get("codebase_version") != "v3.0":
    raise SystemExit("dataset is not LeRobot v3.0")
if info.get("total_episodes") != expected or info.get("total_source_episodes") != expected:
    raise SystemExit("dataset episode count mismatch")
splits = info.get("splits") or {}
cursor = 0
for name in ("train", "val", "test"):
    if name not in splits:
        continue
    start, stop = map(int, splits[name].split(":"))
    if start != cursor or stop <= start:
        raise SystemExit(f"invalid contiguous split ranges: {splits}")
    cursor = stop
if "train" not in splits or cursor != expected:
    raise SystemExit(f"split ranges do not cover the dataset: {splits}")
dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
if dataset.num_episodes != expected or len(dataset) != info.get("total_frames"):
    raise SystemExit("LeRobotDataset length mismatch")
camera_keys = sorted(
    key for key in dataset.features if key.startswith("observation.images.")
)
if len(camera_keys) != 3:
    raise SystemExit(f"expected three policy cameras, got {camera_keys}")
for feature in ("observation.state", "action"):
    if not {"q01", "q99"}.issubset(stats.get(feature, {})):
        raise SystemExit(f"missing quantile stats for {feature}")
for index in (0, len(dataset) - 1):
    sample = dataset[index]
    if tuple(sample["observation.state"].shape) != (18,):
        raise SystemExit("state shape mismatch")
    if tuple(sample["action"].shape) != (18,):
        raise SystemExit("action shape mismatch")
    for key in camera_keys:
        if tuple(sample[key].shape) != (3, 224, 224):
            raise SystemExit(f"camera shape mismatch: {key}")
print(
    f"v3 audit passed: episodes={dataset.num_episodes} "
    f"frames={len(dataset)} cameras={camera_keys}"
)
PY

TRAIN_ARGS=(
  --dataset-root "$DATASET_ROOT"
  --repo-id "$DATASET_REPO_ID"
  --episodes train
  --base-policy "$PROJECT_ROOT/pretrained/pi05_base_remapped"
  --output-dir "$TRAIN_OUTPUT"
  --log-file "$TRAIN_LOG"
  --gpu-ids 0
  --num-processes 1
  --batch-size 1
  --num-workers 2
  --steps 20
  --warmup-steps 2
  --decay-steps 20
  --save-freq 20
  --log-freq 1
  --save-checkpoint false
  --allow-small-batch true
  --train-expert-only true
  --wandb false
  --offline true
)

log "checking the 20-step single-4090 training command"
bash "$PROJECT_ROOT/pi05_train.sh" dry-run "${TRAIN_ARGS[@]}"

log "starting the detached 20-step single-4090 training canary"
bash "$PROJECT_ROOT/pi05_train.sh" start "${TRAIN_ARGS[@]}"

TRAIN_PID=$(<"$TRAIN_LOG.pid")
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done
if ! tail -n 20 "$TRAIN_LOG" | grep -Fq "train command exited rc=0"; then
  tail -n 80 "$TRAIN_LOG"
  raise_message="training canary did not exit successfully"
  printf '%s\n' "$raise_message" >&2
  exit 1
fi

"$ISAAC_PY" - "$TRAIN_LOG" <<'PY'
import math
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
losses = [float(value) for value in re.findall(r"\bloss:([^\s]+)", text)]
gradients = [float(value) for value in re.findall(r"\bgrdn:([^\s]+)", text)]
if not losses or not gradients:
    raise SystemExit("training log has no loss/gradient measurements")
if not all(math.isfinite(value) for value in losses + gradients):
    raise SystemExit("training log contains non-finite loss/gradient")
print(
    f"training audit passed: measurements={len(losses)} "
    f"loss_range={min(losses):.6g}..{max(losses):.6g} "
    f"gradient_range={min(gradients):.6g}..{max(gradients):.6g}"
)
PY

log "pipeline complete"
bash "$PROJECT_ROOT/pi05_train.sh" status \
  --output-dir "$TRAIN_OUTPUT" \
  --log-file "$TRAIN_LOG"
