#!/usr/bin/env bash
# Safe sharded batch collector for the dual-material shelf expert.
set -u -o pipefail

PKG=$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")
cd "$PKG"

usage() {
  cat <<'EOF'
Usage:
  shelf_e2e_batch.sh [options]
  shelf_e2e_batch.sh <target-success> <workers> <gpu-id>  # legacy form

Options:
  --target-success N   Successful source episodes required in this shard (default: 300)
  --workers N          Concurrent simulator workers on this GPU (default: 6)
  --gpu-id N           EGL GPU index (default: 0)
  --seed-start N       First seed for this shard (default: 1)
  --seed-end N         Optional inclusive final seed
  --seed-stride N      Increment between seeds (default: 1)
  --max-attempts N     Stop after N scheduled seeds (default: 4 * target-success)
  --run-id ID          Unique shard id; characters: A-Z a-z 0-9 . _ -
  --output-shard PATH  Episode root (default: out/teleop/shelf_e2e_dual)
  --log-shard PATH     Log root (default: out/logs/shelf_e2e_dual)
  --timeout N          Seconds allowed per expert run (default: 1400)
  --collection-profile PROFILE
                       strict_v1 or sdk_recovery_v1 (default: strict_v1)
  --diversity-mode MODE
                       clean or recovery (default: clean); recovery applies one
                       bounded base-pose shift during empty-gripper navigation
  --layout-mode MODE   random or boundary (default: random); boundary puts one
                       layout axis in the outer 20% of its validated range
  --resume             Validate and count already-published episodes for this run-id
  -h, --help           Show this help

Eight-GPU seed example: GPU i uses --gpu-id i --seed-start $((i+1)) --seed-stride 8.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

TARGET_SUCCESS=300
WORKERS=6
GPU_ID=0
SEED_START=1
SEED_END=""
SEED_STRIDE=1
MAX_ATTEMPTS=""
RUN_ID=""
OUTPUT_SHARD="out/teleop/shelf_e2e_dual"
LOG_SHARD="out/logs/shelf_e2e_dual"
TIMEOUT_SECONDS=1400
COLLECTION_PROFILE="strict_v1"
DIVERSITY_MODE="clean"
LAYOUT_MODE="random"
RESUME=0

if [[ ${1:-} && ${1:-} != -* ]]; then
  TARGET_SUCCESS=${1:-300}
  WORKERS=${2:-6}
  GPU_ID=${3:-0}
  [[ $# -le 3 ]] || die "legacy form accepts at most three arguments"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --target-success) [[ $# -ge 2 ]] || die "$1 needs a value"; TARGET_SUCCESS=$2; shift 2 ;;
      --workers) [[ $# -ge 2 ]] || die "$1 needs a value"; WORKERS=$2; shift 2 ;;
      --gpu-id) [[ $# -ge 2 ]] || die "$1 needs a value"; GPU_ID=$2; shift 2 ;;
      --seed-start) [[ $# -ge 2 ]] || die "$1 needs a value"; SEED_START=$2; shift 2 ;;
      --seed-end) [[ $# -ge 2 ]] || die "$1 needs a value"; SEED_END=$2; shift 2 ;;
      --seed-stride) [[ $# -ge 2 ]] || die "$1 needs a value"; SEED_STRIDE=$2; shift 2 ;;
      --max-attempts) [[ $# -ge 2 ]] || die "$1 needs a value"; MAX_ATTEMPTS=$2; shift 2 ;;
      --run-id) [[ $# -ge 2 ]] || die "$1 needs a value"; RUN_ID=$2; shift 2 ;;
      --output-shard) [[ $# -ge 2 ]] || die "$1 needs a value"; OUTPUT_SHARD=$2; shift 2 ;;
      --log-shard) [[ $# -ge 2 ]] || die "$1 needs a value"; LOG_SHARD=$2; shift 2 ;;
      --timeout) [[ $# -ge 2 ]] || die "$1 needs a value"; TIMEOUT_SECONDS=$2; shift 2 ;;
      --collection-profile) [[ $# -ge 2 ]] || die "$1 needs a value"; COLLECTION_PROFILE=$2; shift 2 ;;
      --diversity-mode) [[ $# -ge 2 ]] || die "$1 needs a value"; DIVERSITY_MODE=$2; shift 2 ;;
      --layout-mode) [[ $# -ge 2 ]] || die "$1 needs a value"; LAYOUT_MODE=$2; shift 2 ;;
      --resume) RESUME=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown option: $1" ;;
    esac
  done
fi

for value_name in TARGET_SUCCESS WORKERS GPU_ID SEED_START SEED_STRIDE TIMEOUT_SECONDS; do
  value=${!value_name}
  [[ $value =~ ^[0-9]+$ ]] || die "$value_name must be a non-negative integer"
done
(( TARGET_SUCCESS > 0 )) || die "TARGET_SUCCESS must be positive"
(( WORKERS > 0 )) || die "WORKERS must be positive"
(( SEED_START > 0 )) || die "SEED_START must be positive"
(( SEED_STRIDE > 0 )) || die "SEED_STRIDE must be positive"
(( TIMEOUT_SECONDS > 0 )) || die "TIMEOUT_SECONDS must be positive"
case "$COLLECTION_PROFILE" in
  strict_v1|sdk_recovery_v1) ;;
  *) die "unsupported collection profile: $COLLECTION_PROFILE" ;;
esac
case "$DIVERSITY_MODE" in
  clean) KICK_COUNT=0 ;;
  recovery) KICK_COUNT=1 ;;
  *) die "unsupported diversity mode: $DIVERSITY_MODE" ;;
esac
case "$LAYOUT_MODE" in
  random|boundary) ;;
  *) die "unsupported layout mode: $LAYOUT_MODE" ;;
esac
if [[ -n $SEED_END ]]; then
  [[ $SEED_END =~ ^[0-9]+$ ]] || die "SEED_END must be a non-negative integer"
  (( SEED_END >= SEED_START )) || die "SEED_END must be >= SEED_START"
fi
if [[ -z $MAX_ATTEMPTS ]]; then
  MAX_ATTEMPTS=$((TARGET_SUCCESS * 4))
fi
[[ $MAX_ATTEMPTS =~ ^[0-9]+$ ]] || die "MAX_ATTEMPTS must be a non-negative integer"
(( MAX_ATTEMPTS > 0 )) || die "MAX_ATTEMPTS must be positive"

if [[ -z $RUN_ID ]]; then
  RUN_ID="gpu${GPU_ID}_s${SEED_START}_k${SEED_STRIDE}"
fi
[[ $RUN_ID =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_ID contains unsupported characters"

case "$OUTPUT_SHARD" in /*) ;; *) OUTPUT_SHARD="$PKG/$OUTPUT_SHARD" ;; esac
case "$LOG_SHARD" in /*) ;; *) LOG_SHARD="$PKG/$LOG_SHARD" ;; esac
OUTPUT_SHARD=$(readlink -m "$OUTPUT_SHARD")
LOG_SHARD=$(readlink -m "$LOG_SHARD")
mkdir -p "$OUTPUT_SHARD" "$LOG_SHARD"

MJ=${RL_MJX_PY:-$PKG/../envs/mjx/bin/python}
[[ -x $MJ ]] || die "MuJoCo Python is not executable: $MJ"
VALIDATOR="$PKG/scripts/collection/shelf_e2e_source.py"
EXPERT="$PKG/scripts/collection/shelf_e2e_dual_expert.py"
[[ -f $VALIDATOR && -f $EXPERT ]] || die "expert or validator script is missing"

BATCH_LOG="$LOG_SHARD/${RUN_ID}.log"
LOCK_DIR="$LOG_SHARD/.${RUN_ID}.lock"
REJECT_SHARD="$OUTPUT_SHARD/.rejected/$RUN_ID"
if [[ -e $BATCH_LOG && $RESUME -ne 1 ]]; then
  die "batch log already exists; choose another --run-id or use --resume: $BATCH_LOG"
fi
mkdir "$LOCK_DIR" 2>/dev/null || die "run-id is already active: $RUN_ID"
mkdir -p "$REJECT_SHARD"
cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
touch "$BATCH_LOG"

log() {
  echo "$*" | tee -a "$BATCH_LOG"
}

run_one() {
  local seed=$1 attempt=$2
  local stem="${RUN_ID}_seed_$(printf '%06d' "$seed")"
  local final="$OUTPUT_SHARD/$stem"
  local run_log="$LOG_SHARD/${stem}_attempt_${attempt}_${BASHPID}.log"
  local validation_log="$LOG_SHARD/${stem}_resume_validation.json"

  if [[ -e $final ]]; then
    if [[ $RESUME -eq 1 && -d $final ]] && "$MJ" -B "$VALIDATOR" "$final" \
        --expected-seed "$seed" --expected-profile "$COLLECTION_PROFILE" \
        --expected-diversity-mode "$DIVERSITY_MODE" \
        --expected-layout-mode "$LAYOUT_MODE" \
        > "$validation_log" 2>&1; then
      printf 'RESUME_PASS seed=%d dir=%s\n' "$seed" "$final" >> "$BATCH_LOG"
      return 0
    fi
    printf 'EXISTING_INVALID seed=%d dir=%s\n' "$seed" "$final" >> "$BATCH_LOG"
    return 1
  fi

  local temp="$OUTPUT_SHARD/.${stem}.tmp_${attempt}_${BASHPID}"
  local rejected="$REJECT_SHARD/${stem}_attempt_${attempt}_${BASHPID}"
  mkdir "$temp" || {
    printf 'TEMP_FAIL seed=%d dir=%s\n' "$seed" "$temp" >> "$BATCH_LOG"
    return 1
  }

  # Clean and recovery episodes use separate run-ids/campaigns. Recovery applies
  # one bounded shift before grasping; source validation requires the actual event.
  SEED=$seed E2E_NOREC=0 E2E_KICKS=$KICK_COUNT \
    E2E_DIVERSITY_MODE="$DIVERSITY_MODE" E2E_LAYOUT_MODE="$LAYOUT_MODE" \
    E2E_COLLECTION_PROFILE="$COLLECTION_PROFILE" \
    EXPERT_OUT="$temp" E2E_RUN_ID="$RUN_ID" \
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_ID TELEOP_RECORD_GPU=$GPU_ID \
    timeout --signal=TERM --kill-after=30 "$TIMEOUT_SECONDS" \
    "$MJ" -B "$EXPERT" > "$run_log" 2>&1
  local expert_rc=$?
  rm -f -- "$PKG/assets/e2e_dual_scene_${RUN_ID}_${seed}.xml"

  "$MJ" -B "$VALIDATOR" "$temp" --expected-seed "$seed" \
    --expected-profile "$COLLECTION_PROFILE" \
    --expected-diversity-mode "$DIVERSITY_MODE" \
    --expected-layout-mode "$LAYOUT_MODE" \
    > "$temp/source_validation.json" 2>&1
  local validation_rc=$?
  if [[ $expert_rc -eq 0 && $validation_rc -eq 0 ]]; then
    if [[ -e $final ]]; then
      mv -- "$temp" "$rejected"
      printf 'PUBLISH_RACE seed=%d preserved=%s\n' "$seed" "$rejected" >> "$BATCH_LOG"
      return 1
    fi
    if mv -- "$temp" "$final"; then
      printf 'PASS seed=%d dir=%s log=%s\n' "$seed" "$final" "$run_log" >> "$BATCH_LOG"
      return 0
    fi
  fi

  if [[ -d $temp ]]; then
    mv -- "$temp" "$rejected"
  fi
  printf 'DROP seed=%d expert_rc=%d validation_rc=%d preserved=%s log=%s\n' \
    "$seed" "$expert_rc" "$validation_rc" "$rejected" "$run_log" >> "$BATCH_LOG"
  return 1
}

log "=== dual batch run=$RUN_ID profile=$COLLECTION_PROFILE diversity=$DIVERSITY_MODE layout=$LAYOUT_MODE target=$TARGET_SUCCESS workers=$WORKERS gpu=$GPU_ID seed_start=$SEED_START stride=$SEED_STRIDE seed_end=${SEED_END:-none} ==="
success=0
failed=0
attempts=0
next_seed=$SEED_START

while (( success < TARGET_SUCCESS && attempts < MAX_ATTEMPTS )); do
  pids=()
  batch_seeds=()
  for ((worker=0; worker<WORKERS; worker++)); do
    if [[ -n $SEED_END ]] && (( next_seed > SEED_END )); then
      break
    fi
    attempts=$((attempts + 1))
    run_one "$next_seed" "$attempts" &
    pids+=("$!")
    batch_seeds+=("$next_seed")
    next_seed=$((next_seed + SEED_STRIDE))
    (( attempts >= MAX_ATTEMPTS )) && break
  done
  (( ${#pids[@]} > 0 )) || break

  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      success=$((success + 1))
    else
      failed=$((failed + 1))
    fi
  done
  log "progress run=$RUN_ID pass=$success/$TARGET_SUCCESS failed=$failed attempts=$attempts next_seed=$next_seed"
done

total=$((success + failed))
rate=0
(( total > 0 )) && rate=$((success * 100 / total))
log "=== dual batch complete run=$RUN_ID PASS=$success DROP=$failed rate=${rate}% attempts=$attempts ==="
if (( success < TARGET_SUCCESS )); then
  log "INCOMPLETE: target not reached; inspect $BATCH_LOG and $REJECT_SHARD"
  exit 1
fi
log "E2E_BATCH_DONE"
