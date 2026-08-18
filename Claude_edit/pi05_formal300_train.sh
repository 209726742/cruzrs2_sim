#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# π0.5 正式训练启动器（CRUZR S2 双物料线 / 300-source LeRobot v3.0）
#
# 修订版 2026-08-18
#   - 修复 scheduler decay_steps 与 steps 不匹配（学习率永远退不到底）
#   - 优化器 / scheduler 超参改为显式传入并落盘，resume 可精确复现
#   - 新增有效 batch 下限保护，避免再次用有效 batch=4 跑长训
#   - 新增 sweep 动作，在多个 batch size 上实测显存与 step 时间
#   - 默认硬件配置改为 2×H100；4×RTX 4090 的旧配置见 usage
#   - resume 找不到完整 checkpoint 时不再静默从头开始
#   改动理由详见仓库根目录 训练参数修改说明_20260818.md
#
# 动作：
#   sweep         多个 batch size 各跑一次短训练，量显存和 step 时间（前台）
#   canary        短步数 DDP 冒烟测试（后台）
#   canary-resume 从 canary checkpoint 续训，验证恢复路径（后台）
#   start         启动正式训练（后台）
#   resume        从最新完整 checkpoint 恢复（后台）
#   plan          只打印配置和样本覆盖率估算，不启动任何进程
#   status        查看进程、最新完整 checkpoint 和日志
# =============================================================================

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)

# ---------- 路径 ----------
DATASET_ROOT=${DATASET_ROOT:-$SIM_ROOT/out/datasets/formal300_v24_lerobot_v30_20260817}
BASE_POLICY=${BASE_POLICY:-$PROJECT_ROOT/pretrained/pi05_base_remapped}
LOG_ROOT=${LOG_ROOT:-$PROJECT_ROOT/log}
ISAAC_PY=${ISAAC_PY:-/isaac-sim/python.sh}
REPO_ID=${REPO_ID:-formal/cruzr_shelf_v24_300source}

# RUN_TAG 让不同硬件 / 不同 batch 的实验落在不同目录，避免与旧的 10k 输出混淆。
RUN_TAG=${RUN_TAG:-h100x2}

CANARY_OUTPUT=${CANARY_OUTPUT:-$SIM_ROOT/out/training/pi05_formal300_canary_${RUN_TAG}}
FORMAL_OUTPUT=${FORMAL_OUTPUT:-$SIM_ROOT/out/training/pi05_formal300_${RUN_TAG}}
FORMAL_JOB_NAME=${FORMAL_JOB_NAME:-pi05_formal300_${RUN_TAG}}
CANARY_JOB_NAME=${CANARY_JOB_NAME:-pi05_formal300_canary_${RUN_TAG}}

FORMAL_LOG=$LOG_ROOT/${FORMAL_JOB_NAME}.log
CANARY_LOG=$LOG_ROOT/${CANARY_JOB_NAME}.log
FORMAL_PID_FILE=$LOG_ROOT/${FORMAL_JOB_NAME}.pid
CANARY_PID_FILE=$LOG_ROOT/${CANARY_JOB_NAME}.pid

# ---------- 硬件与 DataLoader ----------
# 默认针对 2×H100 80GB。BATCH_SIZE 是每卡 batch，有效 batch = BATCH_SIZE × NUM_PROCESSES。
GPU_IDS=${GPU_IDS:-0,1}
NUM_PROCESSES=${NUM_PROCESSES:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-8}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}

# ---------- 训练主参数 ----------
TARGET_STEPS=${TARGET_STEPS:-30000}
SAVE_FREQ=${SAVE_FREQ:-1000}
LOG_FREQ=${LOG_FREQ:-25}
SEED=${SEED:-1000}
CANARY_STEPS=${CANARY_STEPS:-20}
CANARY_RESUME_STEPS=${CANARY_RESUME_STEPS:-40}

# ---------- 优化器与 scheduler ----------
# DECAY_STEPS 默认跟随 TARGET_STEPS。旧版本固定用 LeRobot 默认的 30000，
# 而 steps 只有 10000，导致训练结束时学习率还停在峰值的约 80%。
LEARNING_RATE=${LEARNING_RATE:-2.5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
DECAY_STEPS=${DECAY_STEPS:-$TARGET_STEPS}

# ---------- 策略 ----------
DTYPE=${DTYPE:-bfloat16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
TRAIN_EXPERT_ONLY=${TRAIN_EXPERT_ONLY:-true}
N_ACTION_STEPS=${N_ACTION_STEPS:-50}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}

# ---------- 数据 ----------
VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}
IMAGE_TRANSFORMS=${IMAGE_TRANSFORMS:-false}
USE_IMAGENET_STATS=${USE_IMAGENET_STATS:-true}

# ---------- 保护开关 ----------
MIN_EFFECTIVE_BATCH=${MIN_EFFECTIVE_BATCH:-16}
ALLOW_SMALL_BATCH=${ALLOW_SMALL_BATCH:-0}
ALLOW_RESTART_FROM_BASE=${ALLOW_RESTART_FROM_BASE:-0}

# ---------- sweep ----------
SWEEP_BATCH_SIZES=${SWEEP_BATCH_SIZES:-"8 16 32"}
SWEEP_STEPS=${SWEEP_STEPS:-40}

SAVE_CHECKPOINT=true
MEM_SAMPLER_PID=''

usage() {
  cat <<'EOF'
用法：
  bash scripts/training/pi05_formal300_train.sh <动作>

动作：
  plan           打印配置、有效 batch 和训练帧覆盖率估算，不启动任何进程
  sweep          在多个 batch size 上各跑一次短训练，实测显存与 step 时间（前台，耗时较长）
  canary         启动一次短步数 DDP 冒烟测试（默认 20 步）
  canary-resume  从最新完整 canary checkpoint 续训（默认到第 40 步）
  start          从 pi05_base_remapped 启动正式训练
  resume         从最新完整正式 checkpoint 恢复
  status         查看进程、最新完整 checkpoint 和日志末尾

硬件配置（默认针对 2×H100 80GB）：
  GPU_IDS=0,1 NUM_PROCESSES=2 BATCH_SIZE=16 NUM_WORKERS=8

  回到旧的 4×RTX 4090 配置（注意有效 batch 只有 4，start 会被下限保护拦住）：
    GPU_IDS=0,1,2,3 NUM_PROCESSES=4 BATCH_SIZE=1 NUM_WORKERS=2 \
    MIN_EFFECTIVE_BATCH=4 RUN_TAG=rtx4090x4 \
    bash scripts/training/pi05_formal300_train.sh canary

关键环境变量：
  TARGET_STEPS=30000     目标步数
  DECAY_STEPS=$TARGET_STEPS  余弦退火跨度，默认跟随 TARGET_STEPS
  WARMUP_STEPS=1000      学习率 warmup 步数
  LEARNING_RATE=2.5e-5   峰值学习率
  SAVE_FREQ=1000         checkpoint 间隔（单个 checkpoint 约 11 GiB）
  MIN_EFFECTIVE_BATCH=16 有效 batch 下限；低于此值 start 会拒绝启动
  ALLOW_SMALL_BATCH=1    显式放行小 batch（不推荐，仅用于复现旧实验）
  RUN_TAG=h100x2         输出目录、日志和任务名的后缀
  SWEEP_BATCH_SIZES="8 16 32"  sweep 要试的每卡 batch 列表

典型流程：
  bash scripts/training/pi05_formal300_train.sh plan
  bash scripts/training/pi05_formal300_train.sh sweep
  BATCH_SIZE=<sweep 选出的值> bash scripts/training/pi05_formal300_train.sh canary
  BATCH_SIZE=<同一个值> bash scripts/training/pi05_formal300_train.sh start
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing directory: $1"
}

effective_batch() {
  printf '%d\n' "$((BATCH_SIZE * NUM_PROCESSES))"
}

validate_hardware_args() {
  local -a ids
  IFS=',' read -r -a ids <<< "$GPU_IDS"
  [[ ${#ids[@]} -eq $NUM_PROCESSES ]] ||
    die "GPU_IDS has ${#ids[@]} entries but NUM_PROCESSES=$NUM_PROCESSES"
  [[ $NUM_PROCESSES -gt 0 ]] || die "NUM_PROCESSES must be positive"
  [[ $BATCH_SIZE -gt 0 ]] || die "BATCH_SIZE must be positive"
  [[ $NUM_WORKERS -ge 0 ]] || die "NUM_WORKERS cannot be negative"

  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  local available id
  available=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',')
  for id in "${ids[@]}"; do
    [[ $id =~ ^[0-9]+$ ]] || die "invalid GPU id: $id"
    [[ ,$available == *,$id,* ]] || die "GPU id $id is not visible (visible: $available)"
  done
}

# 学习率排程自检。旧版本完全没有这一步，decay_steps 与 steps 的不匹配无人发现。
validate_schedule() {
  [[ $TARGET_STEPS -gt 0 ]] || die "TARGET_STEPS must be positive"
  [[ $DECAY_STEPS -gt 0 ]] || die "DECAY_STEPS must be positive"
  [[ $WARMUP_STEPS -ge 0 ]] || die "WARMUP_STEPS cannot be negative"
  (( WARMUP_STEPS < DECAY_STEPS )) ||
    die "WARMUP_STEPS=$WARMUP_STEPS 必须小于 DECAY_STEPS=$DECAY_STEPS"
  (( WARMUP_STEPS * 4 <= TARGET_STEPS )) ||
    warn "WARMUP_STEPS=$WARMUP_STEPS 超过 TARGET_STEPS=$TARGET_STEPS 的 1/4，warmup 占比偏高"
  if (( DECAY_STEPS != TARGET_STEPS )); then
    warn "DECAY_STEPS=$DECAY_STEPS 与 TARGET_STEPS=$TARGET_STEPS 不一致；"
    warn "  学习率不会在训练结束时退到底，最后阶段的精调收益会丢失。"
  fi
}

# 有效 batch 下限保护。上一轮 10k 训练的有效 batch 只有 4，
# 累计只见过约 1.2% 的训练帧，闭环 0 grasp。这个保护就是为了不再重复一次。
validate_effective_batch() {
  local eff
  eff=$(effective_batch)
  if (( eff < MIN_EFFECTIVE_BATCH )) && [[ $ALLOW_SMALL_BATCH != 1 ]]; then
    die "有效 batch=$eff（BATCH_SIZE=$BATCH_SIZE × NUM_PROCESSES=$NUM_PROCESSES）低于 MIN_EFFECTIVE_BATCH=$MIN_EFFECTIVE_BATCH。
  先跑 sweep 确定每卡能放下的 batch，再用该值启动；
  确实要用小 batch 复现旧实验时设置 ALLOW_SMALL_BATCH=1。"
  fi
}

validate_dataset() {
  require_dir "$DATASET_ROOT"
  require_file "$DATASET_ROOT/meta/info.json"
  require_file "$DATASET_ROOT/meta/stats.json"
  "$ISAAC_PY" -c '
import json, sys
info = json.load(open(sys.argv[1]))
assert info.get("total_source_episodes") == 300, info.get("total_source_episodes")
assert info.get("source_task_version") == "dual_two_trip_v1"
assert info.get("collection_profile") == "sdk_recovery_v1"
train = info.get("splits", {}).get("train")
assert isinstance(train, str) and train.startswith("0:"), train
' "$DATASET_ROOT/meta/info.json" || die "dataset readiness check failed"
}

train_episodes_json() {
  "$ISAAC_PY" -c '
import json, sys
info = json.load(open(sys.argv[1]))
start, stop = map(int, info["splits"]["train"].split(":"))
assert start == 0 and stop > start
print(json.dumps(list(range(start, stop)), separators=(",", ":")))
' "$DATASET_ROOT/meta/info.json"
}

dataset_frame_stats() {
  "$ISAAC_PY" -c '
import json, sys
info = json.load(open(sys.argv[1]))
total_frames = int(info.get("total_frames", 0))
total_episodes = max(int(info.get("total_episodes", 1)), 1)
start, stop = map(int, info["splits"]["train"].split(":"))
print(total_frames, total_episodes, stop - start)
' "$DATASET_ROOT/meta/info.json"
}

# 把"这次训练到底会见到多少数据"直接打在日志开头，
# 让样本覆盖率不足这类问题在启动时就可见，而不是训练两天后才发现。
print_plan() {
  local mode=$1 steps=$2 output=$3
  local eff samples total_frames total_ep train_ep est_train_frames coverage
  eff=$(effective_batch)
  samples=$((steps * eff))
  read -r total_frames total_ep train_ep < <(dataset_frame_stats)
  est_train_frames=$((total_frames * train_ep / total_ep))
  if (( est_train_frames > 0 )); then
    coverage=$(awk -v s="$samples" -v f="$est_train_frames" 'BEGIN { printf "%.1f", 100 * s / f }')
  else
    coverage="n/a"
  fi

  cat <<EOF
---------------- π0.5 训练配置（$mode） ----------------
数据集            : $DATASET_ROOT
基础策略          : $BASE_POLICY
输出目录          : $output
GPU / 进程        : $GPU_IDS / $NUM_PROCESSES
每卡 batch        : $BATCH_SIZE
有效 batch        : $eff
每进程 workers    : $NUM_WORKERS
目标步数          : $steps
学习率 / warmup   : $LEARNING_RATE / $WARMUP_STEPS
decay_steps       : $DECAY_STEPS
weight decay      : $WEIGHT_DECAY
grad clip         : $GRAD_CLIP_NORM
dtype             : $DTYPE
gradient ckpt     : $GRADIENT_CHECKPOINTING
仅训练 expert     : $TRAIN_EXPERT_ONLY
action chunk      : $N_ACTION_STEPS
推理采样步数      : $NUM_INFERENCE_STEPS
checkpoint 间隔   : $SAVE_FREQ
--- 数据覆盖估算 ---
train episodes    : $train_ep / $total_ep
train 帧数（估算）: $est_train_frames
本次累计样本      : $samples（$steps 步 × 有效 batch $eff）
训练帧覆盖率      : ${coverage}%
--------------------------------------------------------
EOF
}

checkpoint_is_complete() {
  local checkpoint=$1
  local expected_step=$2
  "$ISAAC_PY" -c '
import json, os, sys
from safetensors import safe_open
root, expected = sys.argv[1], int(sys.argv[2])
required_json = [
    "pretrained_model/config.json",
    "pretrained_model/policy_preprocessor.json",
    "pretrained_model/policy_postprocessor.json",
    "pretrained_model/train_config.json",
    "training_state/optimizer_param_groups.json",
    "training_state/scheduler_state.json",
    "training_state/training_step.json",
]
for rel in required_json:
    with open(os.path.join(root, rel)) as handle:
        json.load(handle)
step = json.load(open(os.path.join(root, "training_state/training_step.json")))["step"]
assert step == expected, (step, expected)
for rel in (
    "pretrained_model/model.safetensors",
    "training_state/optimizer_state.safetensors",
    "training_state/rng_state.safetensors",
):
    with safe_open(os.path.join(root, rel), framework="pt", device="cpu") as handle:
        assert len(list(handle.keys())) > 0, rel
' "$checkpoint" "$expected_step" >/dev/null 2>&1
}

latest_valid_checkpoint() {
  local output=$1
  local latest='' checkpoint name step
  shopt -s nullglob
  for checkpoint in "$output"/checkpoints/[0-9][0-9][0-9][0-9][0-9][0-9]; do
    [[ -d "$checkpoint" ]] || continue
    name=${checkpoint##*/}
    step=$((10#$name))
    if checkpoint_is_complete "$checkpoint" "$step"; then
      latest=$checkpoint
    fi
  done
  shopt -u nullglob
  [[ -n $latest ]] && printf '%s\n' "$latest"
}

checkpoint_step() {
  local name=${1##*/}
  printf '%d\n' "$((10#$name))"
}

load_checkpoint_loader_args() {
  local config=$1/pretrained_model/train_config.json
  local target=$2
  local values
  values=$("$ISAAC_PY" -c '
import json, sys
cfg = json.load(open(sys.argv[1]))
policy = cfg.get("policy", {})
print(
    int(cfg["batch_size"]),
    int(cfg["num_workers"]),
    int(policy.get("scheduler_decay_steps", 0)),
    policy.get("optimizer_lr", "n/a"),
)
' "$config")
  local ckpt_decay ckpt_lr
  read -r BATCH_SIZE NUM_WORKERS ckpt_decay ckpt_lr <<< "$values"
  printf 'restored loader config: batch_per_gpu=%s workers_per_gpu=%s lr=%s decay_steps=%s\n' \
    "$BATCH_SIZE" "$NUM_WORKERS" "$ckpt_lr" "$ckpt_decay"
  # resume 会沿用 checkpoint 里的 scheduler；如果这次想训得更长，
  # 排程跨度不会自动跟着变，学习率仍按原计划退火。
  if (( ckpt_decay > 0 )) && (( ckpt_decay != target )); then
    warn "checkpoint 的 scheduler_decay_steps=$ckpt_decay，本次目标步数=$target。"
    warn "  resume 沿用 checkpoint 排程，学习率不会按新的目标步数退火。"
    warn "  要按新跨度退火，需要开一个新的输出目录从基础权重重训。"
  fi
}

archive_checkpoints_after() {
  local output=$1
  local keep_step=$2
  local archive_root='' checkpoint name step
  shopt -s nullglob
  for checkpoint in "$output"/checkpoints/[0-9][0-9][0-9][0-9][0-9][0-9]; do
    [[ -d $checkpoint ]] || continue
    name=${checkpoint##*/}
    step=$((10#$name))
    if (( step > keep_step )); then
      if [[ -z $archive_root ]]; then
        archive_root="$output/interrupted_checkpoints_$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$archive_root"
      fi
      mv "$checkpoint" "$archive_root/"
    fi
  done
  shopt -u nullglob
  if [[ -n $archive_root ]]; then
    printf 'archived checkpoints newer than step %s to %s\n' "$keep_step" "$archive_root"
  fi
}

job_is_running() {
  local job_name=$1
  pgrep -f "[l]erobot.scripts.lerobot_train.*$job_name" >/dev/null
}

accelerate_prefix() {
  ACCELERATE_PREFIX=("$ISAAC_PY" -m accelerate.commands.launch)
  # 单进程时不能带 --multi_gpu，否则 accelerate 会拒绝启动。
  if (( NUM_PROCESSES > 1 )); then
    ACCELERATE_PREFIX+=(--multi_gpu)
  fi
  ACCELERATE_PREFIX+=(
    --num_processes "$NUM_PROCESSES"
    --gpu_ids "$GPU_IDS"
    --main_process_port "$MAIN_PROCESS_PORT"
    -m lerobot.scripts.lerobot_train
  )
}

# 所有超参都显式传入，让 train_config.json 完整记录本次实验，
# 而不是依赖 LeRobot 默认值——默认值会随版本变化，且 resume 时无从追溯。
fresh_command() {
  local output=$1
  local job_name=$2
  local steps=$3
  local save_freq=$4
  local episodes
  episodes=$(train_episodes_json)
  accelerate_prefix
  TRAIN_COMMAND=(
    "${ACCELERATE_PREFIX[@]}"
    --dataset.repo_id="$REPO_ID"
    --dataset.root="$DATASET_ROOT"
    --dataset.episodes="$episodes"
    --dataset.video_backend="$VIDEO_BACKEND"
    --dataset.image_transforms.enable="$IMAGE_TRANSFORMS"
    --dataset.use_imagenet_stats="$USE_IMAGENET_STATS"
    --policy.path="$BASE_POLICY"
    --policy.device=cuda
    --policy.dtype="$DTYPE"
    --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING"
    --policy.train_expert_only="$TRAIN_EXPERT_ONLY"
    --policy.optimizer_lr="$LEARNING_RATE"
    --policy.optimizer_weight_decay="$WEIGHT_DECAY"
    --policy.optimizer_grad_clip_norm="$GRAD_CLIP_NORM"
    --policy.scheduler_warmup_steps="$WARMUP_STEPS"
    --policy.scheduler_decay_steps="$DECAY_STEPS"
    --policy.n_action_steps="$N_ACTION_STEPS"
    --policy.num_inference_steps="$NUM_INFERENCE_STEPS"
    --policy.push_to_hub=false
    --output_dir="$output"
    --job_name="$job_name"
    --resume=false
    --batch_size="$BATCH_SIZE"
    --num_workers="$NUM_WORKERS"
    --steps="$steps"
    --save_checkpoint="$SAVE_CHECKPOINT"
    --save_freq="$save_freq"
    --eval_freq=0
    --log_freq="$LOG_FREQ"
    --seed="$SEED"
    --wandb.enable=false
  )
}

resume_command() {
  local checkpoint=$1
  local output=$2
  local job_name=$3
  local steps=$4
  local save_freq=$5
  accelerate_prefix
  TRAIN_COMMAND=(
    "${ACCELERATE_PREFIX[@]}"
    --config_path="$checkpoint/pretrained_model/train_config.json"
    --resume=true
    --output_dir="$output"
    --job_name="$job_name"
    --steps="$steps"
    --save_freq="$save_freq"
    --eval_freq=0
    --log_freq="$LOG_FREQ"
    --wandb.enable=false
  )
}

# ---------- 显存采样 ----------
start_memory_sampler() {
  local out=$1
  : > "$out"
  (
    while true; do
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$out" 2>/dev/null || true
      sleep 5
    done
  ) &
  MEM_SAMPLER_PID=$!
}

stop_memory_sampler() {
  [[ -n ${MEM_SAMPLER_PID:-} ]] || return 0
  kill "$MEM_SAMPLER_PID" 2>/dev/null || true
  wait "$MEM_SAMPLER_PID" 2>/dev/null || true
  MEM_SAMPLER_PID=''
}

report_peak_memory() {
  local samples=$1
  if [[ ! -s $samples ]]; then
    printf '    peak memory: (no samples)\n'
    return
  fi
  awk -F', *' '{ if ($2 + 0 > max[$1]) max[$1] = $2 + 0 }
       END { for (g in max) printf "    GPU %s peak: %d MiB\n", g, max[g] }' "$samples" | sort
}

report_step_timing() {
  local log=$1
  if grep -qE 'updt_s|update' "$log" 2>/dev/null; then
    grep -E 'updt_s|update' "$log" | tail -n 3 | sed 's/^/    /'
  else
    tail -n 3 "$log" | sed 's/^/    /'
  fi
}

# ---------- 启动 ----------
launch_detached() {
  local log_path=$1
  local pid_file=$2
  local mode=$3
  install -d "$LOG_ROOT"
  if [[ $mode == fresh && -e $log_path ]]; then
    mv "$log_path" "$log_path.previous_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  printf '[%s] launch mode=%s gpu_ids=%s processes=%s batch_per_gpu=%s effective_batch=%s workers_per_gpu=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$mode" "$GPU_IDS" "$NUM_PROCESSES" \
    "$BATCH_SIZE" "$(effective_batch)" "$NUM_WORKERS" >> "$log_path"

  cd "$PROJECT_ROOT"
  nohup setsid bash -c '
    "$@"
    rc=$?
    printf "[%s] train command exited rc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc"
    exit "$rc"
  ' _ env \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    "${TRAIN_COMMAND[@]}" >> "$log_path" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$pid_file"
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    printf 'training command exited immediately; log tail:\n' >&2
    tail -n 20 "$log_path" >&2
    return 1
  fi
  printf 'launched pid=%s\nlog=%s\n' "$pid" "$log_path"
}

run_foreground() {
  local log_path=$1
  install -d "$LOG_ROOT"
  (
    cd "$PROJECT_ROOT"
    env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
      "${TRAIN_COMMAND[@]}"
  ) >> "$log_path" 2>&1
}

# ---------- 动作实现 ----------
run_plan() {
  validate_dataset
  validate_schedule
  print_plan plan "$TARGET_STEPS" "$FORMAL_OUTPUT"
  local eff
  eff=$(effective_batch)
  if (( eff < MIN_EFFECTIVE_BATCH )); then
    warn "有效 batch=$eff 低于 MIN_EFFECTIVE_BATCH=$MIN_EFFECTIVE_BATCH，start 会被拒绝。"
  fi
}

# sweep 每档都是一次完整冷启动（数据集索引 + 权重加载），单档约十几分钟，
# 三档合计接近一小时。想快速确认单个 batch 时用 SWEEP_BATCH_SIZES="16"。
run_sweep() {
  validate_dataset
  require_dir "$BASE_POLICY"
  install -d "$LOG_ROOT"

  local original_batch=$BATCH_SIZE
  local b out log samples rc job
  SAVE_CHECKPOINT=false

  printf '=== batch size sweep：每档 %s 步，不保存 checkpoint ===\n' "$SWEEP_STEPS"
  printf '进程数=%s，GPU=%s，每进程 workers=%s\n' "$NUM_PROCESSES" "$GPU_IDS" "$NUM_WORKERS"

  for b in $SWEEP_BATCH_SIZES; do
    BATCH_SIZE=$b
    validate_hardware_args
    job=pi05_sweep_${RUN_TAG}_b${b}
    out=$SIM_ROOT/out/training/$job
    log=$LOG_ROOT/$job.log
    samples=$LOG_ROOT/$job.mem
    rm -rf "$out"
    : > "$log"

    printf '\n--- 每卡 batch=%s（有效 batch=%s）---\n' "$b" "$((b * NUM_PROCESSES))"
    fresh_command "$out" "$job" "$SWEEP_STEPS" "$SWEEP_STEPS"
    start_memory_sampler "$samples"
    rc=0
    run_foreground "$log" || rc=$?
    stop_memory_sampler

    if (( rc != 0 )); then
      printf '  结果：失败（rc=%s）\n' "$rc"
      if grep -qiE 'out of memory|CUDA out of memory' "$log"; then
        printf '  判定：显存不足，更大的 batch 不必再试\n'
      else
        printf '  判定：非 OOM 失败，请查看日志：%s\n' "$log"
      fi
      grep -iE 'out of memory|CUDA error|Traceback' "$log" | tail -n 3 | sed 's/^/    /' || true
      rm -rf "$out"
      break
    fi

    printf '  结果：通过\n'
    report_peak_memory "$samples"
    report_step_timing "$log"
    rm -rf "$out"
  done

  BATCH_SIZE=$original_batch
  SAVE_CHECKPOINT=true
  printf '\nsweep 完成。挑一个峰值显存留有余量（建议 ≤ 单卡显存的 85%%）且 step 时间可接受的 batch，\n'
  printf '然后用 BATCH_SIZE=<该值> 依次执行 canary 和 start。\n'
}

start_canary() {
  validate_dataset
  validate_hardware_args
  validate_schedule
  require_dir "$BASE_POLICY"
  job_is_running "$CANARY_JOB_NAME" && die "canary is already running"

  local latest
  latest=$(latest_valid_checkpoint "$CANARY_OUTPUT" || true)
  if [[ -n $latest ]] && [[ $(checkpoint_step "$latest") -ge $CANARY_STEPS ]]; then
    printf 'canary already complete at %s\n' "$latest"
    printf '换硬件或换 batch 后要重新验证时，请改 RUN_TAG 或删除该目录。\n'
    return
  fi
  [[ ! -e $CANARY_OUTPUT ]] || die "canary output exists without a complete target checkpoint: $CANARY_OUTPUT"

  print_plan canary "$CANARY_STEPS" "$CANARY_OUTPUT"
  fresh_command "$CANARY_OUTPUT" "$CANARY_JOB_NAME" "$CANARY_STEPS" "$CANARY_STEPS"
  launch_detached "$CANARY_LOG" "$CANARY_PID_FILE" fresh
}

resume_canary() {
  validate_dataset
  job_is_running "$CANARY_JOB_NAME" && die "canary is already running"
  local latest step
  latest=$(latest_valid_checkpoint "$CANARY_OUTPUT" || true)
  [[ -n $latest ]] || die "no complete canary checkpoint found"
  step=$(checkpoint_step "$latest")
  if [[ $step -ge $CANARY_RESUME_STEPS ]]; then
    printf 'canary resume target already reached: step=%s\n' "$step"
    return
  fi
  load_checkpoint_loader_args "$latest" "$CANARY_RESUME_STEPS"
  validate_hardware_args
  archive_checkpoints_after "$CANARY_OUTPUT" "$step"
  resume_command "$latest" "$CANARY_OUTPUT" "$CANARY_JOB_NAME" "$CANARY_RESUME_STEPS" "$CANARY_STEPS"
  launch_detached "$CANARY_LOG" "$CANARY_PID_FILE" resume
}

start_formal() {
  validate_dataset
  validate_hardware_args
  validate_schedule
  validate_effective_batch
  require_dir "$BASE_POLICY"
  job_is_running "$FORMAL_JOB_NAME" && die "formal training is already running"
  [[ ! -e $FORMAL_OUTPUT ]] || die "formal output already exists; use resume: $FORMAL_OUTPUT"

  print_plan start "$TARGET_STEPS" "$FORMAL_OUTPUT"
  fresh_command "$FORMAL_OUTPUT" "$FORMAL_JOB_NAME" "$TARGET_STEPS" "$SAVE_FREQ"
  {
    printf '[%s] plan:\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    print_plan start "$TARGET_STEPS" "$FORMAL_OUTPUT"
  } >> "$FORMAL_LOG" 2>/dev/null || true
  launch_detached "$FORMAL_LOG" "$FORMAL_PID_FILE" fresh
}

resume_formal() {
  validate_dataset
  job_is_running "$FORMAL_JOB_NAME" && die "formal training is already running"
  local latest step archive
  latest=$(latest_valid_checkpoint "$FORMAL_OUTPUT" || true)

  if [[ -z $latest ]]; then
    # 旧版本在这里会静默归档并从基础权重重新开始。长训中途这样做会
    # 无声地丢掉全部进度，因此改为默认拒绝，必须显式放行。
    if [[ -e $FORMAL_OUTPUT ]]; then
      [[ $ALLOW_RESTART_FROM_BASE == 1 ]] || die \
        "输出目录存在但没有完整 checkpoint：$FORMAL_OUTPUT
  先确认是否真的没有可用进度（checkpoints/ 下是否有半成品目录）。
  确认要丢弃并从基础权重重训时，设置 ALLOW_RESTART_FROM_BASE=1 再执行 resume。"
      archive=$FORMAL_OUTPUT.interrupted_$(date -u +%Y%m%dT%H%M%SZ)
      mv "$FORMAL_OUTPUT" "$archive"
      printf 'no complete checkpoint; archived partial output to %s\n' "$archive"
    fi
    validate_hardware_args
    validate_schedule
    validate_effective_batch
    require_dir "$BASE_POLICY"
    print_plan restart "$TARGET_STEPS" "$FORMAL_OUTPUT"
    fresh_command "$FORMAL_OUTPUT" "$FORMAL_JOB_NAME" "$TARGET_STEPS" "$SAVE_FREQ"
    launch_detached "$FORMAL_LOG" "$FORMAL_PID_FILE" fresh
    return
  fi

  step=$(checkpoint_step "$latest")
  if [[ $step -ge $TARGET_STEPS ]]; then
    printf 'formal training already reached target: step=%s\n' "$step"
    return
  fi
  load_checkpoint_loader_args "$latest" "$TARGET_STEPS"
  validate_hardware_args
  archive_checkpoints_after "$FORMAL_OUTPUT" "$step"
  printf 'resuming formal training from %s\n' "$latest"
  resume_command "$latest" "$FORMAL_OUTPUT" "$FORMAL_JOB_NAME" "$TARGET_STEPS" "$SAVE_FREQ"
  launch_detached "$FORMAL_LOG" "$FORMAL_PID_FILE" resume
}

show_job_status() {
  local label=$1
  local job_name=$2
  local output=$3
  local log_path=$4
  local latest
  printf '%s:\n' "$label"
  if job_is_running "$job_name"; then
    pgrep -af "[l]erobot.scripts.lerobot_train.*$job_name"
  else
    printf '  process: not running\n'
  fi
  latest=$(latest_valid_checkpoint "$output" || true)
  if [[ -n $latest ]]; then
    printf '  latest complete checkpoint: %s\n' "$latest"
  else
    printf '  latest complete checkpoint: none\n'
  fi
  if [[ -f $log_path ]]; then
    printf '  log tail (%s):\n' "$log_path"
    tail -n 8 "$log_path" | sed 's/^/    /'
  fi
}

show_status() {
  show_job_status canary "$CANARY_JOB_NAME" "$CANARY_OUTPUT" "$CANARY_LOG"
  show_job_status formal "$FORMAL_JOB_NAME" "$FORMAL_OUTPUT" "$FORMAL_LOG"
}

trap stop_memory_sampler EXIT

main() {
  local action=${1:-}
  case "$action" in
    plan)          run_plan ;;
    sweep)         run_sweep ;;
    canary)        start_canary ;;
    canary-resume) resume_canary ;;
    start)         start_formal ;;
    resume)        resume_formal ;;
    status)        show_status ;;
    help|-h|--help) usage ;;
    *)             usage >&2; exit 2 ;;
  esac
}

main "$@"
