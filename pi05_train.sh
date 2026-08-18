#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# π0.5 一键训练脚本
#
# 最简单的用法：
#   bash pi05_train.sh dry-run   # 只检查并打印配置，不启动训练
#   bash pi05_train.sh           # 使用下方默认配置启动新训练
#   bash pi05_train.sh resume    # 从输出目录中最新的完整 checkpoint 续训
#   bash pi05_train.sh auto      # 没有输出则新训，有完整 checkpoint 则续训
#   bash pi05_train.sh status    # 查看后台进程、checkpoint 和日志
#
# 配置优先级：命令行参数 > 同名环境变量 > 下方默认值。
# ==============================================================================

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=$PROJECT_ROOT/cruzr_mujoco_sim

# ==============================================================================
# 用户常改配置区
# ==============================================================================

# 本地 LeRobot v3.0 数据集。当前训练代码一次只支持一个 dataset root/repo_id。
DATASET_ROOT=${DATASET_ROOT:-$SIM_ROOT/out/datasets/formal300_v24_lerobot_v30_20260817}
DATASET_REPO_ID=${DATASET_REPO_ID:-formal/cruzr_shelf_v24_300source}

# episode 选择器：all、train、val、test、split:<名称>、0:100、0,2,7 或 JSON 列表。
EPISODES=${EPISODES:-train}

# 初始策略可以是基础模型，也可以是某个 checkpoint 下的 pretrained_model 目录。
BASE_POLICY=${BASE_POLICY:-$PROJECT_ROOT/pretrained/pi05_base_remapped}

# 每个新实验必须使用独立输出目录，避免覆盖已有 checkpoint。
OUTPUT_DIR=${OUTPUT_DIR:-$SIM_ROOT/out/training/pi05_formal300_v24_10k_20260817}
JOB_NAME=${JOB_NAME:-}
LOG_FILE=${LOG_FILE:-}

# 多卡和 DataLoader 配置。4×H100 且仍用 GPU 0–3 时可以只调整 batch size。
GPU_IDS=${GPU_IDS:-0,1,2,3}
NUM_PROCESSES=${NUM_PROCESSES:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29500}

# 主要训练配置。
TARGET_STEPS=${TARGET_STEPS:-10000}
SAVE_FREQ=${SAVE_FREQ:-500}
LOG_FREQ=${LOG_FREQ:-10}
SEED=${SEED:-1000}
DTYPE=${DTYPE:-bfloat16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
TRAIN_EXPERT_ONLY=${TRAIN_EXPERT_ONLY:-true}
CUDNN_DETERMINISTIC=${CUDNN_DETERMINISTIC:-false}

# π0.5 优化器、scheduler 与动作配置。
LEARNING_RATE=${LEARNING_RATE:-2.5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
DECAY_STEPS=${DECAY_STEPS:-30000}
N_ACTION_STEPS=${N_ACTION_STEPS:-50}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}

# 数据与日志配置。
VIDEO_BACKEND=${VIDEO_BACKEND:-pyav}
IMAGE_TRANSFORMS=${IMAGE_TRANSFORMS:-false}
USE_IMAGENET_STATS=${USE_IMAGENET_STATS:-true}
WANDB_ENABLE=${WANDB_ENABLE:-false}
WANDB_PROJECT=${WANDB_PROJECT:-lerobot}
WANDB_MODE=${WANDB_MODE:-offline}
OFFLINE=${OFFLINE:-true}
ISAAC_PY=${ISAAC_PY:-/isaac-sim/python.sh}

ACTION=start
STEPS_OVERRIDDEN=false
SAVE_FREQ_OVERRIDDEN=false
LOG_FREQ_OVERRIDDEN=false
BATCH_SIZE_OVERRIDDEN=false
NUM_WORKERS_OVERRIDDEN=false
JOB_NAME_OVERRIDDEN=false

usage() {
  cat <<'EOF'
用法：
  bash pi05_train.sh [动作] [参数]

动作：
  start             启动全新的后台训练；无动作时默认执行 start
  resume            从 OUTPUT_DIR 最新的完整 checkpoint 继续训练
  auto              OUTPUT_DIR 不存在时 start；存在完整 checkpoint 时 resume
  status            查看进程、最新完整 checkpoint 和日志末尾
  dry-run           校验新训练配置并打印命令，不启动
  dry-run-resume    校验恢复配置并打印命令，不启动
  help              显示帮助

数据和路径：
  --dataset-root PATH       本地 LeRobot v3.0 数据集目录
  --repo-id ID              数据集逻辑名称，例如 local/my_dataset
  --episodes SELECTOR       all/train/val/test/split:name/0:100/0,2,7/[0,2,7]
  --base-policy PATH        基础 π0.5 或 pretrained_model 目录
  --output-dir PATH         checkpoint 最终输出目录
  --job-name NAME           训练任务名
  --log-file PATH           后台日志文件

硬件和 DataLoader：
  --gpu-ids 0,1,2,3
  --num-processes 4
  --batch-size 1
  --num-workers 2
  --port 29500

主要训练参数：
  --steps 10000
  --save-freq 500
  --log-freq 10
  --seed 1000
  --dtype bfloat16|float32
  --gradient-checkpointing true|false
  --train-expert-only true|false
  --deterministic true|false
  --learning-rate 2.5e-5
  --weight-decay 0.01
  --grad-clip-norm 1.0
  --warmup-steps 1000
  --decay-steps 30000
  --n-action-steps 50
  --num-inference-steps 10

数据处理和联网：
  --video-backend pyav
  --image-transforms true|false
  --use-imagenet-stats true|false
  --wandb true|false
  --wandb-project NAME
  --wandb-mode online|offline|disabled
  --offline true|false
  --isaac-python PATH

示例：
  # 当前正式 300-source 配置，只检查不启动
  bash pi05_train.sh dry-run

  # 自定义数据、训练步数和输出目录
  bash pi05_train.sh start \
    --dataset-root /path/to/lerobot_v30 \
    --repo-id local/my_dataset \
    --episodes train \
    --steps 20000 \
    --output-dir cruzr_mujoco_sim/out/training/my_pi05_run

  # 服务器重启后恢复；使用非默认输出目录时需再次给出同一路径
  bash pi05_train.sh resume --output-dir cruzr_mujoco_sim/out/training/my_pi05_run

注意：resume 默认使用 checkpoint 保存的数据集、policy、batch、worker、步数和 scheduler。
只有在 resume 命令中显式传入 --steps/--save-freq/--log-freq/--batch-size/
--num-workers 时，才会覆盖对应恢复值。
EOF
}

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

need_value() {
  [[ $# -ge 2 ]] || die "参数 $1 缺少值"
}

normalize_bool() {
  local name=$1
  local value=${2,,}
  case "$value" in
    true|1|yes|y|on) printf 'true\n' ;;
    false|0|no|n|off) printf 'false\n' ;;
    *) die "$name 必须是 true 或 false，当前值：$2" ;;
  esac
}

parse_args() {
  case "${1:-}" in
    start|resume|auto|status|dry-run|dry-run-resume)
      ACTION=$1
      shift
      ;;
    help|-h|--help)
      usage
      exit 0
      ;;
    '') ;;
    --*) ;;
    *) die "未知动作：$1" ;;
  esac

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset-root) need_value "$@"; DATASET_ROOT=$2; shift 2 ;;
      --repo-id) need_value "$@"; DATASET_REPO_ID=$2; shift 2 ;;
      --episodes) need_value "$@"; EPISODES=$2; shift 2 ;;
      --base-policy) need_value "$@"; BASE_POLICY=$2; shift 2 ;;
      --output-dir) need_value "$@"; OUTPUT_DIR=$2; shift 2 ;;
      --job-name) need_value "$@"; JOB_NAME=$2; JOB_NAME_OVERRIDDEN=true; shift 2 ;;
      --log-file) need_value "$@"; LOG_FILE=$2; shift 2 ;;
      --gpu-ids) need_value "$@"; GPU_IDS=$2; shift 2 ;;
      --num-processes) need_value "$@"; NUM_PROCESSES=$2; shift 2 ;;
      --batch-size) need_value "$@"; BATCH_SIZE=$2; BATCH_SIZE_OVERRIDDEN=true; shift 2 ;;
      --num-workers) need_value "$@"; NUM_WORKERS=$2; NUM_WORKERS_OVERRIDDEN=true; shift 2 ;;
      --port) need_value "$@"; MAIN_PROCESS_PORT=$2; shift 2 ;;
      --steps) need_value "$@"; TARGET_STEPS=$2; STEPS_OVERRIDDEN=true; shift 2 ;;
      --save-freq) need_value "$@"; SAVE_FREQ=$2; SAVE_FREQ_OVERRIDDEN=true; shift 2 ;;
      --log-freq) need_value "$@"; LOG_FREQ=$2; LOG_FREQ_OVERRIDDEN=true; shift 2 ;;
      --seed) need_value "$@"; SEED=$2; shift 2 ;;
      --dtype) need_value "$@"; DTYPE=$2; shift 2 ;;
      --gradient-checkpointing) need_value "$@"; GRADIENT_CHECKPOINTING=$2; shift 2 ;;
      --train-expert-only) need_value "$@"; TRAIN_EXPERT_ONLY=$2; shift 2 ;;
      --deterministic) need_value "$@"; CUDNN_DETERMINISTIC=$2; shift 2 ;;
      --learning-rate) need_value "$@"; LEARNING_RATE=$2; shift 2 ;;
      --weight-decay) need_value "$@"; WEIGHT_DECAY=$2; shift 2 ;;
      --grad-clip-norm) need_value "$@"; GRAD_CLIP_NORM=$2; shift 2 ;;
      --warmup-steps) need_value "$@"; WARMUP_STEPS=$2; shift 2 ;;
      --decay-steps) need_value "$@"; DECAY_STEPS=$2; shift 2 ;;
      --n-action-steps) need_value "$@"; N_ACTION_STEPS=$2; shift 2 ;;
      --num-inference-steps) need_value "$@"; NUM_INFERENCE_STEPS=$2; shift 2 ;;
      --video-backend) need_value "$@"; VIDEO_BACKEND=$2; shift 2 ;;
      --image-transforms) need_value "$@"; IMAGE_TRANSFORMS=$2; shift 2 ;;
      --use-imagenet-stats) need_value "$@"; USE_IMAGENET_STATS=$2; shift 2 ;;
      --wandb) need_value "$@"; WANDB_ENABLE=$2; shift 2 ;;
      --wandb-project) need_value "$@"; WANDB_PROJECT=$2; shift 2 ;;
      --wandb-mode) need_value "$@"; WANDB_MODE=$2; shift 2 ;;
      --offline) need_value "$@"; OFFLINE=$2; shift 2 ;;
      --isaac-python) need_value "$@"; ISAAC_PY=$2; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) die "未知参数：$1" ;;
    esac
  done
}

finalize_config() {
  command -v realpath >/dev/null || die "系统缺少 realpath"
  DATASET_ROOT=$(realpath -m "$DATASET_ROOT")
  BASE_POLICY=$(realpath -m "$BASE_POLICY")
  OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")
  ISAAC_PY=$(realpath -m "$ISAAC_PY")

  [[ -n $JOB_NAME ]] || JOB_NAME=$(basename "$OUTPUT_DIR")
  if [[ -z $LOG_FILE ]]; then
    LOG_FILE=$PROJECT_ROOT/log/$(basename "$OUTPUT_DIR").log
  fi
  LOG_FILE=$(realpath -m "$LOG_FILE")
  PID_FILE=$LOG_FILE.pid

  GRADIENT_CHECKPOINTING=$(normalize_bool GRADIENT_CHECKPOINTING "$GRADIENT_CHECKPOINTING")
  TRAIN_EXPERT_ONLY=$(normalize_bool TRAIN_EXPERT_ONLY "$TRAIN_EXPERT_ONLY")
  CUDNN_DETERMINISTIC=$(normalize_bool CUDNN_DETERMINISTIC "$CUDNN_DETERMINISTIC")
  IMAGE_TRANSFORMS=$(normalize_bool IMAGE_TRANSFORMS "$IMAGE_TRANSFORMS")
  USE_IMAGENET_STATS=$(normalize_bool USE_IMAGENET_STATS "$USE_IMAGENET_STATS")
  WANDB_ENABLE=$(normalize_bool WANDB_ENABLE "$WANDB_ENABLE")
  OFFLINE=$(normalize_bool OFFLINE "$OFFLINE")
}

validate_positive_int() {
  local name=$1 value=$2
  [[ $value =~ ^[0-9]+$ ]] && (( value > 0 )) || die "$name 必须是正整数，当前值：$value"
}

validate_nonnegative_int() {
  local name=$1 value=$2
  [[ $value =~ ^[0-9]+$ ]] || die "$name 必须是非负整数，当前值：$value"
}

validate_number() {
  local name=$1 value=$2
  [[ $value =~ ^[+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] ||
    die "$name 必须是非负数，当前值：$value"
}

validate_common_config() {
  [[ -x $ISAAC_PY ]] || die "Isaac Python 不可执行：$ISAAC_PY"
  [[ $DTYPE == bfloat16 || $DTYPE == float32 ]] || die "DTYPE 只支持 bfloat16 或 float32"
  [[ $WANDB_MODE == online || $WANDB_MODE == offline || $WANDB_MODE == disabled ]] ||
    die "WANDB_MODE 只支持 online、offline 或 disabled"

  validate_positive_int NUM_PROCESSES "$NUM_PROCESSES"
  validate_positive_int BATCH_SIZE "$BATCH_SIZE"
  validate_nonnegative_int NUM_WORKERS "$NUM_WORKERS"
  validate_positive_int TARGET_STEPS "$TARGET_STEPS"
  validate_positive_int SAVE_FREQ "$SAVE_FREQ"
  validate_positive_int LOG_FREQ "$LOG_FREQ"
  validate_nonnegative_int SEED "$SEED"
  validate_positive_int MAIN_PROCESS_PORT "$MAIN_PROCESS_PORT"
  validate_nonnegative_int WARMUP_STEPS "$WARMUP_STEPS"
  validate_positive_int DECAY_STEPS "$DECAY_STEPS"
  validate_positive_int N_ACTION_STEPS "$N_ACTION_STEPS"
  validate_positive_int NUM_INFERENCE_STEPS "$NUM_INFERENCE_STEPS"
  (( N_ACTION_STEPS <= 50 )) || die "当前 π0.5 chunk_size=50，N_ACTION_STEPS 不能大于 50"
  validate_number LEARNING_RATE "$LEARNING_RATE"
  validate_number WEIGHT_DECAY "$WEIGHT_DECAY"
  validate_number GRAD_CLIP_NORM "$GRAD_CLIP_NORM"
}

validate_hardware() {
  command -v nvidia-smi >/dev/null || die "找不到 nvidia-smi"
  local -a requested visible
  IFS=',' read -r -a requested <<< "$GPU_IDS"
  mapfile -t visible < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
  [[ ${#requested[@]} -eq $NUM_PROCESSES ]] ||
    die "GPU_IDS 有 ${#requested[@]} 项，但 NUM_PROCESSES=$NUM_PROCESSES"

  local id candidate found
  for id in "${requested[@]}"; do
    [[ $id =~ ^[0-9]+$ ]] || die "非法 GPU ID：$id"
    found=false
    for candidate in "${visible[@]}"; do
      [[ $id == "$candidate" ]] && found=true
    done
    [[ $found == true ]] || die "GPU $id 不可见；当前可见：${visible[*]}"
  done
}

validate_port_available() {
  "$ISAAC_PY" -c '
import socket, sys
port = int(sys.argv[1])
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", port))
finally:
    sock.close()
' "$MAIN_PROCESS_PORT" >/dev/null 2>&1 ||
    die "DDP 端口 $MAIN_PROCESS_PORT 已被占用，请用 --port 更换"
}

validate_start_paths() {
  [[ -d $DATASET_ROOT ]] || die "数据集目录不存在：$DATASET_ROOT"
  [[ -f $DATASET_ROOT/meta/info.json ]] || die "缺少 meta/info.json：$DATASET_ROOT"
  [[ -f $DATASET_ROOT/meta/stats.json ]] || die "缺少 meta/stats.json：$DATASET_ROOT"
  [[ -d $BASE_POLICY ]] || die "基础策略目录不存在：$BASE_POLICY"
  [[ -f $BASE_POLICY/config.json ]] || die "基础策略缺少 config.json：$BASE_POLICY"
  [[ -f $BASE_POLICY/model.safetensors ]] || die "基础策略缺少 model.safetensors：$BASE_POLICY"
}

# 解析 split 或自定义 episode 选择器，并在启动前检查越界、重复和数据格式。
resolve_episodes() {
  local result
  result=$("$ISAAC_PY" -c '
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
selector = sys.argv[2].strip()
info = json.loads((root / "meta/info.json").read_text())
json.loads((root / "meta/stats.json").read_text())

if info.get("codebase_version") != "v3.0":
    version = info.get("codebase_version")
    raise SystemExit(f"只支持 LeRobot v3.0，当前为 {version!r}")
total = int(info.get("total_episodes", 0))
if total <= 0:
    raise SystemExit("total_episodes 必须大于 0")
features = info.get("features", {})
for key in ("observation.state", "action"):
    if key not in features:
        raise SystemExit(f"数据集缺少特征 {key}")
visual = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") in {"video", "image"}]
if not visual:
    raise SystemExit("数据集没有图像或视频特征")

def parse_spec(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    if not isinstance(value, str):
        raise ValueError(f"不支持的 episode 规格：{value!r}")
    value = value.strip()
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("JSON episode 选择器必须是列表")
        return [int(x) for x in parsed]
    output = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start, stop = token.split(":", 1)
            output.extend(range(int(start), int(stop)))
        else:
            output.append(int(token))
    return output

if selector == "all":
    episodes = None
elif selector.startswith("split:"):
    name = selector.split(":", 1)[1]
    if name not in info.get("splits", {}):
        split_names = sorted(info.get("splits", {}))
        raise SystemExit(f"split {name!r} 不存在，可用值：{split_names}")
    episodes = parse_spec(info["splits"][name])
elif selector in info.get("splits", {}):
    episodes = parse_spec(info["splits"][selector])
else:
    episodes = parse_spec(selector)

if episodes is not None:
    if not episodes:
        raise SystemExit("episode 选择结果为空")
    if len(episodes) != len(set(episodes)):
        raise SystemExit("episode 选择结果包含重复索引")
    invalid = [x for x in episodes if x < 0 or x >= total]
    if invalid:
        raise SystemExit(f"episode 越界，示例：{invalid[:10]}，total={total}")
    encoded = json.dumps(episodes, separators=(",", ":"))
    summary = f"{len(episodes)} episodes，首尾={episodes[0]}/{episodes[-1]}"
else:
    encoded = "__ALL__"
    summary = f"全部 {total} episodes"

state_shape = features["observation.state"].get("shape")
action_shape = features["action"].get("shape")
print(encoded)
print(f"{summary}；state={state_shape}；action={action_shape}；视觉特征={len(visual)}")
' "$DATASET_ROOT" "$EPISODES") || die "数据集或 episode 选择校验失败"

  EPISODES_JSON=${result%%$'\n'*}
  DATASET_DESCRIPTION=${result#*$'\n'}
  if [[ $EPISODES_JSON == __ALL__ ]]; then
    EPISODES_JSON=''
    EPISODE_COUNT=all
  else
    EPISODE_COUNT=$("$ISAAC_PY" -c 'import json,sys; print(len(json.loads(sys.argv[1])))' "$EPISODES_JSON")
  fi
}

# 完整 checkpoint 不只看目录名，还会解析 JSON 并打开模型、优化器和 RNG 的 safetensors 头。
checkpoint_is_complete() {
  local checkpoint=$1 expected_step=$2
  "$ISAAC_PY" -c '
import json, sys
from pathlib import Path
from safetensors import safe_open

root = Path(sys.argv[1])
expected = int(sys.argv[2])
required_json = (
    "pretrained_model/config.json",
    "pretrained_model/policy_preprocessor.json",
    "pretrained_model/policy_postprocessor.json",
    "pretrained_model/train_config.json",
    "training_state/optimizer_param_groups.json",
    "training_state/scheduler_state.json",
    "training_state/training_step.json",
)
for rel in required_json:
    json.loads((root / rel).read_text())
step = json.loads((root / "training_state/training_step.json").read_text())["step"]
if int(step) != expected:
    raise ValueError((step, expected))
for rel in (
    "pretrained_model/model.safetensors",
    "training_state/optimizer_state.safetensors",
    "training_state/rng_state.safetensors",
):
    with safe_open(root / rel, framework="pt", device="cpu") as handle:
        if not list(handle.keys()):
            raise ValueError(f"空 safetensors：{rel}")
' "$checkpoint" "$expected_step" >/dev/null 2>&1
}

latest_valid_checkpoint() {
  local latest='' latest_step=-1 checkpoint name step
  [[ -d $OUTPUT_DIR/checkpoints ]] || return 1
  shopt -s nullglob
  for checkpoint in "$OUTPUT_DIR"/checkpoints/*; do
    [[ -d $checkpoint ]] || continue
    name=${checkpoint##*/}
    [[ $name =~ ^[0-9]+$ ]] || continue
    step=$((10#$name))
    if (( step > latest_step )) && checkpoint_is_complete "$checkpoint" "$step"; then
      latest=$checkpoint
      latest_step=$step
    fi
  done
  shopt -u nullglob
  [[ -n $latest ]] && printf '%s\n' "$latest"
}

checkpoint_step() {
  local name=${1##*/}
  printf '%d\n' "$((10#$name))"
}

# 断电可能留下比最新完整 checkpoint 更新的半成品目录；只移动归档，不删除。
archive_incomplete_after() {
  local keep_step=$1 archive_root='' checkpoint name step
  shopt -s nullglob
  for checkpoint in "$OUTPUT_DIR"/checkpoints/*; do
    [[ -d $checkpoint ]] || continue
    name=${checkpoint##*/}
    [[ $name =~ ^[0-9]+$ ]] || continue
    step=$((10#$name))
    if (( step > keep_step )); then
      if [[ -z $archive_root ]]; then
        archive_root=$OUTPUT_DIR/interrupted_checkpoints_$(date -u +%Y%m%dT%H%M%SZ)
        mkdir -p "$archive_root"
      fi
      mv "$checkpoint" "$archive_root/"
    fi
  done
  shopt -u nullglob
  [[ -z $archive_root ]] || printf '已归档不完整 checkpoint：%s\n' "$archive_root"
}

job_is_running() {
  local line
  while IFS= read -r line; do
    [[ $line == *"--job_name=$JOB_NAME"* ]] && return 0
  done < <(pgrep -af '[l]erobot.scripts.lerobot_train' || true)
  return 1
}

build_accelerate_prefix() {
  ACCELERATE_PREFIX=("$ISAAC_PY" -m accelerate.commands.launch)
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

# 新训练会把所有主要配置显式写进 train_config.json，方便之后准确复现。
build_fresh_command() {
  build_accelerate_prefix
  TRAIN_COMMAND=(
    "${ACCELERATE_PREFIX[@]}"
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.root="$DATASET_ROOT"
  )
  [[ -z $EPISODES_JSON ]] || TRAIN_COMMAND+=(--dataset.episodes="$EPISODES_JSON")
  TRAIN_COMMAND+=(
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
    --output_dir="$OUTPUT_DIR"
    --job_name="$JOB_NAME"
    --resume=false
    --batch_size="$BATCH_SIZE"
    --num_workers="$NUM_WORKERS"
    --steps="$TARGET_STEPS"
    --save_checkpoint=true
    --save_freq="$SAVE_FREQ"
    --eval_freq=0
    --log_freq="$LOG_FREQ"
    --seed="$SEED"
    --cudnn_deterministic="$CUDNN_DETERMINISTIC"
    --wandb.enable="$WANDB_ENABLE"
    --wandb.project="$WANDB_PROJECT"
    --wandb.mode="$WANDB_MODE"
  )
}

load_resume_config() {
  local checkpoint=$1 values saved_steps saved_save saved_log saved_batch saved_workers saved_job
  local -a saved
  values=$("$ISAAC_PY" -c '
import json, sys
cfg = json.load(open(sys.argv[1]))
dataset = cfg["dataset"]
policy = cfg["policy"]
episodes = dataset.get("episodes")
if episodes is None:
    episode_summary = "全部 episodes"
else:
    episode_summary = f"{len(episodes)} episodes，首尾={episodes[0]}/{episodes[-1]}"
print(
    int(cfg["steps"]), int(cfg["save_freq"]), int(cfg["log_freq"]),
    int(cfg["batch_size"]), int(cfg["num_workers"]), cfg.get("job_name") or "pi05",
    sep="\t",
)
print(dataset.get("root") or "")
print(dataset.get("repo_id") or "")
print(episode_summary)
print(
    policy.get("dtype", "bfloat16"),
    str(bool(policy.get("gradient_checkpointing", False))).lower(),
    str(bool(policy.get("train_expert_only", False))).lower(),
    policy.get("optimizer_lr", 2.5e-5),
    policy.get("optimizer_weight_decay", 0.01),
    policy.get("optimizer_grad_clip_norm", 1.0),
    int(policy.get("scheduler_warmup_steps", 1000)),
    int(policy.get("scheduler_decay_steps", 30000)),
    int(policy.get("n_action_steps", 50)),
    int(policy.get("num_inference_steps", 10)),
    sep="\t",
)
transforms = dataset.get("image_transforms") or {}
print(
    int(cfg.get("seed", 1000)),
    str(bool(cfg.get("cudnn_deterministic", False))).lower(),
    dataset.get("video_backend", "pyav"),
    str(bool(transforms.get("enable", False))).lower(),
    str(bool(dataset.get("use_imagenet_stats", True))).lower(),
    sep="\t",
)
' "$checkpoint/pretrained_model/train_config.json")
  mapfile -t saved <<< "$values"
  IFS=$'\t' read -r saved_steps saved_save saved_log saved_batch saved_workers saved_job <<< "${saved[0]}"

  [[ $STEPS_OVERRIDDEN == true ]] || TARGET_STEPS=$saved_steps
  [[ $SAVE_FREQ_OVERRIDDEN == true ]] || SAVE_FREQ=$saved_save
  [[ $LOG_FREQ_OVERRIDDEN == true ]] || LOG_FREQ=$saved_log
  [[ $BATCH_SIZE_OVERRIDDEN == true ]] || BATCH_SIZE=$saved_batch
  [[ $NUM_WORKERS_OVERRIDDEN == true ]] || NUM_WORKERS=$saved_workers
  [[ $JOB_NAME_OVERRIDDEN == true ]] || JOB_NAME=$saved_job
  DATASET_ROOT=${saved[1]}
  DATASET_REPO_ID=${saved[2]}
  DATASET_DESCRIPTION=${saved[3]}
  IFS=$'\t' read -r DTYPE GRADIENT_CHECKPOINTING TRAIN_EXPERT_ONLY LEARNING_RATE \
    WEIGHT_DECAY GRAD_CLIP_NORM WARMUP_STEPS DECAY_STEPS N_ACTION_STEPS \
    NUM_INFERENCE_STEPS <<< "${saved[4]}"
  IFS=$'\t' read -r SEED CUDNN_DETERMINISTIC VIDEO_BACKEND IMAGE_TRANSFORMS \
    USE_IMAGENET_STATS <<< "${saved[5]}"
  BASE_POLICY=$checkpoint/pretrained_model
  CURRENT_STEP=$(checkpoint_step "$checkpoint")
  EPISODE_COUNT=checkpoint
}

# 恢复时默认尊重 checkpoint 内的完整配置；只追加用户显式要求覆盖的运行参数。
build_resume_command() {
  local checkpoint=$1
  build_accelerate_prefix
  TRAIN_COMMAND=(
    "${ACCELERATE_PREFIX[@]}"
    --config_path="$checkpoint/pretrained_model/train_config.json"
    --resume=true
    --output_dir="$OUTPUT_DIR"
    --job_name="$JOB_NAME"
    --wandb.enable="$WANDB_ENABLE"
    --wandb.project="$WANDB_PROJECT"
    --wandb.mode="$WANDB_MODE"
  )
  [[ $STEPS_OVERRIDDEN == false ]] || TRAIN_COMMAND+=(--steps="$TARGET_STEPS")
  [[ $SAVE_FREQ_OVERRIDDEN == false ]] || TRAIN_COMMAND+=(--save_freq="$SAVE_FREQ")
  [[ $LOG_FREQ_OVERRIDDEN == false ]] || TRAIN_COMMAND+=(--log_freq="$LOG_FREQ")
  [[ $BATCH_SIZE_OVERRIDDEN == false ]] || TRAIN_COMMAND+=(--batch_size="$BATCH_SIZE")
  [[ $NUM_WORKERS_OVERRIDDEN == false ]] || TRAIN_COMMAND+=(--num_workers="$NUM_WORKERS")
}

print_command() {
  local arg shown
  printf '实际训练命令（episode 列表已缩写）：\n '
  for arg in "${TRAIN_COMMAND[@]}"; do
    shown=$arg
    case "$arg" in
      --dataset.episodes=*) shown="--dataset.episodes=<${EPISODE_COUNT} episodes>" ;;
    esac
    printf ' %q' "$shown"
  done
  printf '\n'
}

print_summary() {
  local mode=$1
  cat <<EOF
---------------- π0.5 训练配置 ----------------
模式              : $mode
数据集            : $DATASET_ROOT
repo_id           : $DATASET_REPO_ID
episode           : $DATASET_DESCRIPTION
基础策略          : $BASE_POLICY
输出目录          : $OUTPUT_DIR
日志              : $LOG_FILE
任务名            : $JOB_NAME
GPU / 进程        : $GPU_IDS / $NUM_PROCESSES
batch / workers   : $BATCH_SIZE / $NUM_WORKERS（每个进程）
目标步数          : $TARGET_STEPS
保存 / 日志频率   : $SAVE_FREQ / $LOG_FREQ
dtype             : $DTYPE
梯度检查点        : $GRADIENT_CHECKPOINTING
仅训练 expert     : $TRAIN_EXPERT_ONLY
学习率            : $LEARNING_RATE
warmup / decay    : $WARMUP_STEPS / $DECAY_STEPS
离线模式          : $OFFLINE
------------------------------------------------
EOF
}

# supervisor 自身由 setsid 脱离 SSH，并负责把最终退出码写入日志。
launch_detached() {
  local mode=$1
  install -d "$(dirname "$LOG_FILE")"
  if [[ $mode == fresh && -e $LOG_FILE ]]; then
    mv "$LOG_FILE" "$LOG_FILE.previous_$(date -u +%Y%m%dT%H%M%SZ)"
  fi

  {
    printf '[%s] mode=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$mode"
    print_summary "$mode"
    print_command
  } >> "$LOG_FILE"

  RUN_ENV=(
    "PYTHONPATH=$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  )
  if [[ $OFFLINE == true ]]; then
    RUN_ENV+=(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)
  fi

  cd "$PROJECT_ROOT"
  nohup setsid bash -c '
"$@"
rc=$?
printf "[%s] train command exited rc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc"
exit "$rc"
' _ env "${RUN_ENV[@]}" "${TRAIN_COMMAND[@]}" >> "$LOG_FILE" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"

  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    tail -n 30 "$LOG_FILE" >&2 || true
    die "训练命令启动后立即退出"
  fi
  printf '训练已在后台启动：pid=%s\n日志：%s\n' "$pid" "$LOG_FILE"
}

prepare_fresh() {
  validate_common_config
  validate_start_paths
  validate_hardware
  validate_port_available
  resolve_episodes
  [[ ! -e $OUTPUT_DIR ]] || die "输出目录已存在；新训练请换目录，续训请用 resume：$OUTPUT_DIR"
  job_is_running && die "同名任务正在运行：$JOB_NAME"
  build_fresh_command
}

prepare_resume() {
  validate_common_config
  [[ -d $OUTPUT_DIR ]] || die "续训输出目录不存在：$OUTPUT_DIR"
  LATEST_CHECKPOINT=$(latest_valid_checkpoint || true)
  [[ -n $LATEST_CHECKPOINT ]] || die "没有找到完整 checkpoint；不会覆盖现有残缺输出"
  load_resume_config "$LATEST_CHECKPOINT"
  job_is_running && die "同名任务正在运行：$JOB_NAME"
  validate_common_config
  validate_hardware
  validate_port_available
  (( TARGET_STEPS >= CURRENT_STEP )) ||
    die "目标步数 $TARGET_STEPS 小于当前 checkpoint step $CURRENT_STEP"
  build_resume_command "$LATEST_CHECKPOINT"
}

start_training() {
  prepare_fresh
  print_summary fresh
  print_command
  launch_detached fresh
}

resume_training() {
  prepare_resume
  if (( TARGET_STEPS == CURRENT_STEP )); then
    printf '训练已达到目标：step=%s，checkpoint=%s\n' "$CURRENT_STEP" "$LATEST_CHECKPOINT"
    return
  fi
  archive_incomplete_after "$CURRENT_STEP"
  printf '将从完整 checkpoint 恢复：%s\n' "$LATEST_CHECKPOINT"
  print_summary resume
  print_command
  launch_detached resume
}

show_status() {
  local latest saved_job
  latest=$(latest_valid_checkpoint || true)
  if [[ -n $latest && $JOB_NAME_OVERRIDDEN == false ]]; then
    saved_job=$("$ISAAC_PY" -c '
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("job_name") or "pi05")
' "$latest/pretrained_model/train_config.json")
    JOB_NAME=$saved_job
  fi

  printf '任务名：%s\n' "$JOB_NAME"
  if job_is_running; then
    printf '进程：运行中'
    [[ -f $PID_FILE ]] && printf '（supervisor pid=%s）' "$(<"$PID_FILE")"
    printf '\n'
  else
    printf '进程：未运行\n'
  fi

  if [[ -n $latest ]]; then
    printf '最新完整 checkpoint：%s\n' "$latest"
  else
    printf '最新完整 checkpoint：无\n'
  fi
  if [[ -f $LOG_FILE ]]; then
    printf '日志末尾（%s）：\n' "$LOG_FILE"
    tail -n 12 "$LOG_FILE" | sed 's/^/  /'
  else
    printf '日志：尚未生成（%s）\n' "$LOG_FILE"
  fi
}

main() {
  parse_args "$@"
  finalize_config

  case "$ACTION" in
    start)
      start_training
      ;;
    resume)
      resume_training
      ;;
    auto)
      if job_is_running; then
        show_status
      elif [[ ! -e $OUTPUT_DIR ]]; then
        start_training
      elif latest_valid_checkpoint >/dev/null 2>&1; then
        resume_training
      else
        die "输出目录存在但没有完整 checkpoint；为避免覆盖，auto 已停止"
      fi
      ;;
    status)
      show_status
      ;;
    dry-run)
      prepare_fresh
      print_summary dry-run
      print_command
      printf 'dry-run 通过：没有启动训练，也没有创建输出目录。\n'
      ;;
    dry-run-resume)
      prepare_resume
      print_summary dry-run-resume
      print_command
      printf 'dry-run-resume 通过：没有启动训练。\n'
      ;;
  esac
}

main "$@"
