#!/bin/bash
# =============================================================================
# 立柱线最小闭环 smoke：录专家数据 -> 训练 π0.5 -> 端到端 rollout 出视频
#
# 路径约定：本脚本位于 <ROOT>/cruzr_mujoco_sim/smoke/，ROOT 即迁移包解压根目录
# （MANIFEST 中的 ~/pillar，本机可能是 cruzr_sim 等任意路径）。
#
# 用法:
#   bash smoke/run_pillar_smoke.sh all       # 全流程（已有产物会跳过，FORCE=1 重做）
#   bash smoke/run_pillar_smoke.sh env       # 仅环境检查
#   bash smoke/run_pillar_smoke.sh record    # 单步: record|dataset|train|rollout|clean
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG="$ROOT/cruzr_mujoco_sim"
MJX_PY="$ROOT/envs/mjx/bin/python"
DOCKER_IMG="${DOCKER_IMG:-ghrc_2026:v0}"
DATA_ROOT="$ROOT/smoke_data/lerobot/smoke"
DATASET="${DATASET:-cruzr_shelf_e2e_smoke}"
OUT_ROOT="$ROOT/smoke_outputs"
RUN="$OUT_ROOT/pi05_shelf_e2e_smoke"
HF_CACHE="$ROOT/smoke_data/hf_cache"
PRETRAINED="$ROOT/pretrained"

GPU_SIM="${GPU_SIM:-0}"
GPU_POLICY="${GPU_POLICY:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-800}"
PORT="${PORT:-8731}"
EP_SEEDS="${EP_SEEDS:-1 2 3}"

cd "$PKG"

step() { echo; echo "================ STEP $1: $2 ================"; }
report() { echo; echo "----- 本步产出 -----"; echo "$1"; echo; echo "----- 下一步 -----"; echo "$2"; echo; }

docker_run() {
  docker run --rm --gpus all --ipc=host --entrypoint bash \
    -v "$ROOT:/hxb" \
    -v "$PRETRAINED/paligemma-3b-pt-224:/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/pretrained/paligemma-3b-pt-224:ro" \
    -w /hxb \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/hxb/smoke_data/hf_cache -e PYTHONPATH=/hxb \
    "$DOCKER_IMG" -c "$1"
}

prep_hf_cache() {
  local C="$HF_CACHE/hub/models--google--paligemma-3b-pt-224"
  if [ ! -f "$C/refs/main" ]; then
    mkdir -p "$C/refs" "$C/snapshots/smoke_local"
    echo "smoke_local" > "$C/refs/main"
    for f in added_tokens.json special_tokens_map.json tokenizer_config.json tokenizer.model \
             tokenizer.json preprocessor_config.json generation_config.json config.json; do
      cp "$PRETRAINED/paligemma-3b-pt-224/$f" "$C/snapshots/smoke_local/$f"
    done
  fi
}

check_env() {
  step 0 "环境检查"
  [ -x "$MJX_PY" ] || { echo "缺少 mjx 环境: $MJX_PY（先解压 mjx.tar.gz 并 conda-unpack）"; exit 1; }
  "$MJX_PY" -c 'import mujoco; assert mujoco.__version__=="3.9.0", mujoco.__version__; print("mujoco 3.9.0 OK")'
  "$MJX_PY" -c 'import pyarrow, openpi_client, imageio_ffmpeg, PIL; print("mjx deps OK")'
  command -v docker >/dev/null || { echo "WARN: docker 未安装，训练/推理步骤不可用"; }
  if command -v docker >/dev/null; then
    docker image inspect "$DOCKER_IMG" >/dev/null && echo "docker image $DOCKER_IMG OK"
  fi
  which ffmpeg >/dev/null && echo "ffmpeg OK"
  nvidia-smi --query-gpu=index,name --format=csv,noheader
  [ -d "$PRETRAINED/pi05_base_remapped" ] && echo "pi05_base_remapped OK"
  [ -d "$PRETRAINED/paligemma-3b-pt-224" ] && echo "paligemma-3b-pt-224 OK"
  prep_hf_cache && echo "HF 离线缓存 OK: $HF_CACHE"
}

record() {
  step 1 "录制双物料专家数据（seeds: $EP_SEEDS）"
  mkdir -p out/teleop/shelf_e2e_dual out/logs/smoke
  local ok=0
  for s in $EP_SEEDS; do
    local out="shelf_e2e_dual_smoke_$(printf '%06d' $s)"
    if [ -f "out/teleop/shelf_e2e_dual/$out/meta.json" ] && [ "${FORCE:-0}" != 1 ]; then
      echo "seed $s 已有录制，跳过（FORCE=1 重录）"
      grep -q 'DUAL EPISODE PASS' "out/logs/smoke/_smoke_pillar_s${s}.log" 2>/dev/null || echo "  (先前日志缺失，视为 PASS)"
      ok=$((ok + 1))
      continue
    fi
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_SIM TELEOP_RECORD_GPU=$GPU_SIM \
      SEED=$s E2E_KICKS=0 EXPERT_OUT=out/teleop/shelf_e2e_dual/$out \
      "$MJX_PY" scripts/collection/shelf_e2e_dual_expert.py \
      > "out/logs/smoke/_smoke_pillar_s${s}.log" 2>&1 || true
    if grep -q 'DUAL EPISODE PASS' "out/logs/smoke/_smoke_pillar_s${s}.log"; then
      echo "seed $s: PASS"
      ok=$((ok + 1))
    else
      echo "seed $s: FAIL"
      rm -rf "out/teleop/shelf_e2e_dual/$out"
    fi
  done
  [ "$ok" -ge 1 ] || { echo "没有成功的 episode，请查看 out/logs/smoke/_smoke_pillar_s*.log"; exit 1; }
  report \
    "PASS 的 episode: out/teleop/shelf_e2e_dual/shelf_e2e_dual_smoke_<seed>/（meta.json + npz + 3 相机 224x224 jpg）" \
    "bash $0 dataset"
}

dataset() {
  step 2 "构建 LeRobot 数据集（v2.1 -> v3.0 -> 分位数统计）"
  local ep_glob="$PKG/out/teleop/shelf_e2e_dual/shelf_e2e_dual_smoke_*"
  local n
  n=$(ls -d $ep_glob 2>/dev/null | wc -l)
  [ "$n" -ge 1 ] || { echo "没有找到 episode，先跑 record"; exit 1; }

  rm -rf "$DATA_ROOT/$DATASET"
  EPISODES="$ep_glob" OUT="$DATA_ROOT/$DATASET" "$MJX_PY" scripts/collection/shelf_e2e_build_v2.py

  docker_run "/isaac-sim/python.sh src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id smoke/$DATASET --root /hxb/smoke_data/lerobot/smoke/$DATASET --push-to-hub false"

  docker_run "/isaac-sim/python.sh src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
    --repo-id smoke/$DATASET --root /hxb/smoke_data/lerobot/smoke/$DATASET" || true

  "$MJX_PY" -c "
import json
s = json.load(open('$DATA_ROOT/$DATASET/meta/stats.json'))
assert 'q01' in s['action'], '分位数统计缺失'
print('数据集校验 OK: action', list(s['action'].keys()))"
  report \
    "$DATA_ROOT/$DATASET —— v3.0 LeRobot 数据集（18 维 state / 18 维 action / 3 相机）" \
    "bash $0 train"
}

train() {
  step 3 "训练 π0.5（$TRAIN_STEPS 步，LEROBOT_STATE_N_DIMS=18）"
  [ -d "$DATA_ROOT/$DATASET" ] || { echo "数据集不存在，先跑 dataset"; exit 1; }
  [ ! -d "$RUN" ] || [ "${FORCE:-0}" = 1 ] || { echo "$RUN 已存在，跳过（FORCE=1 重训）"; return; }
  docker run --rm --gpus all --ipc=host --name pillar_smoke_train --entrypoint bash \
    -v "$ROOT:/hxb" \
    -v "$PRETRAINED/paligemma-3b-pt-224:/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/pretrained/paligemma-3b-pt-224:ro" \
    -w /hxb \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/hxb/smoke_data/hf_cache -e PYTHONPATH=/hxb \
    -e LEROBOT_STATE_N_DIMS=18 -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$DOCKER_IMG" -c "/isaac-sim/python.sh src/lerobot/scripts/lerobot_train.py \
      --dataset.repo_id=smoke/$DATASET --dataset.root=/hxb/smoke_data/lerobot/smoke/$DATASET --dataset.video_backend=pyav \
      --policy.path=/hxb/pretrained/pi05_base_remapped \
      --policy.device=cuda --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
      --policy.train_expert_only=true --policy.push_to_hub=false \
      --output_dir=/hxb/smoke_outputs/pi05_shelf_e2e_smoke --job_name=shelf_e2e_smoke \
      --batch_size=4 --num_workers=4 --steps=$TRAIN_STEPS \
      --save_checkpoint=true --save_freq=$((TRAIN_STEPS / 2)) --log_freq=100 --eval_freq=0 --wandb.enable=false" \
    2>&1 | tee "$OUT_ROOT/train_shelf_e2e.log" | grep -E 'loss:|Checkpoint|Error' || true
  report \
    "$RUN/checkpoints/00$TRAIN_STEPS/pretrained_model/" \
    "bash $0 rollout"
}

rollout() {
  step 4 "推理展示（pillar policy server + shelf_e2e_rollout）"
  local CKPT="/hxb/smoke_outputs/pi05_shelf_e2e_smoke/checkpoints/00$TRAIN_STEPS/pretrained_model"
  [ -d "$RUN/checkpoints/00$TRAIN_STEPS/pretrained_model" ] \
    || { echo "checkpoint 不存在，先跑 train"; exit 1; }

  docker rm -f pillar_policy_server >/dev/null 2>&1 || true
  echo "启动 pillar policy server（GPU $GPU_POLICY，端口 $PORT）..."
  docker run --rm -d --gpus all --ipc=host --name pillar_policy_server -p "$PORT:$PORT" --entrypoint bash \
    -v "$ROOT:/hxb" \
    -v "$PRETRAINED/paligemma-3b-pt-224:/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/pretrained/paligemma-3b-pt-224:ro" \
    -w /hxb \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/hxb/smoke_data/hf_cache -e PYTHONPATH=/hxb \
    -e CUDA_VISIBLE_DEVICES=$GPU_POLICY -e POLICY_CKPT=$CKPT -e POLICY_PORT=$PORT \
    "$DOCKER_IMG" -c "/isaac-sim/python.sh -m pip install -q /hxb/cruzr_mujoco_sim/smoke/openpi_client-0.1.2-py3-none-any.whl && \
                      /isaac-sim/python.sh /hxb/cruzr_mujoco_sim/smoke/pillar_lerobot_policy_server.py"
  for _ in $(seq 1 60); do
    docker logs pillar_policy_server 2>&1 | grep -q 'listening' && break
    sleep 5
  done
  docker logs pillar_policy_server 2>&1 | grep -q 'listening' \
    || { echo "server 启动失败:"; docker logs pillar_policy_server | tail -20; exit 1; }
  echo "server 就绪"

  set +e
  mkdir -p out/logs/smoke out/rollout/shelf_e2e
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_SIM TELEOP_RECORD_GPU=$GPU_SIM \
    POLICY_PORT=$PORT SEED="${ROLLOUT_SEED:-10}" \
    "$MJX_PY" scripts/collection/shelf_e2e_rollout.py 2>&1 | tee out/logs/smoke/_smoke_pillar_rollout.log
  set -e
  docker rm -f pillar_policy_server >/dev/null 2>&1 || true

  report \
    "out/rollout/shelf_e2e/ 下 rollout 视频 + out/logs/smoke/_smoke_pillar_rollout.log" \
    "想真能用: 参照 MANIFEST 批量采集 50+ 条，步数加大到 20k+"
}

clean() {
  docker rm -f pillar_policy_server pillar_smoke_train 2>/dev/null || true
  rm -rf "$OUT_ROOT/pi05_shelf_e2e_smoke" "$DATA_ROOT/$DATASET" "$DATA_ROOT/${DATASET}_old"
  echo "已清理训练输出和数据集（录制的原始 episode 保留）"
}

case "${1:-all}" in
  env)     check_env ;;
  record)  check_env; record ;;
  dataset) dataset ;;
  train)   train ;;
  rollout) rollout ;;
  clean)   clean ;;
  all)     check_env; record; dataset; train; rollout ;;
  *) echo "用法: bash $0 [env|record|dataset|train|rollout|clean|all]"; exit 1 ;;
esac
