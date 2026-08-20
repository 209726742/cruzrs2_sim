#!/bin/bash
# =============================================================================
# ECU 线最小闭环 smoke：录 1~2 条专家数据 -> 训练 π0.5 -> 推理出视频
#
# 技术路线（2026-07-30 实测跑通）:
#   仿真/录制/rollout : conda 环境 mjx (python3.11 + mujoco==3.9.0)
#   训练/推理 server  : 本地 docker 镜像 ghrc_2026:v0 (Baseline LeRobot PyTorch π0.5)
#   数据集            : ECU 包产出 LeRobot v2.1 -> Baseline 自带脚本转 v3.0 -> 补分位数统计
#   策略权重          : Baseline/pretrained/pi05_base_remapped（本地 pi05_base 的视觉塔键名
#                       重映射版；原版键名多一层 .vision_model 前缀，训练侧加载会静默丢
#                       437 个视觉权重 —— 重映射脚本见本文件 train 步骤注释，已一次性生成）
#
# 关键环境差异（相对原开发机，已在此脚本中处理）:
#   1) mujoco 必须是 3.9.0 —— 3.11 下 S2 bridge 验收门会差 2mm 挂掉（确定性现象）
#   2) Baseline lerobot_train.py 的 state 截断维度默认 20，本流程用环境变量
#      LEROBOT_STATE_N_DIMS=21（脚本已注入；该改动见 Baseline 文件头部注释）
#   3) 本机访问不了 huggingface.co：HF_HUB_OFFLINE=1 + 本地 HF cache（脚本自动准备）
#   4) Baseline tokenizer_processor.py 写死 paligemma 路径 /workspace/...，
#      用 docker bind mount 把本地权重挂到那个路径（脚本已处理）
#   5) 数据集只保留 3 路相机（BUILD_CAMS），与 rollout 推理时可用相机严格一致
#
# 用法:
#   bash smoke/run_ecu_smoke.sh all       # 全流程（已存在的产物会跳过，FORCE=1 重做）
#   bash smoke/run_ecu_smoke.sh record    # 只跑某一步: record|split|dataset|train|rollout|clean
# =============================================================================
set -euo pipefail

# ----------------------------- 配置 -----------------------------
PKG=/mnt/my_hdd/hxb/cruzr_mujoco_sim
MJX_PY=/home/lh/miniconda3/envs/mjx/bin/python
BASELINE=/mnt/my_hdd/hxb/Baseline
DOCKER_IMG=ghrc_2026:v0
DATA_ROOT=/mnt/my_hdd/hxb/smoke_data/lerobot/smoke
DATASET=cruzr_ecu_smoke3cam
OUT_ROOT=/mnt/my_hdd/hxb/smoke_outputs
RUN=$OUT_ROOT/pi05_ecu_smoke3cam
HF_CACHE=/mnt/my_hdd/hxb/smoke_data/hf_cache

GPU_SIM=0          # 录制/rollout 渲染用哪块卡
GPU_POLICY=1       # 推理 server 用哪块卡（训练用卡由 --gpus 全部可见，默认占 cuda:0）
TRAIN_STEPS=${TRAIN_STEPS:-800}
PORT=8735

EP_SEEDS="1 2 3"   # 并行尝试的专家种子（seed0 在部分机器上 bridge 门差 1~2mm）
EP_BAY=0,1

cd "$PKG"

# ----------------------------- 输出工具 -----------------------------
step() { echo; echo "================ STEP $1: $2 ================"; }
report() { echo; echo "----- 本步产出 -----"; echo "$1"; echo; echo "----- 下一步 -----"; echo "$2"; echo; }

docker_run() {  # 统一 docker 调用（Baseline 代码 + 离线 HF + paligemma 挂载）
  docker run --rm --gpus all --ipc=host --entrypoint bash \
    -v /mnt/my_hdd/hxb:/hxb \
    -v $BASELINE/pretrained/paligemma-3b-pt-224:/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/pretrained/paligemma-3b-pt-224:ro \
    -w /hxb/Baseline \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/hxb/smoke_data/hf_cache -e PYTHONPATH=/hxb/Baseline \
    "$DOCKER_IMG" -c "$1"
}

prep_hf_cache() {  # 本机连不上 huggingface.co，用本地权重目录伪造 HF cache（离线模式可读）
  local C=$HF_CACHE/hub/models--google--paligemma-3b-pt-224
  if [ ! -f "$C/refs/main" ]; then
    mkdir -p "$C/refs" "$C/snapshots/smoke_local"
    echo "smoke_local" > "$C/refs/main"
    for f in added_tokens.json special_tokens_map.json tokenizer_config.json tokenizer.model \
             tokenizer.json preprocessor_config.json generation_config.json config.json; do
      cp "$BASELINE/pretrained/paligemma-3b-pt-224/$f" "$C/snapshots/smoke_local/$f"
    done
  fi
}

# ----------------------------- STEP 0 环境检查 -----------------------------
check_env() {
  step 0 "环境检查"
  "$MJX_PY" -c "import mujoco; assert mujoco.__version__=='3.9.0', mujoco.__version__; print('mujoco 3.9.0 OK')"
  "$MJX_PY" -c "import pyarrow, openpi_client, imageio_ffmpeg, PIL; print('mjx deps OK (pyarrow/openpi-client/imageio-ffmpeg/PIL)')"
  docker image inspect "$DOCKER_IMG" >/dev/null && echo "docker image $DOCKER_IMG OK"
  which ffmpeg >/dev/null && echo "ffmpeg OK"
  nvidia-smi --query-gpu=index,name --format=csv,noheader
  prep_hf_cache && echo "HF 离线缓存 OK ($HF_CACHE)"
}

# ----------------------------- STEP 1 录制 -----------------------------
record() {
  step 1 "录制专家数据（并行种子: $EP_SEEDS，bay=$EP_BAY）"
  mkdir -p out/teleop/ecu out/logs/smoke out/smoke
  for s in $EP_SEEDS; do
    if [ -f "out/teleop/ecu/ecu_smoke_s$s/meta.json" ] && [ "${FORCE:-0}" != 1 ]; then
      echo "seed $s 已有录制，跳过（FORCE=1 重录）"; continue
    fi
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_SIM TELEOP_RECORD_GPU=$GPU_SIM TELEOP_HOME=droop \
    EXPERT_SEED=$s EXPERT_BAY=$EP_BAY EXPERT_OUT=ecu/ecu_smoke_s$s \
      "$MJX_PY" scripts/ecu_expert_record.py > out/logs/smoke/_smoke_s$s.log 2>&1 &
  done
  wait || true
  local ok=0
  for s in $EP_SEEDS; do
    if grep -q 'EPISODE PASS' out/logs/smoke/_smoke_s$s.log 2>/dev/null; then
      echo "seed $s: PASS ($(grep -o '([0-9]* frames)' out/logs/smoke/_smoke_s$s.log | tail -1))"; ok=$((ok+1))
    else
      echo "seed $s: FAIL ($(grep 'failed gate' out/logs/smoke/_smoke_s$s.log | tail -1 | cut -c1-90))"
      rm -rf "out/teleop/ecu/ecu_smoke_s$s"
    fi
  done
  [ "$ok" -ge 1 ] || { echo "没有成功的 episode，请先看 out/logs/smoke/_smoke_s*.log"; exit 1; }
  report \
    "PASS 的 episode 目录: out/teleop/ecu/ecu_smoke_s{种子}/ （meta.json + episode_data.npz + frames/5 相机 jpg）
       out/smoke/_s1_ready_pose.json  —— S1 抓取就绪位姿，rollout 起步要用
       作用: 专家示范数据，是整个流程的原料" \
    "bash $0 split  —— 把整条 episode 切成 S1~S4 四个阶段子集（各带自己的 prompt）"
}

# ----------------------------- STEP 2 切分 -----------------------------
split() {
  step 2 "stage_split 切四段"
  local srcs=()
  for s in $EP_SEEDS; do
    [ -f "out/teleop/ecu/ecu_smoke_s$s/meta.json" ] && srcs+=("out/teleop/ecu/ecu_smoke_s$s")
  done
  [ "${#srcs[@]}" -ge 1 ] || { echo "没有找到完整 episode，先跑 record"; exit 1; }
  "$MJX_PY" scripts/stage_split.py "${srcs[@]}"
  report \
    "每个完整 episode 派生 4 个子目录 *_st1.._st4（独立 meta.json，各带阶段 prompt）
       作用: 让策略学到「阶段→动作」的对应关系，rollout 按阶段给 prompt" \
    "bash $0 dataset  —— 打包成 LeRobot v2.1 数据集并转成 Baseline 能吃的 v3.0"
}

# ----------------------------- STEP 3 数据集 -----------------------------
dataset() {
  step 3 "构建数据集（3 相机 v2.1 -> v3.0 -> 分位数统计）"
  local LIST=out/smoke/_smoke_list.txt
  mkdir -p out/smoke
  : > "$LIST"
  for d in out/teleop/ecu/ecu_smoke_s*_st?; do echo "$(realpath "$d")" >> "$LIST"; done
  echo "清单 ($(wc -l < "$LIST") 条):"; cat "$LIST"

  rm -rf "$DATA_ROOT/$DATASET"
  BUILD_CAMS=stereo_left,stereo_right,waist_front \
    "$MJX_PY" scripts/build_carton_lerobot.py --list "$LIST" --out "$DATA_ROOT/$DATASET" --root /

  docker_run "/isaac-sim/python.sh src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id smoke/$DATASET --root /hxb/smoke_data/lerobot/smoke/$DATASET --push-to-hub false"

  # 该脚本算完分位数后会尝试 push_to_hub（必然失败，本机连不上 HF），属预期，统计已落盘
  docker_run "/isaac-sim/python.sh src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
    --repo-id smoke/$DATASET --root /hxb/smoke_data/lerobot/smoke/$DATASET" || true

  python3 -c "
import json
s = json.load(open('$DATA_ROOT/$DATASET/meta/stats.json'))
assert 'q01' in s['action'], '分位数统计缺失'
print('数据集校验 OK: action', list(s['action'].keys()))"
  report \
    "$DATA_ROOT/$DATASET  —— v3.0 LeRobot 数据集（parquet + mp4 + 归一化统计含 q01/q99）
       ${DATASET}_old 是 v2.1 原版的备份
       作用: 训练的直接输入；统计值同时用于训练归一化和推理反归一化" \
    "bash $0 train  —— docker 容器内微调 π0.5（冻结 VLM 只训 action expert）"
}

# ----------------------------- STEP 4 训练 -----------------------------
train() {
  step 4 "训练 π0.5（$TRAIN_STEPS 步，batch 4，train_expert_only）"
  [ -d "$DATA_ROOT/$DATASET" ] || { echo "数据集不存在，先跑 dataset"; exit 1; }
  [ ! -d "$RUN" ] || [ "${FORCE:-0}" = 1 ] || { echo "$RUN 已存在，跳过（FORCE=1 重训）"; return; }
  docker run --rm --gpus all --ipc=host --name ecu_smoke_train --entrypoint bash \
    -v /mnt/my_hdd/hxb:/hxb \
    -v $BASELINE/pretrained/paligemma-3b-pt-224:/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/pretrained/paligemma-3b-pt-224:ro \
    -w /hxb/Baseline \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/hxb/smoke_data/hf_cache -e PYTHONPATH=/hxb/Baseline \
    -e LEROBOT_STATE_N_DIMS=21 -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$DOCKER_IMG" -c "/isaac-sim/python.sh src/lerobot/scripts/lerobot_train.py \
      --dataset.repo_id=smoke/$DATASET --dataset.root=/hxb/smoke_data/lerobot/smoke/$DATASET --dataset.video_backend=pyav \
      --policy.path=/hxb/Baseline/pretrained/pi05_base_remapped \
      --policy.device=cuda --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
      --policy.train_expert_only=true --policy.push_to_hub=false \
      --output_dir=/hxb/smoke_outputs/pi05_ecu_smoke3cam --job_name=ecu_smoke3cam \
      --batch_size=4 --num_workers=4 --steps=$TRAIN_STEPS \
      --save_checkpoint=true --save_freq=$((TRAIN_STEPS/2)) --log_freq=100 --eval_freq=0 --wandb.enable=false" \
    2>&1 | tee "$OUT_ROOT/train3cam.log" | grep -E 'loss:|Checkpoint|Error' || true
  report \
    "$RUN/checkpoints/00$TRAIN_STEPS/pretrained_model/ （~8.8GB）
       config.json + model.safetensors + policy_pre/postprocessor（含归一化统计）
       训练日志: $OUT_ROOT/train3cam.log
       作用: 微调后的策略 checkpoint，推理 server 加载它" \
    "bash $0 rollout  —— 起推理 server + hybrid rollout 出视频"
}

# ----------------------------- STEP 5 推理展示 -----------------------------
rollout() {
  step 5 "推理展示（policy server + hybrid rollout）"
  local CKPT=/hxb/smoke_outputs/pi05_ecu_smoke3cam/checkpoints/00$TRAIN_STEPS/pretrained_model
  [ -d "/mnt/my_hdd/hxb/smoke_outputs/pi05_ecu_smoke3cam/checkpoints/00$TRAIN_STEPS/pretrained_model" ] \
    || { echo "checkpoint 不存在，先跑 train"; exit 1; }

  docker rm -f ecu_policy_server >/dev/null 2>&1 || true
  echo "启动 policy server（GPU $GPU_POLICY，端口 $PORT）..."
  docker run --rm -d --gpus all --ipc=host --name ecu_policy_server -p $PORT:$PORT --entrypoint bash \
    -v /mnt/my_hdd/hxb:/hxb \
    -v $BASELINE/pretrained/paligemma-3b-pt-224:/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/pretrained/paligemma-3b-pt-224:ro \
    -w /hxb/Baseline \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/hxb/smoke_data/hf_cache -e PYTHONPATH=/hxb/Baseline \
    -e CUDA_VISIBLE_DEVICES=$GPU_POLICY -e POLICY_CKPT=$CKPT -e POLICY_PORT=$PORT \
    "$DOCKER_IMG" -c "/isaac-sim/python.sh -m pip install -q /hxb/cruzr_mujoco_sim/smoke/openpi_client-0.1.2-py3-none-any.whl && \
                      /isaac-sim/python.sh /hxb/cruzr_mujoco_sim/smoke/ecu_lerobot_policy_server.py"
  for i in $(seq 1 60); do
    docker logs ecu_policy_server 2>&1 | grep -q 'listening' && break
    sleep 5
  done
  docker logs ecu_policy_server 2>&1 | grep -q 'listening' || { echo "server 启动失败:"; docker logs ecu_policy_server | tail -20; exit 1; }
  echo "server 就绪"

  set +e
  # SPAWN_SEED 必须落在训练集内（本 smoke 的成功 episode 来自 seed 2/3，seed 1 未入集）
  mkdir -p out/logs/smoke out/rollout/ecu
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_SIM TELEOP_RECORD_GPU=$GPU_SIM \
  POLICY_PORT=$PORT ROLLOUT_SPAWN_SEED=${ROLLOUT_SPAWN_SEED:-2} ROLLOUT_BAY=$EP_BAY ROLLOUT_STAGE_TIMEOUT=70 \
  ROLLOUT_OUT=out/rollout/ecu/hybrid_rollout \
    "$MJX_PY" scripts/ecu_hybrid_rollout.py 2>&1 | tee out/logs/smoke/_smoke_rollout.log
  set -e
  docker rm -f ecu_policy_server >/dev/null 2>&1 || true

  report \
    "out/rollout/ecu/hybrid_rollout/hybrid_top_head.mp4  —— rollout 全程视频（top_head 视角）
       out/logs/smoke/_smoke_rollout.log               —— M1~M4 各阶段结果
       作用: 直观检验策略行为；注意 smoke 数据量极小，动作不稳/失败是正常的" \
    "想真能用: 参照 ../docs/archive/ECU旧流程手册.md 采 100+ 条多 bay 多 seed 数据，步数加大到 20k+"
}

clean() {
  docker rm -f ecu_policy_server ecu_smoke_train 2>/dev/null || true
  rm -rf "$OUT_ROOT/pi05_ecu_smoke3cam" "$DATA_ROOT/$DATASET" "$DATA_ROOT/${DATASET}_old"
  echo "已清理训练输出和数据集（录制的原始 episode 保留）"
}

case "${1:-all}" in
  env)     check_env ;;
  record)  check_env; record ;;
  split)   split ;;
  dataset) dataset ;;
  train)   train ;;
  rollout) rollout ;;
  clean)   clean ;;
  all)     check_env; record; split; dataset; train; rollout ;;
  *) echo "用法: bash $0 [env|record|split|dataset|train|rollout|clean|all]"; exit 1 ;;
esac
