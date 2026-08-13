#!/bin/bash
# =============================================================================
# 立柱线迁移打包：把「在另一台服务器直接开干」所需的一切打成 3 个 tar 包。
#
# 前提（已确认）:
#   - 目标服务器已有 ghrc_2026 docker 镜像 + NVIDIA 驱动 -> 训练/推理环境不打
#   - 只做 BC 闭环（采集 -> π0.5 BC 训练 -> 推理视频），不含在线 RL 支线
#
# 产出（默认到 /mnt/my_hdd/hxb/pillar_bundle/）:
#   1) mjx.tar.gz       仿真环境（conda-pack，mujoco 3.9.0），唯一要解压激活的环境
#   2) code.tar.gz      cruzr_mujoco_sim（含立柱线资产与新 pillar server）+ Baseline/src + HF 缓存
#   3) weights.tar.gz   pi05_base_remapped + paligemma-3b-pt-224（π0.5 训练起点 + VLM 骨干）
#   + MANIFEST.md       目标机使用手册（解压顺序 + 立柱线采集/训练/推理命令）
#
# 用法: bash smoke/package_for_pillar.sh [输出目录]
# =============================================================================
set -euo pipefail

OUT=${1:-/mnt/my_hdd/hxb/pillar_bundle}
PKG=/mnt/my_hdd/hxb/cruzr_mujoco_sim
BASELINE=/mnt/my_hdd/hxb/Baseline
MJX=/home/lh/miniconda3/envs/mjx
HF_CACHE=/mnt/my_hdd/hxb/smoke_data/hf_cache

mkdir -p "$OUT"
cd "$OUT"

echo "===== [1/4] 打包仿真环境 mjx (conda-pack) ====="
# conda-pack 需要先装在 base；若没有则用 pip 装到临时环境
if ! /home/lh/miniconda3/bin/conda-pack --help >/dev/null 2>&1; then
  echo "安装 conda-pack 到 base..."
  /home/lh/miniconda3/bin/pip install -q conda-pack
fi
/home/lh/miniconda3/bin/conda-pack -n mjx -o mjx.tar.gz --compress-level 1
echo "  -> $(du -sh mjx.tar.gz | cut -f1)"

echo "===== [2/4] 打包代码 + 资产 + HF 缓存 ====="
# cruzr_mujoco_sim 排除运行时大数据目录（out/ 里是录制产物，不带）
tar --exclude='./cruzr_mujoco_sim/out' \
    --exclude='./cruzr_mujoco_sim/**/__pycache__' \
    -czf code.tar.gz \
    -C /mnt/my_hdd/hxb \
      cruzr_mujoco_sim \
      -C "$BASELINE" src \
      -C /mnt/my_hdd/hxb smoke_data/hf_cache
echo "  -> $(du -sh code.tar.gz | cut -f1)"

echo "===== [3/4] 打包权重 ====="
tar -czf weights.tar.gz \
    -C "$BASELINE/pretrained" \
      pi05_base_remapped \
      paligemma-3b-pt-224
echo "  -> $(du -sh weights.tar.gz | cut -f1)"

echo "===== [4/4] 生成 MANIFEST.md ====="
cat > MANIFEST.md <<'MANIFEST_EOF'
# 立柱线迁移包使用手册

本包含 3 个 tar 包，在目标服务器上解压即可直接开始立柱线（pillar / shelf_e2e）的
「采集 → π0.5 BC 训练 → 推理视频」全流程。**前提：目标机已有 ghrc_2026 docker 镜像 + NVIDIA 驱动。**

## 解压（约 5 分钟）

```bash
mkdir -p ~/pillar && cd ~/pillar
# 1. 仿真环境（唯一要"装"的）
mkdir -p envs/mjx && tar -xzf mjx.tar.gz -C envs/mjx
./envs/mjx/bin/conda-unpack          # 修复路径前缀（conda-pack 标准步骤）
# 2. 代码 + HF 缓存
tar -xzf code.tar.gz                 # 得到 cruzr_mujoco_sim/ 、src/ 、smoke_data/
# 3. 权重
mkdir -p pretrained && tar -xzf weights.tar.gz -C pretrained
```

解压后目录约定（后续命令以此为基准，可改但要同步改环境变量）：
```
~/pillar/
├── envs/mjx/                  # 仿真 python
├── cruzr_mujoco_sim/          # 仿真包（立柱线脚本+资产+server）
├── src/                       # Baseline lerobot 框架（容器内 PYTHONPATH 指向它的上一级）
├── smoke_data/hf_cache/       # HF 离线缓存
└── pretrained/
    ├── pi05_base_remapped/    # π0.5 训练起点
    └── paligemma-3b-pt-224/   # VLM 骨干
```

## 立柱线三步走

### ① 采集数据（mjx 环境，GPU 渲染）

```bash
PKG=~/pillar/cruzr_mujoco_sim; cd $PKG
MJX=~/pillar/envs/mjx/bin/python
# 当前双物料单条自检（seed 1，不落视频）
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 SEED=1 E2E_NOREC=1 E2E_KICKS=0 \
  $MJX scripts/collection/shelf_e2e_dual_expert.py
# 批量产数据：<目标成功数> <并发> <GPU>
bash scripts/collection/shelf_e2e_batch.sh 50 4 0
```
产出根：`out/teleop/shelf_e2e_dual/`（meta.json + npz + 3 路 224×224 相机帧）。
当前严格任务准入仍未通过；正式采集前必须按 `scripts/README.md` 重新执行预检。

### ② 建数据集 + 训练（docker 容器）

```bash
# 建 LeRobot 数据集（v2.1），再去 docker 里转 v3.0 + 补分位数 + 训练
# —— 详细命令参照 cruzr_mujoco_sim/smoke/run_pillar_smoke.sh（立柱版一键脚本）
```

### ③ 推理 + 视频（mjx 连 docker server）

```bash
# 起 server（容器内）: 加载训练 checkpoint，端口 8731
# 跑 rollout（mjx）: scripts/collection/shelf_e2e_rollout.py，端到端单策略，录视频
```

## 与 ECU 线的关键差异（已知，脚本已处理）

| 项 | ECU | 立柱线 |
|---|---|---|
| 状态维度 | 21 | **18**（16 臂/夹爪 + 2 底盘速度）→ 训练设 `LEROBOT_STATE_N_DIMS=18` |
| 相机 | 历史 ECU 配置 | stereo_left / waist_front / chassis_front（224×224） |
| rollout | 混合式（导航脚本+策略） | **端到端**（单策略连底盘都驱动），端口默认 8731 |
| 推理 server | ecu_lerobot_policy_server.py | **pillar_lerobot_policy_server.py** |

## 环境要点（从 ECU smoke 继承）

- mujoco 必须 **3.9.0**（mjx 环境已锁定）
- 训练用 **pi05_base_remapped**（视觉键名已修复），不要用 pi05_base
- HF 离线：`HF_HUB_OFFLINE=1` + `HF_HOME=~/pillar/smoke_data/hf_cache`
- docker 需 `--ipc=host`；paligemma 挂到容器写死路径 `/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/pretrained/paligemma-3b-pt-224`
MANIFEST_EOF

echo
echo "===== 打包完成 ====="
ls -lh "$OUT"/{mjx,code,weights}.tar.gz "$OUT"/MANIFEST.md
echo
echo "下一步: 把 $OUT 整个目录传到目标服务器（scp/rsync/U盘），按 MANIFEST.md 解压即用。"
