# Sorting Roll 正式数据采集与 π0.5 训练指南

> 状态：2026-08-27。v15 数据流水线已完成并通过准入。暂存点后移 10 mm 后，最终 5×20 fresh 准入分别为 19/20、20/20、18/20、20/20、20/20，所有组均达到至少 18/20；正式首轮采集 293/300 成功，7 个同配额补采全部成功，最终选择 300 个唯一成功回合。源 validator 为 300/300，LeRobot v3.0 审计为 `passed=true`、`errors=[]`。4×H100 全参数 fresh/resume canary 已通过，28,000-step 训练已正常完成，当前正从完整 `028000` checkpoint 低学习率续训到 42,000 step。

> 4×H100 执行状态：28k 以退出码 0 完成，最终 `loss=0.002`、`grad_norm=0.053`、`lr=2.5e-6`，训练循环用时约 12 小时 16 分。42k 续训由 tmux `sorting_roll_v15_h100x4_fullft42k_resume` 启动；首个新日志点对应 step 28,010，`loss=0.001`、`grad_norm=0.046`，可训练参数与总参数均为 4,143,404,816，四卡显存约 51.1 GB，未见 OOM、NaN 或 Inf。

v15 从“安装在左右手局部坐标系中的同一 D405 变换”重新推导初始姿态：双臂直接初始化在前向、左右镜像且可观察自身夹爪的任务就绪位，不再执行原来前 25 秒的抬臂绕行。高风险 heavy/low-friction seed 10084 在 50.517 秒内成功，早期 806 次碰撞检查无事件；validator 1/1 通过，相机总覆盖和必需角色覆盖均为 100%，平均每个关键样本有 2.78 路可用视角。棒子最终由一体式顶层槽提供四点接触，松手后稳定 2 秒且双手撤回。

## 0. v15 当前执行门

当前定版结果：

- 专家动作：`sorting_roll_v15_d405_isomorphic_forward_park_sim`
- 多样性数据：`sorting_roll_v15_diverse_sim`
- 相机配置：`sorting_roll_d405_candidate_v6`
- 策略输入：`stereo_left + left_wrist_realsense + right_wrist_realsense`
- 采集结果：8×RTX 4090 已完成；最终 300 个成功回合、458,786 帧，train/val/test 为 240/30/30。
- 数据格式：LeRobot v3.0；三路 H.264/yuv420p RGB、224×224@30 FPS；state/action 均为 18 维 float32。
- 数据集：`cruzr_mujoco_sim/out/datasets/sorting_roll_v15_diverse300_20260826_8x4090_lerobot_v30`
- 审计：`log/sorting_roll_v15_diverse300_20260826_8x4090/dataset_v30_audit.json`
- 准入：`cruzr_mujoco_sim/output/sorting_roll_expert/sorting_roll_v15_diverse300_20260826_8x4090/data_training_readiness.json`
- 代表视频：`cruzr_mujoco_sim/output/sorting_roll_expert/sorting_roll_v15_diverse300_20260826_8x4090/review_bundle/seed_3090/sorting_roll_robot_multiview.mp4`
- 正式训练：4×H100 80GB、每卡 batch 16、有效 batch 64、BF16、gradient checkpointing、`train_expert_only=false`；28k 已完成，从原 optimizer/scheduler 状态续训到 42k。原 cosine schedule 在 28k 后保持 `2.5e-6` 下限，不重新 warmup 或抬高学习率；每 1,000 step 保存完整 checkpoint，并保留 28k 作为回退与对比基线。

进入正式训练的固定顺序：

1. v15 5×20 准入，每组至少 18/20，并通过 validator 和相机门。
2. 重新采集 v15 多样性 300 回合；旧 v13 回合不得复用。
3. 构建并审计 LeRobot v3.0：三路 H.264/yuv420p RGB、30 FPS、224×224，state/action 均为 18 维 float32，train/val/test 为 240/30/30。
4. 4×H100 全参 fresh 200-step canary，再 resume 到 250 step；核对显存、有限 loss/gradient 和 checkpoint 完整性。
5. 只有上述证据全部通过，才启动约 20 小时、28k step 的正式全参训练。

后续训练入口：

```bash
# 推荐：一条命令启动 tmux 总控；硬门失败时不会进入正式训练
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_auto20h.sh start

# 查看总控、canary 或正式训练状态
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_auto20h.sh status

# 换到 4×H100 80GB 后；先检查硬件和 dry-run
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_fullft20h.sh hardware-check
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_fullft20h.sh canary-dry-run

# 依次 fresh 200 step、resume 到 250 step、审计；每个 tmux 阶段结束后再执行下一条
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_fullft20h.sh tmux-canary
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_fullft20h.sh tmux-canary-resume
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_fullft20h.sh canary-audit
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_fullft20h.sh recommend-20h

# 最终 dry-run 通过后，才启动约 20 小时的 28k 全参数训练
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_fullft20h.sh dry-run
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_fullft20h.sh tmux-start

# 查看状态；SSH/进程中断时从最近完整 checkpoint 恢复
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_fullft20h.sh status
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_fullft20h.sh tmux-resume

# 28k 完成后的 42k 低学习率延长；dry-run 或从最近完整 checkpoint 启动/再恢复
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_fullft20h.sh extend42k-dry-run
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_fullft20h.sh tmux-extend42k
```

总控脚本按顺序等待 fresh canary、resume canary 和审计全部成功，再启动正式训练；任一阶段失败都会停止。正式训练由独立 supervisor 脱离 SSH，且每 1,000 step 保存完整 checkpoint。中断后可执行：

```bash
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v15_h100x4_auto20h.sh resume-formal
```

`data_training_readiness.json` 和 `log/sorting_roll_v15_diverse300_20260826_8x4090/pi05_h100x4_fullft_canary_audit.json` 均已通过，42k 续训正在运行。完成后必须使用相同 val/test seed 对比 28k、35k 和 42k 闭环成功率；训练 loss 更低不能单独证明策略更好。本文后续出现的 v13/v10 路径、expert-only checkpoint 和采集模板均为历史流程说明，禁止作为当前启动命令。

## 1. 已定版的基线与版本边界

- `sorting_roll_v9_d405_sim` 是历史动作与一体式槽位基线。
- `sorting_roll_v10_diverse_sim` 是已完成 300 回合与短训练验证的历史多样性版本。
- `sorting_roll_v11_d405_upright_support_sim` 是历史视觉基线，只使用旧诊断相机代理。
- `sorting_roll_v12_d405_bracket_mount_sim` 是被拒绝的安装图初始候选；物理成功不能覆盖其插入/释放阶段的可观测性缺陷。
- `sorting_roll_v13_d405_rearward_mount_sim` / `sorting_roll_v13_diverse_sim` 是历史基线，不再用于新数据或新训练。
- `sorting_roll_v15_d405_isomorphic_forward_park_sim` / `sorting_roll_v15_diverse_sim` 是当前候选基线：左右腕各使用独立 D405，相机父节点分别为 `L_pgc140_mount` / `R_pgc140_mount`，两个侧特定安装坐标系采用相同局部外参。
- v15 D405 局部外参为 `pos=(0, -0.180, 0.070) m`、`quat_wxyz=(0.5, 0.8660254, 0, 0)`；仿真硬门要求初始相机和夹爪都在前方、左右位置镜像误差不超过 12 mm、光轴镜像误差不超过 1°，并能在 D405 视场和 7–50 cm 工作距离内看到各自夹爪。该值尚未由实机 CAD 或尺量标定。
- v9、v10、v11、v12、v13、v15 不得混入同一个 campaign 或 dataset；任何策略图像、场景几何、相机外参或初始动作变化都必须提升版本并重新准入。
- 策略输入固定为三路 RGB：
  - `observation.images.stereo_left`
  - `observation.images.left_wrist_realsense`
  - `observation.images.right_wrist_realsense`
- 第三视角和槽位近景只用于人工审核，不进入 π0.5；`sorting_roll_robot_multiview.mp4` 固定为上方 `stereo_left`、下方左腕和右腕。
- 源画面默认 `640×360 @ 30 FPS`；构建时保持长宽比并补黑边到 `224×224`，禁止横向拉伸。
- 深度可以另行保存，但当前不进入 π0.5。
- 每个 seed 对应一个连续 episode；禁止把一个回合切成多个 episode，也禁止重复 seed。
- 只有成功回合进入训练集。失败、超时、人工审核视频和诊断回合均不得混入。
- 当前仿真元数据中的 `training_eligible=false` 表示尚未取得真机部署资格，不妨碍进行仿真 π0.5 实验；不得将它改写成真机已验证。

成功回合必须同时满足：

1. 棒子两端均在顶层一体式槽内。
2. 棒子与槽底或支撑面真实接触，不能浮空。
3. 松手后稳定至少 2 秒。
4. 双手已经撤回，且棒子不再受夹爪或桌面支撑。
5. 线速度、角速度和轴向误差均通过物理门限。
6. 单回合不超过 60 秒。

## 2. 正式采集目标

v15 第一阶段重新采集 **300 个通过 validator 的多样化成功回合**：

- train：240 个；seed 末位不是 `0` 或 `1`。
- val：30 个；seed 末位为 `1`。
- test：30 个；seed 末位为 `0`。
- v15 正式 seed 使用新的独立区间；补采 seed 不能复用失败 seed，也不能与 v13 重号后混写到同一 campaign。
- 初始尝试成功率至少 `90%`；最终构建必须恰好选出 300 个成功且互不重复的源回合。

其中 240 个 train 回合进入优化器，val/test 只用于选型和最终评测；把全部 300 个回合都用于梯度更新会造成评测泄漏。旧版本数据继续作为历史证据，不自动混入 v15。300 回合用于验证数据、训练和闭环评测管线，不代表数据量一定足够。只有测试集闭环成功率和分层成功率达标后，才扩大到 1,000 回合以上。

## 3. 多样性要求

### 3.1 基础位姿随机化

当前 v13 保留以下确定性 seed 随机化：

| 维度 | 当前范围 |
| --- | --- |
| 机器人基座 X/Y | 各 `±15 mm` |
| 机器人基座 yaw | `±0.025 rad`，约 `±1.43°` |
| 棒子 X/Y | 各 `±4 mm` |
| 棒子 yaw | `±0.012 rad`，约 `±0.69°` |

这些基础位姿变化足以做 pilot，但不足以满足“多种类数据”的正式要求。因此，正式采集必须使用下一节的分层 manifest。

### 3.2 v13 沿用的分层配置

`sorting_roll_v13_diverse_sim` 沿用已经验证过的 v10 多样性范围和配额，只把任务基线切换为 v13 的对称 D405 安装外参、任务专用相机支架及其碰撞代理；不改变任务语义、成功标准和专家动作结构。

300 个回合应由 campaign manifest 预先分配，而不是事后查看随机结果。以下每一行都是覆盖约束，各维度可以交叉组合：

| 维度 | 300 回合最低覆盖 | 约束 |
| --- | --- | --- |
| 初始位姿难度 | easy 120 / medium 120 / boundary 60 | 均在已验证可达范围内；boundary 不能越过安全边界 |
| 棒子几何类型 | 3 类，各 100 | 候选长度为架子可用内宽的约 `82% / 85% / 88%`；精确直径、重量必须记录 |
| 棒子外观 | red/orange/yellow/green/blue，各 60 | 使用无纹理纯色材质；颜色不能表达任务语义 |
| 光照 | normal 180 / dim 60 / bright 60 | 相机必须仍能观察抓取、运输、插入和释放阶段 |
| 动力学 | nominal 180 / 两个边界组各 60 | 质量、摩擦候选变化先限制在基准值的 `±15%`，每组单独准入 |
| 图像编码 | JPEG quality 92：180 / 84：60 / 76：60 | 只改变压缩质量；相机位姿、分辨率和帧率不得变化 |
| 任务指令 | 5 个同义英文指令，各 60 | 语义完全等价，不能引入新任务 |

精确候选值如下：

| profile | 长度 | 直径 | 质量 | 滑动摩擦 |
| --- | ---: | ---: | ---: | ---: |
| `short_slim + nominal` | 467.4 mm | 22.5 mm | 0.2500 kg | 1.2500 |
| `medium + nominal` | 484.5 mm | 24.0 mm | 0.2500 kg | 1.2500 |
| `long_baseline + nominal` | 500.0 mm | 24.0 mm | 0.2500 kg | 1.2500 |
| `long_baseline + light_high_friction` | 500.0 mm | 24.0 mm | 0.2125 kg | 1.4375 |
| `long_baseline + heavy_low_friction` | 500.0 mm | 24.0 mm | 0.2875 kg | 1.0625 |

长棒直径保持已验证的 24 mm；25 mm 在边界姿态会把腕部垫片净空降到 2 mm 以下，因此不进入当前 campaign。未来真机范围已知后应使用实测值另起版本，不要制造现实中不存在的样本。

### 3.3 多样性准入门

每一种新增棒子几何或动力学 profile 在进入 300 回合前，都必须独立完成：

1. 20 个新 seed 的带渲染测试。
2. 至少 `18/20` 成功。
3. 三路策略相机通过可观测性审核。
4. 至少人工抽检 3 个完整视频，其中必须包含一个 boundary 回合。
5. validator 全部通过，且动作仍然自然、平稳、低于 60 秒。

任何 profile 未通过时，只剔除该 profile，不得降低物理成功标准来保留它。

单卡准入入口：

```bash
ROOT=/share/home/tm1128689517650000/a852937540/cruzr_sim
ADMISSION_ROOT=$ROOT/cruzr_mujoco_sim/output/sorting_roll_expert/v13_diverse_admission_20260825
tmux new-session -d -s sorting_roll_v13_admission \
  "cd '$ROOT' && bash cruzr_mujoco_sim/scripts/collection/sorting_roll_v13_admission.sh \
    --out-root '$ADMISSION_ROOT' --gpu 0 > '$ADMISSION_ROOT/admission.log' 2>&1"
```

该脚本按风险优先顺序运行 5 组 × 20 seed；每组要求至少 18/20、validator 全通过，并审核至少 3 个三路相机回合。只有 `admission_report.json` 中 `passed=true` 才能进入正式 300 回合。

## 4. 4 卡与 8 卡的使用原则

- 300 回合优先使用 **4×4090**：每卡一个 EGL 渲染 shard，成本和吞吐更均衡。
- 8×4090 只在以下情况使用：目标扩大到 1,000 回合以上、4 卡预估无法满足截止时间，或已经确认 CPU、文件系统和 JPEG 写入不是瓶颈。
- 单张 4090 同时只运行一个渲染 shard。增加同卡并发通常会放大显存、EGL 和 I/O 风险。
- 多卡只缩短采集墙钟时间，不改变 seed、数据分布、成功标准或格式。
- 本机 GPU 0–3 的 4×4090 已完成 v13 的 5×20 准入、正式 300 回合采集和 20→40-step DDP canary。若迁移到 8 卡机器，还必须重新执行环境检查和 DDP canary。

## 5. 采集前检查

从仓库根目录执行：

```bash
cd /share/home/tm1128689517650000/a852937540/cruzr_sim
bash Sorting_Roll/run_scene.sh check
cd cruzr_mujoco_sim
../envs/mjx/bin/python -m unittest discover -s scripts/tests -p 'test_*.py'
cd ..
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
```

只有以下条件全部满足才能启动：

- 场景检查退出码为 0。
- 完整测试全部通过。
- 能看到计划使用的 GPU ID。
- `campaign_manifest.json` 恰好包含 300 个唯一 seed，并满足第 3 节的所有配额。
- 每个 manifest 记录都包含 `seed`、`split`、`object_profile`、`pose_bin`、`lighting_profile`、`dynamics_profile`、`image_profile` 和 `prompt_id`。
- 采集器、validator 和 LeRobot 构建器使用同一个 task/profile 版本。

## 6. tmux 多卡采集模板

以下模板适用于 4 卡或 8 卡。修改 `GPU_COUNT` 即可；300 个 seed 会无重叠地均分到各卡，所有 shard 共用一个只读 manifest。

```bash
ROOT=/share/home/tm1128689517650000/a852937540/cruzr_sim
PY=$ROOT/envs/mjx/bin/python
GPU_COUNT=4
TOTAL=300
FIRST_SEED=2000
CAMPAIGN=sorting_roll_v13_diverse300_$(date -u +%Y%m%d)
RAW_ROOT=$ROOT/cruzr_mujoco_sim/output/sorting_roll_expert/$CAMPAIGN
LOG_ROOT=$ROOT/log/$CAMPAIGN
MANIFEST=$RAW_ROOT/campaign_manifest.json

mkdir -p "$RAW_ROOT" "$LOG_ROOT"
"$PY" cruzr_mujoco_sim/scripts/core/sorting_roll_diversity.py generate \
  --out "$MANIFEST" --campaign "$CAMPAIGN" \
  --seed-start "$FIRST_SEED" --count "$TOTAL"
"$PY" cruzr_mujoco_sim/scripts/core/sorting_roll_diversity.py \
  check "$MANIFEST"

base_count=$((TOTAL / GPU_COUNT))
extra=$((TOTAL % GPU_COUNT))
offset=0

for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
  count=$base_count
  ((gpu < extra)) && count=$((count + 1))
  seed_start=$((FIRST_SEED + offset))
  min_success=$(((count * 9 + 9) / 10))
  shard=$RAW_ROOT/shard_$gpu
  log=$LOG_ROOT/shard_$gpu.log
  session=${CAMPAIGN}_g$gpu

  tmux new-session -d -s "$session" \
    "cd '$ROOT' && '$PY' cruzr_mujoco_sim/scripts/collection/sorting_roll_batch.py \
      --out-root '$shard' --seed-start '$seed_start' --count '$count' \
      --min-success '$min_success' --gpu '$gpu' --timeout 1800 --render \
      --resume --manifest '$MANIFEST' \
      > '$log' 2>&1"
  offset=$((offset + count))
done
```

4 卡时每卡 75 次尝试；8 卡时前四卡各 38 次、后四卡各 37 次。tmux 创建成功后可以断开 SSH。

监控命令：

```bash
tmux list-sessions
nvidia-smi
tail -f "$LOG_ROOT/shard_0.log"
```

判断进度必须读取各 shard 的 `summary.json`，不能只看目录数量。SSH 或进程中断后使用相同参数加 `--resume`；如果存在没有 `result.json` 的半成品 seed 目录，先移动到 campaign 下的 `rejected_incomplete/` 留证，再恢复，不能直接覆盖。

## 7. 成功源筛选与补采

所有 shard 结束后，从 `summary.json` 只提取 `passed=true` 的源：

```bash
SELECTED=$RAW_ROOT/selected_sources.txt
"$PY" - "$RAW_ROOT" "$SELECTED" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
output = Path(sys.argv[2])
records = []
for summary_path in sorted(root.rglob("summary.json")):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records.extend(item for item in summary["records"] if item.get("passed"))

by_seed = {}
for item in records:
    seed = int(item["seed"])
    if seed in by_seed:
        raise SystemExit(f"duplicate successful seed: {seed}")
    by_seed[seed] = str(Path(item["episode"]).resolve())

output.write_text(
    "".join(f"{by_seed[seed]}\n" for seed in sorted(by_seed)),
    encoding="utf-8",
)
print(f"selected successes: {len(by_seed)}")
PY
wc -l "$SELECTED"
```

不足 300 个时，从 seed 2300 开始使用独立的 `replacement_*` shard 补采。补采也必须满足 campaign manifest 的缺失配额；不能只补最容易成功的类别。最终应生成恰好 300 行的新 `selected_sources.txt`。

## 8. 源数据 validator

```bash
mapfile -t SOURCES < "$SELECTED"
REPORT=$RAW_ROOT/validation_report.json
"$PY" cruzr_mujoco_sim/scripts/collection/sorting_roll_validate.py \
  "${SOURCES[@]}" --manifest "$MANIFEST" --report "$REPORT"
```

必须看到：

```text
episode_count=300
passed_count=300
failed_count=0
passed=true
```

v13 validator 会核对 manifest 与 episode assignment、实际应用的物理/视觉参数、prompt 和 campaign 完全一致，并输出各 diversity stratum 的数量。只要有重复 seed、混合版本、混合 campaign、缺失类别或分层偏斜，就停止构建。

## 9. π0.5 数据格式契约

每个源 episode 必须包含：

```text
seed_xxxx/
├── meta.json
├── result.json
├── episode_data.npz
├── sdk_timestamps.npz
└── frames/
    ├── stereo_left/frame_000000.jpg ...
    ├── left_wrist_realsense/frame_000000.jpg ...
    └── right_wrist_realsense/frame_000000.jpg ...
```

格式门槛：

- 三路 RGB 帧数、状态数和动作数完全相等，编号连续。
- 30 FPS 时间戳严格递增，三路相机相对状态的最大时间偏差不超过 20 ms。
- 源 `state/action` 各 16 维，另有 2 维 `base_velocity/base_action`。
- LeRobot 中合并为 `observation.state/action` 各 18 维 `float32`，不能包含 NaN 或 Inf。
- 三路视频均为 `224×224`、H.264、`yuv420p`、30 FPS，帧数和 PTS 必须精确一致。
- LeRobot 最终版本为 v3.0；一个源回合只对应一个 dataset episode。
- prompt、seed、profile、任务版本、多样性标签和 split 必须可追溯回源数据。

构建 v2.1 staging 数据集：

```bash
DATASET_V21=$ROOT/cruzr_mujoco_sim/out/datasets/sorting_roll_v13_diverse300_lerobot_v21
"$PY" cruzr_mujoco_sim/scripts/collection/sorting_roll_build_v21.py \
  "${SOURCES[@]}" --out "$DATASET_V21" --encode-workers 4 \
  --manifest "$MANIFEST"
```

转换为 LeRobot v3.0：

```bash
ISAAC_PY=/isaac-sim/python.sh
REPO_ID=local/sorting_roll_v13_diverse300
DATASET=$ROOT/cruzr_mujoco_sim/out/datasets/sorting_roll_v13_diverse300_lerobot_v30
cp -a "$DATASET_V21" "$DATASET"
PYTHONPATH=. "$ISAAC_PY" \
  src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
  --repo-id "$REPO_ID" --root "$DATASET" --push-to-hub false
```

转换脚本会原地修改目标，因此先复制并保留 `$DATASET_V21`。转换后必须核对 `meta/info.json` 和统计文件，并用真实 `LeRobotDataset` 解码首帧、尾帧及每个 diversity stratum 的随机样本。v3 审计通过前不要删除 v2.1 staging 备份。

v13 实际审计结果：300 episodes、519,776 frames，train/val/test 为 240/30/30；三路视频均为 H.264、`yuv420p`、`224×224 @ 30 FPS`，每路均有 519,776 帧；`observation.state` 与 `action` 均为 18 维 `float32`，全量 Parquet 数值有限。审计实际通过 `LeRobotDataset` 解码 8 个分层 episode 的 24 个样本，未发现错误。审计报告位于 `log/sorting_roll_v13_diverse300_20260825_4gpu/dataset_v30_audit.json`。

## 10. π0.5 训练前 canary

先用 4×4090 做 20-step expert-only DDP canary；4090 上不要直接照搬 H100 的每卡 batch。当前已验证的安全起点是每卡 batch 1、BF16 和 gradient checkpointing。

```bash
bash pi05_train.sh dry-run \
  --dataset-root "$DATASET" --repo-id "$REPO_ID" --episodes train \
  --gpu-ids 0,1,2,3 --num-processes 4 \
  --batch-size 1 --num-workers 2 --allow-small-batch true \
  --steps 20 --warmup-steps 2 --decay-steps 20 \
  --save-freq 20 --train-expert-only true \
  --output-dir cruzr_mujoco_sim/out/training/pi05_sorting_roll_v13_canary \
  --log-file log/pi05_sorting_roll_v13_canary.log
```

`dry-run` 通过后把动作改为 `start`。canary 必须满足：

- 20 步正常退出，无 OOM、NaN、Inf 或视频解码错误。
- loss 和梯度范数全部有限。
- checkpoint 完整可读取，并能从该 checkpoint 恢复到至少第 40 步。
- 训练只使用 train split，val/test 不得进入优化器。

v13 已只使用 train split 的 240 个回合完成 4×4090 canary：fresh 1→20 和 checkpoint resume 21→40 均以退出码 0 完成，step 20/40 的模型、优化器和 RNG safetensors 均完整可读。40 个测量全部有限；step 32 曾出现有限的 loss/gradient 瞬时峰值，下一步即恢复，未触发 NaN/Inf 或训练失败。正式训练需继续监控此指标，若峰值持续或发散则按第 12 节停止。

正式训练使用专用入口；默认动作是 `dry-run`，不会启动训练：

```bash
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_train.sh dry-run
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_train.sh start
```

入口固定检查 v13 数据版本和审计证据，默认使用 4×4090、每卡 batch 1、BF16、gradient checkpointing、expert-only，训练 10,000 steps，每 500 steps 保存。底层启动器使用 `nohup + setsid` 脱离 SSH；断开连接不会终止训练。查看状态或断点续训：

```bash
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_train.sh status
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_train.sh resume
```

`resume` 必须保留 `--allow-small-batch true`；专用入口已经固定传入该参数。不要改用不属于本任务的 `pi05_formal300_train.sh`。

只有 4 卡 canary 稳定后才考虑 8 卡。8 卡需要重新做 DDP canary；不得假定卡数翻倍就一定更省钱或更快。

### 10.1 当前采用的 2×H100 正式训练配置

当前硬件已切换为两张 H100 80GB，使用专用入口：

```bash
# 只检查，不启动
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh dry-run

# 用户确认后才执行
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh start

# 启动后查看状态或从最近完整 checkpoint 恢复
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh status
bash cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh resume
```

固定配置为：2 个 DDP 进程、每卡 batch 16、有效 batch 32、每进程 8 个 DataLoader worker、BF16、gradient checkpointing、`train_expert_only=true`、20,000 steps、warmup 1,000 steps、每 1,000 steps 保存。训练参数总数为 4,143,404,816，可训练 expert 参数为 693,422,112；因此每卡约 20.9 GiB 显存是正常结果，不能以“填满显存”为目标擅自改为全参数训练。

本机实测结果：fresh 1→200 正常结束，loss 从 2.610 降至 0.170，无 OOM、NaN 或 Inf；step 200 的模型、优化器和 RNG 状态均完整可读。随后从 step 200 恢复到 step 250，loss 保持在 0.160–0.180、梯度范数保持有限，最终 checkpoint 完整。8 workers 预热后的 `data_s` 为 0.422–0.790 秒，优于 4 workers 常见的 1.2–3.6 秒，因此正式配置保留 8 workers。

截至本文更新时，正式 20k 输出目录尚未创建，正式训练进程未运行。只有用户明确确认后才执行 `start`；底层使用 `nohup + setsid`，启动成功后可以断开 SSH。

## 11. 扩大数据和正式训练的进入条件

300 回合训练完成后，先在未见过的 test seed 上做至少 50 次闭环仿真评测：

- 总成功率至少 `90%`。
- 每个棒子几何类型成功率至少 `85%`。
- 每个 lighting/dynamics 边界组成功率至少 `80%`。
- 没有碰撞、安全门失效、浮空或错误成功判定。
- 单回合仍不超过 60 秒。

满足后才扩大到 1,000 回合以上。若某个分层明显较弱，优先补该分层的高质量成功示范，不要无差别堆叠 nominal 数据。

## 12. 必须停止的情况

遇到以下任一情况立即停止采集或训练：

- 相机名称、顺序、外参、分辨率或 FPS 与定版契约不一致。
- task/profile 版本发生变化，却仍写入同一个 campaign 或 dataset。
- 初始采集成功率低于 90%。
- validator 不是全通过，或 manifest 配额不完整。
- 出现重复 seed、train/val/test 泄漏、视频缺帧或时间戳偏差超限。
- v3 数据不能被 `LeRobotDataset` 解码。
- 训练出现 OOM、NaN、Inf、数据加载错误或不完整 checkpoint。

修复后必须使用新 campaign 名称重新验证；不要覆盖旧数据来掩盖失败。

## 13. 当前保留的定版证据

v13 定版证据：

- `cruzr_mujoco_sim/output/sorting_roll_expert/v13_diverse_admission_20260825_4gpu/groups/dynamics_heavy_low_friction/seed_10084/sorting_roll_robot_multiview.mp4`（高风险 boundary 三路审核视频：1280×720、30 FPS、58.47 秒）
- `cruzr_mujoco_sim/output/sorting_roll_expert/v13_diverse_admission_20260825_4gpu/groups/dynamics_heavy_low_friction/seed_10084/pilot_validator_report.json`（validator 1/1 通过）
- `cruzr_mujoco_sim/output/sorting_roll_expert/v13_diverse_admission_20260825_4gpu/groups/dynamics_heavy_low_friction/seed_10084/camera_observability.json`（54/54 覆盖，所需相机角色 54/54，平均 2.81 路可用视角）
- `cruzr_mujoco_sim/output/sorting_roll_expert/v13_diverse_admission_20260825_4gpu/admission_report.json`（5×20 准入 98/100 成功，五组 validator 与相机门全部通过）
- `cruzr_mujoco_sim/output/sorting_roll_expert/sorting_roll_v13_diverse300_20260825_4gpu/initial_collection_report.json`（正式首轮 292/300 成功，成功率 97.33%）
- `cruzr_mujoco_sim/output/sorting_roll_expert/sorting_roll_v13_diverse300_20260825_4gpu/replacement_plan.json`（8 个失败源的配额保持替补计划）
- `cruzr_mujoco_sim/output/sorting_roll_expert/sorting_roll_v13_diverse300_20260825_4gpu/selection_report.json`（最终 300 个唯一成功源，所有分层配额精确满足）
- `cruzr_mujoco_sim/output/sorting_roll_expert/sorting_roll_v13_diverse300_20260825_4gpu/validation_report.json`（源数据 300/300 validator 通过）
- `cruzr_mujoco_sim/out/datasets/sorting_roll_v13_diverse300_lerobot_v21_20260825/`（v2.1 staging：300 episodes、519,776 frames、900 个源视频流）
- `cruzr_mujoco_sim/out/datasets/sorting_roll_v13_diverse300_lerobot_v30_20260825/`（LeRobot v3.0 正式训练数据）
- `log/sorting_roll_v13_diverse300_20260825_4gpu/dataset_v30_audit.json`（格式、类型、视频和真实解码审计，`passed=true`）
- `cruzr_mujoco_sim/out/training/pi05_sorting_roll_v13_canary_20260825/`（π0.5 fresh step 20 与 resume step 40 完整 checkpoint）
- `log/sorting_roll_v13_diverse300_20260825_4gpu/pi05_canary_audit.json`（4×4090 短训练及恢复审计，`passed=true`）
- `cruzr_mujoco_sim/output/sorting_roll_expert/sorting_roll_v13_diverse300_20260825_4gpu/formal_training_readiness.json`（全部前置检查通过，`ready_for_formal_training=true`）
- `cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_train.sh`（固定到上述数据与审计结果的正式训练入口）
- `cruzr_mujoco_sim/scripts/training/pi05_sorting_roll_v13_h100x2_train.sh`（2×H100 专用 200→250 canary、20k dry-run/start/resume/status 入口）
- `cruzr_mujoco_sim/out/training/pi05_sorting_roll_v13_h100x2_canary_seed1000/checkpoints/000250/`（2×H100 expert-only fresh/resume 最终完整 checkpoint）
- `log/pi05_sorting_roll_v13_h100x2_canary_seed1000.log`（2×H100 canary 完整日志，最终退出码 0）
- `cruzr_mujoco_sim/output/sorting_roll_expert/v12_diverse_admission_20260825_4gpu/`（拒绝证据：物理 97/100，但首个重放审计仅 90.2% 总覆盖、70.6% 所需角色覆盖，无准入报告）

v10 历史证据（不可混入 v13）：

- `cruzr_mujoco_sim/output/sorting_roll_expert/v10_diverse_admission_20260824/`（5×20 准入已通过：98/100 成功，validator 与每组 3 回合相机审计通过）
- `cruzr_mujoco_sim/output/sorting_roll_expert/sorting_roll_v10_diverse300_20260824_4gpu/`（正式 300 回合四卡采集完成：初采 295/300，通过 5 个定向补采恢复为 300/300；train/val/test 为 240/30/30）
- `cruzr_mujoco_sim/out/datasets/sorting_roll_v10_diverse300_lerobot_v21_20260824/`（v2.1 staging：300 episodes、519866 frames、900 个源视频流）
- `cruzr_mujoco_sim/out/datasets/sorting_roll_v10_diverse300_lerobot_v30_20260824/`（LeRobot v3.0：三路 224×224 RGB，state/action 各 18 维）
- `cruzr_mujoco_sim/out/training/pi05_sorting_roll_v10_canary_20260824/`（π0.5 20-step fresh 与 20→40-step resume checkpoint）
- `log/sorting_roll_v10_diverse300_lerobot_v30_20260824_audit.json`（v3.0 数据与视频抽样审计，`passed=true`）
- `log/pi05_sorting_roll_v10_canary_20260824_audit.json`（短训练、checkpoint 和恢复审计，`passed=true`）
- `cruzr_mujoco_sim/output/sorting_roll_expert/v9_d405_20seed_final_20260823/`
- `cruzr_mujoco_sim/output/sorting_roll_expert/v9_d405_review_seed0120/`
- `cruzr_mujoco_sim/output/sorting_roll_expert/v9_d405_canary30_final_seed0200_0229/`
- `cruzr_mujoco_sim/output/sorting_roll_expert/v9_d405_canary30_replacements/`
- `cruzr_mujoco_sim/out/datasets/sorting_roll_d405_canary30_lerobot_v30_20260823/`

2026-08-24 已删除 22 组定版前调参/诊断产物，约释放 32 MiB。最终 30 回合原始源约 5.47 GiB，因承担验证和数据重建的可追溯性而保留。

材质修复前生成的 v10 冒烟已移入 `rejected_pre_texture_fix/`，不会被 validator 或构建器扫描；它只作为问题留证，不可用于训练。
