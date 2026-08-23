# Sorting Roll 正式数据采集与 π0.5 训练指南

> 状态：2026-08-24 定版。后续 Sorting Roll 仿真采集、数据验收和 π0.5 训练以本文档为准。历史推导与调试过程见 `docs/current/Sorting_Roll数据采集指南_0820.md`。

当前基线和数据管线已经通过验证，但正式多样化采集尚未启动。下一步不是直接扩采 v9，而是先实现并验收第 3 节定义的 `sorting_roll_v10_diverse_sim` manifest、元数据和 validator；完成后再执行第 5 节开始的采集流程。

## 1. 已定版的基线

- 任务版本：`sorting_roll_v9_d405_sim`；已批准的动作、相机位姿、槽位几何和成功判定不得在同一批数据中静默修改。
- 策略输入固定为三路 RGB：
  - `observation.images.stereo_left`
  - `observation.images.left_wrist_realsense`
  - `observation.images.right_wrist_realsense`
- 第三视角和槽位近景只用于人工审核，不进入 π0.5。
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

第一阶段采集 **300 个通过 validator 的多样化成功回合**：

- train：240 个；seed 末位不是 `0` 或 `1`。
- val：30 个；seed 末位为 `1`。
- test：30 个；seed 末位为 `0`。
- 建议正式 seed 使用 `1000–1299`；补采从 `1300` 开始，不能复用失败 seed。
- 初始尝试成功率至少 `90%`；最终构建必须恰好选出 300 个成功且互不重复的源回合。

其中 240 个 train 回合进入优化器，val/test 只用于选型和最终评测；把全部 300 个回合都用于梯度更新会造成评测泄漏。现有 30 回合 v9 canary 继续作为基线证据，不自动混入 v10 正式数据。300 回合用于验证数据、训练和闭环评测管线，不代表数据量一定足够。只有测试集闭环成功率和分层成功率达标后，才扩大到 1,000 回合以上。

## 3. 多样性要求

### 3.1 当前已经支持的多样性

当前 v9 只实现了以下确定性 seed 随机化：

| 维度 | 当前范围 |
| --- | --- |
| 机器人基座 X/Y | 各 `±15 mm` |
| 机器人基座 yaw | `±0.025 rad`，约 `±1.43°` |
| 棒子 X/Y | 各 `±4 mm` |
| 棒子 yaw | `±0.012 rad`，约 `±0.69°` |

这些变化足以做基线 canary，但不足以满足“多种类数据”的正式要求。因此，**不得直接把当前 v9 无修改扩采 300 回合并称为多样化正式数据**。

### 3.2 正式采集前必须增加的分层配置

新增多样性必须作为新的版本化 profile 实现，建议名称为 `sorting_roll_v10_diverse_sim`。它只改变任务相关分布，不改变已定版的相机外参、任务语义、成功标准和专家动作结构。

300 个回合应由 campaign manifest 预先分配，而不是事后查看随机结果。以下每一行都是覆盖约束，各维度可以交叉组合：

| 维度 | 300 回合最低覆盖 | 约束 |
| --- | --- | --- |
| 初始位姿难度 | easy 120 / medium 120 / boundary 60 | 均在已验证可达范围内；boundary 不能越过安全边界 |
| 棒子几何类型 | 3 类，各 100 | 候选长度为架子可用内宽的约 `82% / 85% / 88%`；精确直径、重量必须记录 |
| 棒子外观 | 至少 5 类，每类至少 40 | 颜色、明暗和表面纹理需保持棒子可见；不能只靠颜色表达任务语义 |
| 光照 | normal 180 / dim 60 / bright 60 | 相机必须仍能观察抓取、运输、插入和释放阶段 |
| 动力学 | nominal 180 / 两个边界组各 60 | 质量、摩擦候选变化先限制在基准值的 `±15%`，每组单独准入 |
| 图像扰动 | clean 180 / mild 60 / strong 60 | 只允许真实可能出现的亮度、噪声和压缩变化；相机位姿不得抖动 |
| 任务指令 | 至少 5 个同义英文指令，每个至少 40 | 语义必须完全等价，不能引入新任务 |

候选几何值必须先通过 `Sorting_Roll/run_scene.sh check`。如果未来真机物体范围已知，应以实测范围替代上述候选值；不要为了“多样性”制造现实中不存在的样本。

### 3.3 多样性准入门

每一种新增棒子几何或动力学 profile 在进入 300 回合前，都必须独立完成：

1. 20 个新 seed 的带渲染测试。
2. 至少 `18/20` 成功。
3. 三路策略相机通过可观测性审核。
4. 至少人工抽检 3 个完整视频，其中必须包含一个 boundary 回合。
5. validator 全部通过，且动作仍然自然、平稳、低于 60 秒。

任何 profile 未通过时，只剔除该 profile，不得降低物理成功标准来保留它。

## 4. 4 卡与 8 卡的使用原则

- 300 回合优先使用 **4×4090**：每卡一个 EGL 渲染 shard，成本和吞吐更均衡。
- 8×4090 只在以下情况使用：目标扩大到 1,000 回合以上、4 卡预估无法满足截止时间，或已经确认 CPU、文件系统和 JPEG 写入不是瓶颈。
- 单张 4090 同时只运行一个渲染 shard。增加同卡并发通常会放大显存、EGL 和 I/O 风险。
- 多卡只缩短采集墙钟时间，不改变 seed、数据分布、成功标准或格式。
- 当前机器只检测到 GPU 0；以下多卡命令应在真正具备 4/8 张卡的机器上执行。

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

以下模板适用于 4 卡或 8 卡。修改 `GPU_COUNT` 即可；300 个 seed 会无重叠地均分到各卡。正式运行前，采集器必须已经接入并记录第 3 节的 campaign manifest。

```bash
ROOT=/share/home/tm1128689517650000/a852937540/cruzr_sim
PY=$ROOT/envs/mjx/bin/python
GPU_COUNT=4
TOTAL=300
FIRST_SEED=1000
CAMPAIGN=sorting_roll_v10_diverse300_$(date -u +%Y%m%d)
RAW_ROOT=$ROOT/cruzr_mujoco_sim/output/sorting_roll_expert/$CAMPAIGN
LOG_ROOT=$ROOT/log/$CAMPAIGN

mkdir -p "$RAW_ROOT" "$LOG_ROOT"
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

不足 300 个时，从 seed 1300 开始使用独立的 `replacement_*` shard 补采。补采也必须满足 campaign manifest 的缺失配额；不能只补最容易成功的类别。最终应生成恰好 300 行的新 `selected_sources.txt`。

## 8. 源数据 validator

```bash
mapfile -t SOURCES < "$SELECTED"
REPORT=$RAW_ROOT/validation_report.json
"$PY" cruzr_mujoco_sim/scripts/collection/sorting_roll_validate.py \
  "${SOURCES[@]}" --report "$REPORT"
```

必须看到：

```text
episode_count=300
passed_count=300
failed_count=0
passed=true
```

除现有 validator 外，正式 v10 validator 还必须核对 manifest 与 episode 元数据完全一致，并输出每个 diversity stratum 在 train/val/test 中的数量。只要有重复 seed、缺失类别或分层偏斜，就停止构建。

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
DATASET=$ROOT/cruzr_mujoco_sim/out/datasets/sorting_roll_v10_diverse300_lerobot_v30
"$PY" cruzr_mujoco_sim/scripts/collection/sorting_roll_build_v21.py \
  "${SOURCES[@]}" --out "$DATASET" --encode-workers 4
```

转换为 LeRobot v3.0：

```bash
ISAAC_PY=/isaac-sim/python.sh
REPO_ID=local/sorting_roll_v10_diverse300
PYTHONPATH=. "$ISAAC_PY" \
  src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
  --repo-id "$REPO_ID" --root "$DATASET" --push-to-hub false
```

转换后必须核对 `meta/info.json` 和 `meta/stats.json`，并用真实 `LeRobotDataset` 解码首帧、尾帧及每个 diversity stratum 的随机样本。v3 审计通过前不要删除 v2.1 staging 备份。

## 10. π0.5 训练前 canary

先用 4×4090 做 20-step expert-only DDP canary；4090 上不要直接照搬 H100 的每卡 batch。当前已验证的安全起点是每卡 batch 1、BF16 和 gradient checkpointing。

```bash
bash pi05_train.sh dry-run \
  --dataset-root "$DATASET" --repo-id "$REPO_ID" --episodes train \
  --gpu-ids 0,1,2,3 --num-processes 4 \
  --batch-size 1 --num-workers 2 --allow-small-batch true \
  --steps 20 --warmup-steps 2 --decay-steps 20 \
  --save-freq 20 --train-expert-only true \
  --output-dir cruzr_mujoco_sim/out/training/pi05_sorting_roll_v10_canary \
  --log-file log/pi05_sorting_roll_v10_canary.log
```

`dry-run` 通过后把动作改为 `start`。canary 必须满足：

- 20 步正常退出，无 OOM、NaN、Inf 或视频解码错误。
- loss 和梯度范数全部有限。
- checkpoint 完整可读取，并能从该 checkpoint 恢复到至少第 40 步。
- 训练只使用 train split，val/test 不得进入优化器。

只有 4 卡 canary 稳定后才考虑 8 卡。8 卡需要重新做 DDP canary；不得假定卡数翻倍就一定更省钱或更快。

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

- `cruzr_mujoco_sim/output/sorting_roll_expert/v9_d405_20seed_final_20260823/`
- `cruzr_mujoco_sim/output/sorting_roll_expert/v9_d405_review_seed0120/`
- `cruzr_mujoco_sim/output/sorting_roll_expert/v9_d405_canary30_final_seed0200_0229/`
- `cruzr_mujoco_sim/output/sorting_roll_expert/v9_d405_canary30_replacements/`
- `cruzr_mujoco_sim/out/datasets/sorting_roll_d405_canary30_lerobot_v30_20260823/`

2026-08-24 已删除 22 组定版前调参/诊断产物，约释放 32 MiB。最终 30 回合原始源约 5.47 GiB，因承担验证和数据重建的可追溯性而保留。
