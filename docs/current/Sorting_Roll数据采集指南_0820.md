# Sorting Roll 数据采集指南

> 最后更新：2026-08-20
>
> 适用范围：`Sorting_Roll` MuJoCo 场景、CRUZR S2 人工遥操作，以及后续正式数据管线建设。

## 先说结论

仓库当前可以生成并检查 Sorting Roll 场景、进行人工遥操作，并以 30 FPS 录制五路相机、机器人状态、动作和底盘数据。但“录制”和 Sorting Roll 成功判断尚未接通，因此当前入口只适合采集 **pilot 原始回合**，不能直接生成正式训练数据：

- `cruzr_teleop.py` 会把 `meta.json.task` 固定写成 `transport_carton_cruzr`；
- 按 Enter 会直接写入 `success=true`，不会调用 `sorting_roll_task.py`；
- 录制文件没有保存棒子的位姿、速度和接触证据，事后无法严格复算任务成功；
- `scripts/collection/` 下现有验收、批采和 LeRobot 构建脚本属于 `shelf_e2e` 双物料任务，不能直接用于 Sorting Roll；
- 棒子质量仍使用临时值 `0.25 kg`，正式动力学采集前应替换为实测值。

建议先按本文完成少量人工 pilot，验证可达性、操作流程和相机视野；再补齐“正式采集前必须完成的开发”后启动批量采集。若目标是在 48 小时内获得仿真 canary 数据，采用“单卡开发与验收、四卡短时批采、单卡补采与收尾”的成本敏感方案；不要从开发阶段就租用 4 卡或 8 卡。

## 1. 当前任务契约

任务流程是：机器人从自然下垂姿态出发，移动到右侧桌子，抓起横放的棒子，移动到正前方架子，将棒子沿世界坐标 `Y` 方向横置于顶层底槽，完全释放并等待其稳定。

代码中的成功判据为：

| 检查项 | 当前阈值 |
|---|---:|
| 棒子中心相对目标中心的 `X/Y/Z` 误差 | `≤ 3 / 55 / 12 mm` |
| 棒轴与目标轴夹角 | `≤ 10°`，正反向均接受 |
| 棒子必须完整位于架子内宽 | 是 |
| 槽底支撑力 | `≥ 0.5 N` |
| 双夹爪接触力 | `≤ 0.2 N` |
| 桌面支撑力 | `< 0.5 N` |
| 线速度 | `≤ 0.02 m/s` |
| 角速度 | `≤ 0.10 rad/s` |
| 所有条件连续成立 | `≥ 0.5 s` |

权威实现位于：

- 场景：`Sorting_Roll/sorting_roll_scene.xml`
- 场景生成与检查：`cruzr_mujoco_sim/scripts/core/sorting_roll_scene.py`
- 成功判定：`cruzr_mujoco_sim/scripts/core/sorting_roll_task.py`
- 遥操作与原始录制：`cruzr_mujoco_sim/scripts/core/cruzr_teleop.py`

## 2. 采集前检查

所有命令都从仓库根目录执行。

```bash
# 生成场景，并检查布局、初始失败状态和目标槽成功正例
bash Sorting_Roll/run_scene.sh check

# 生成初始布局和目标位置预览
bash Sorting_Roll/run_scene.sh preview

# 可选：核对原始资产和当前低模资产
sha256sum -c Sorting_Roll/ASSET_BACKUP_MANIFEST.sha256
sha256sum -c Sorting_Roll/LOWPOLY_ASSET_MANIFEST.sha256
```

`check` 必须以退出码 0 结束，并同时满足：场景运行检查全部通过、桌面初始状态不是成功、目标槽正例连续稳定 0.5 秒后成功。

默认解释器是 `envs/mjx/bin/python`。换机器时可指定带 MuJoCo 3.9 的 Python：

```bash
RL_MJX_PY=/path/to/python bash Sorting_Roll/run_scene.sh check
```

## 3. 立即可做：采集一个 pilot 原始回合

以下示例会写入 `outputs/teleop/sorting_roll_pilot_000001/`。每个回合必须使用新名称；当前录制器不会拒绝已有目录，重复名称可能混入或覆盖旧数据。

```bash
EPISODE_NAME=sorting_roll_pilot_000001
test ! -e "outputs/teleop/$EPISODE_NAME"

CRUZR_EP_SEED=1 \
REC_PROMPT="Pick up the roll from the table and place it stably in the top shelf slot" \
REC_CAMS=stereo_left,stereo_right,waist_front,hand_left,hand_right \
REC_SAVE_RAW_TIMESTAMPS=1 \
TELEOP_RECORD_GPU=0 \
TELEOP_RECORD="$EPISODE_NAME" \
bash Sorting_Roll/run_scene.sh view
```

说明：

- 默认查看器使用 EGL GPU、`1280×720 / 60 FPS`；训练相机按 `640×480 / 30 FPS` 单独录制。
- 键盘焦点必须放在启动命令所在的终端；图形窗口只负责显示和鼠标调视角。
- 不要设置 `TELEOP_HOME=pregrasp` 或 `TELEOP_HOME=sidegrasp`。它们属于旧 `jig` 任务；Sorting Roll 应使用默认 `droop`。
- `CRUZR_EP_SEED` 当前只写入元数据，还不会随机化 Sorting Roll 场景。
- `REC_SAVE_RAW_TIMESTAMPS=1` 会额外保存同步采集时间戳，建议 pilot 也开启。

### 单回合操作顺序

1. 场景打开后，在终端按 `z`，确认出现 `[REC] ● RECORDING ON`，再开始移动机器人。
2. 用底盘移动到右侧桌子并停车；接近物体前保持低速。
3. 调整双臂并夹住棒子，抬起后确认棒子未滑落。
4. 将底盘移动到正前方架子，停车后再执行精细放置。
5. 将棒子沿架子宽度方向放入顶层底槽，打开两侧夹爪并退出接触。
6. 肉眼确认棒子由槽底支撑、不再接触桌面和夹爪，至少等待 0.5 秒。
7. pilot 看起来成功时按 Enter 保存并退出；明显失败时按 Backspace 丢弃整个回合。

第 6 步目前只是人工检查。按 Enter 写出的 `success=true` 不是 `sorting_roll_task.py` 的物理判定结果，所以该回合仍只能标记为 pilot。

不要用 ESC、图形窗口的关闭按钮或终止进程结束有效回合；这些路径不会调用 `finalize()`，可能只留下不完整的 JPG 目录。Backspace 会递归删除当前回合目录，删除后不能从录制器恢复。

## 4. 常用遥操作按键

| 功能 | 按键 |
|---|---|
| 开始/继续录制 | `z` |
| 暂停录制，保留已录帧 | `x` |
| 保存并退出 | Enter |
| 删除本回合并退出 | Backspace |
| 切换左右手 | Tab |
| 双臂同时控制开关 | `m` |
| 关节模式/末端平移模式切换 | `v` |
| 夹爪开合切换 | Space |
| 夹爪全开/全合 | `[` / `]` |
| 前进/后退 | `8` / `2` |
| 左转/右转 | `7` / `9` |
| 底盘停车并保持当前位置 | `0` |
| 将关节目标同步到当前实测状态 | `=` |

关节模式下，`qwertyu` 分别增加激活手的 7 个关节，`asdfghj` 分别减小对应关节。末端模式下，`q/a`、`w/s`、`e/d` 分别沿世界坐标 `X/Y/Z` 正反方向移动末端，每次默认 `0.01 m`；`t/g`、`y/h`、`u/j` 仍控制最后三个关节。

底盘速度命令会持续生效。每次转向或前后移动结束后都应按 `0` 停车，不要用反向键猜测抵消速度。

## 5. 原始回合输出

正常保存后目录应类似：

```text
outputs/teleop/sorting_roll_pilot_000001/
├── frames/
│   ├── stereo_left/frame_000000.jpg
│   ├── stereo_right/frame_000000.jpg
│   ├── waist_front/frame_000000.jpg
│   ├── hand_left/frame_000000.jpg
│   └── hand_right/frame_000000.jpg
├── episode_data.npz
├── sdk_timestamps.npz
└── meta.json
```

`episode_data.npz` 当前包含：

- `timestamp`：均匀的 30 FPS 时间轴；
- `state` / `action`：各 16 维，左右臂各 7 个关节加左右夹爪开合比例；
- `action_real`：动作是否为真实控制命令；
- `base`：底盘实测 `x/y/yaw`；
- `base_velocity`：底盘实测前向速度和角速度；
- `base_action`：底盘命令前向速度和角速度；
- `phase`：当前通用遥操作阶段标签。

## 6. 每个 pilot 回合的检查

先检查元数据、数组和五路相机帧数：

```bash
EPISODE_DIR=outputs/teleop/sorting_roll_pilot_000001
envs/mjx/bin/python - "$EPISODE_DIR" <<'PY'
import json
import pathlib
import sys
import numpy as np

episode = pathlib.Path(sys.argv[1])
meta = json.loads((episode / "meta.json").read_text())
with np.load(episode / "episode_data.npz", allow_pickle=False) as data:
    frames = int(meta["num_frames"])
    counts = {
        camera: len(list((episode / "frames" / camera).glob("frame_*.jpg")))
        for camera in meta["cameras"]
    }
    print("task/seed/frames:", meta["task"], meta["seed"], frames)
    print("camera_counts:", counts)
    print("state/action:", data["state"].shape, data["action"].shape)
    assert frames > 0 and set(counts.values()) == {frames}
    assert data["state"].shape == data["action"].shape == (frames, 16)
    for key in ("timestamp", "state", "action", "base", "base_velocity", "base_action"):
        assert np.isfinite(data[key]).all(), key
PY
```

看到 `task: transport_carton_cruzr` 是当前已知限制。不要手改成 Sorting Roll 后就把它当作正式数据；物理成功证据仍然缺失。

可复用任务无关的视频工具生成每路相机和五路拼接预览：

```bash
envs/mjx/bin/python \
  cruzr_mujoco_sim/scripts/collection/shelf_e2e_make_videos.py \
  outputs/teleop/sorting_roll_pilot_000001
```

这里只复用 JPG 转 MP4 的通用能力。脚本名称虽然带 `shelf_e2e`，但它只读取相机清单；不要据此推断其他 `shelf_e2e_*` 脚本也兼容 Sorting Roll。

视频至少检查：

- 五路相机帧数一致、无黑帧、卡帧、明显曝光或遮挡问题；
- 找棒、抓取、运输、槽位对准和释放阶段在训练相机中可观察；
- 没有明显穿模、剧烈碰撞、关节抖动或长时间无意义停顿；
- 动作从统一初始状态开始，结束时包含释放后的稳定观察窗口。

## 7. pilot 数据管理规则

- pilot 与正式数据使用不同目录，建议命名为 `sorting_roll_pilot_<六位编号>`。
- 每个编号只使用一次；启动前用 `test ! -e` 检查目标目录不存在。
- 保存失败回合用于分析时，应移入独立 `failed/` 或 `debug/` 区域并写明原因，不能混进成功训练源。
- 不要直接修改 `meta.json.success` 或 `meta.json.task` 来“修复”标签；应由正式采集程序根据物理证据生成。
- 数据、视频和输出目录受 `.gitignore` 管理，不要强制加入 Git。

建议先录少量 pilot，覆盖成功、抓取失败、运输滑落、槽位错位和未完全释放等情况。这里不规定正式条数；正式规模应在成功门、随机化和验收器完成后，根据首批通过率与训练需求确定。

## 8. 正式采集前必须完成的开发

1. 新增 Sorting Roll 专用采集入口，避免依赖旧 `jig` 任务的对象名称、初始姿态和元数据。
2. 将 `evaluate_placement()` 和 `SortingRollSuccessTracker` 接入运行循环；只有物理条件连续满足 0.5 秒才允许写 `success=true`。
3. 每帧或至少终局保存棒子位姿、线/角速度、槽底/桌面/夹爪接触力及各项成功检查。
4. 把任务名、任务版本、采集配置、场景/资产哈希、seed、prompt 和相机契约写入 `meta.json`。
5. 实现按 seed 重置与随机化，至少覆盖棒子初始位姿、机器人初始位姿和合理的视觉/动力学变化；train/val/test 按源 seed 隔离。
6. 新增 Sorting Roll source validator，拒绝空回合、错误任务版本、错误相机、帧数不一致、非有限数组、失败标签和缺少终局稳定窗的回合。
7. 新增或参数化 LeRobot 构建器，明确 16D 关节状态与底盘字段如何组成训练 schema，并通过小数据 canary。
8. 用实测棒子质量替换 `0.25 kg`，重新运行场景、抓取和成功判定测试。
9. 冻结正式相机组合前，确认仿真相机与计划使用的真机相机一致；不能只因当前录了五路就默认全部进入策略输入。

正式准入顺序建议为：

```text
场景/资产检查
  → 唯一 seed 和输出目录检查
  → 采集完整回合
  → 物理成功门
  → 数组、时间戳和相机一致性检查
  → 视频抽检
  → 按源 seed 划分 train/val/test
  → 构建 LeRobot 数据集
  → 小规模训练 canary
  → 扩大正式采集
```

## 9. 两天内的成本敏感方案

本节使用以下估算参数：目标为 300 个成功仿真回合，自动专家单次尝试平均 4 分钟，成功率 80%，GPU 按卡时计费。参数只用于规划，正式扩卡前必须用本任务实测值替换。

两天目标指“可通过自动成功门、可构建 LeRobot 并用于短训练 canary 的仿真数据”，不等同于已经完成实物质量标定、完整随机化和 sim-to-real 验证的最终数据。

| 时间 | GPU | 工作及通过条件 |
|---|---:|---|
| 第 1 天，0–10 小时 | 1×4090 | 实现自动专家、reset、录制、Sorting Roll 成功门和正确元数据 |
| 第 1 天，10–14 小时 | 1×4090 | 跑 20 个独立 seed；至少 18 个通过完整物理成功门 |
| 第 1 天，14–20 小时 | 1×4090 | 采集 30–50 个 canary，完成 validator、视频抽检和 LeRobot 加载测试 |
| 第 2 天，0–1 小时 | 4×4090 | 短时并行基准，确认无 seed/目录冲突，吞吐至少达到单卡的 3.2 倍 |
| 第 2 天，1–9 小时 | 4×4090 | 主批次按唯一 seed 分片；达到目标成功数后立即停止并释放额外 GPU |
| 第 2 天，后续 | 1–2×4090 | 只补失败 seed 和尾部缺口；validator、MP4、LeRobot 构建主要使用 CPU/单卡 |

如果自动专家到第 1 天第 14 小时仍未达到 `18/20`，不要扩卡。并行只会更快地产生失败数据，应继续用单卡修复专家，或将两天交付降级为少量人工轨迹加确定性重放的管线 canary；重放变体不能统计为独立专家源。

## 10. GPU 时间与费用估算

定义：

- `N`：目标成功回合数；
- `t`：单次尝试平均分钟数；
- `p`：专家成功率；
- `W`：并行 GPU/worker 数；
- `η`：并行效率，包含 CPU、JPEG 和磁盘竞争。

```text
墙钟小时 = N × t ÷ (60 × p × W × η)
总 GPU 小时 = W × 墙钟小时
预计费用 = 总 GPU 小时 × 每张 GPU 的小时单价
```

以 `N=300`、`t=4 分钟`、`p=80%` 为例：

| 配置 | 并行效率假设 | 预计墙钟时间 | 预计 GPU 小时 | 相对判断 |
|---|---:|---:|---:|---|
| 1×4090 | 100% | 约 25 小时 | 约 25 GPUh | 最省，但批采时间最长 |
| 4×4090 | 90% | 约 6.9 小时 | 约 27.8 GPUh | 多约 11% 卡时，节省约 18 小时 |
| 8×4090 | 75% | 约 4.2 小时 | 约 33.3 GPUh | 比四卡只快约 2.7 小时，卡时再增加约 20% |

如果各卡小时单价相同，增加 GPU 主要购买的是更短的墙钟时间，而不是更少的总费用。对于首批 300 个回合，四卡通常是成本与时间的平衡点，八卡默认不启用。

实际结算还要考虑云平台整机价格、最小计费粒度和多卡折扣。扩卡前应先用单卡实测 `t` 和 `p`，再代入公式，不以表中的 4 分钟和 80% 作为结算依据。

## 11. 何时使用 1、4 或 8 张 4090

| 阶段 | 建议配置 | 原因 |
|---|---:|---|
| 编写和调试自动专家 | 1 张 | 代码、物理和成功门调试不能有效并行 |
| 单回合相机/录制 smoke | 1 张 | 先消除错误标签、视角和输出格式问题 |
| 20-seed readiness | 1 张 | 在扩大费用前获得真实成功率 |
| 30–50 回合 canary | 1 张 | 验证长时间稳定性、磁盘和数据构建 |
| 四卡扩展基准 | 4 张，10–60 分钟 | 只测吞吐、资源冲突和费用模型 |
| 主要批量采集 | 4 张 | 300 回合规模的默认性价比配置 |
| 失败补采和尾部缺口 | 1–2 张 | 防止多卡等待少量任务 |
| 验收、MP4、LeRobot 构建 | 0–1 张 | 通常受 CPU、FFmpeg 和磁盘限制 |
| 8 卡批采 | 条件启用 | 仅在更大规模或硬截止时间下使用 |

四卡必须同时满足以下条件才可开启：

- 20 个独立 seed 至少成功 18 个；
- 连续运行 1–2 小时没有显存泄漏、空回合或非有限状态；
- 每个 worker 使用不重叠的 seed、独立输出目录和正确 GPU ID；
- validator 能自动拒绝失败、缺帧、错相机和错误任务版本；
- 磁盘空间足以容纳目标批次，持续写入不会拖慢仿真；
- 四卡短测吞吐至少达到单卡的 3.2 倍。

只有满足下列任一条件，且八卡实测并行效率仍不低于约 80%，才考虑 8 卡：

- 目标扩大到 600–1000 个以上成功回合；
- 单回合实测超过 6–8 分钟；
- 四卡按实测速度无法在硬截止时间内完成；
- 需要同时采集多个相互独立的随机化 profile；
- 云平台的八卡整机存在足以抵消并行损失的价格折扣。

### 降低费用的执行规则

- pilot 阶段保留五路相机用于确定视野；冻结策略输入后，正式批采只录实际需要的三路相机，可比五路减少 40% 图像渲染和写盘。
- 抓取失败、棒子掉落或发生不可恢复碰撞时提前终止，不跑完整超时。
- 目标 300 个成功回合时，先把最大尝试数限制在 400–450；达到成功数立即全局停止。
- 采集过程中不为每个回合生成 MP4，只审查全部失败回合和约 10% 成功回合。
- 四卡按成功目标分片，而不是按固定运行时间；例如 300 个目标可先分配每卡 75 个，并由协调器处理尾部差额。
- 额外 GPU 只在四卡基准和主批次期间租用；数据验收、视频编码、数据集构建和失败分析开始前立即释放。
- 如果四卡吞吐不足单卡的 3.2 倍，先处理 CPU、JPEG 或磁盘瓶颈，不扩到八卡。

## 12. 当前不要执行的操作

- 不要直接运行 `shelf_e2e_batch.sh` 或 `shelf_e2e_multigpu_collect.py` 采集 Sorting Roll；它们使用双物料专家和双物料成功门。
- 不要用 `shelf_e2e_source.py` 或 `shelf_e2e_build_v2.py` 验收/构建 Sorting Roll；任务版本、阶段和质量契约不匹配。
- 不要从 `scripts/archive/` 选择旧转换脚本作为新正式流程，除非先逐字段验证并显式改造成 Sorting Roll 专用实现。
- 不要把人工看到“差不多放进去了”当作成功标签；正式标签必须来自 `sorting_roll_task.py` 的完整稳定判据。

## 13. 常见问题

### `view` 提示找不到 Python

```bash
RL_MJX_PY=/path/to/mujoco/python bash Sorting_Roll/run_scene.sh view
```

### EGL 窗口启动失败

可先指定 GPU：

```bash
TELEOP_EGL_GPU=0 TELEOP_RECORD_GPU=0 bash Sorting_Roll/run_scene.sh view
```

仅做兼容性查看时可回退 CPU viewer；正式五路相机录制仍需要 EGL：

```bash
TELEOP_VIEWER=passive bash Sorting_Roll/run_scene.sh view
```

### 目录中只有 JPG，没有 `meta.json` 或 `episode_data.npz`

该回合没有正常 finalize，通常由关闭窗口、按 ESC、异常退出或强制终止造成。把它视为不完整回合，不要进入数据集；使用新编号重新采集。

### 按键没有反应

把焦点切回启动命令所在的终端。每次有效按键都应看到 `[KEY]` 回显。

## 14. 当前推荐决策

先用当前单张 4090 完成场景检查、少量人工 pilot、自动专家开发、20-seed readiness 和 30–50 回合 canary。只有成功率、validator、磁盘和四卡吞吐门全部通过，才在第二天短时租用另外三张 4090 进行主批次；达到目标后立即释放，失败补采和收尾恢复单卡。首批 300 回合不使用八卡，除非四卡实测无法满足硬截止时间且八卡并行效率通过基准。
