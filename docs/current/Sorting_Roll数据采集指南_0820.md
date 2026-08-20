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

建议先按本文完成少量人工 pilot，验证可达性、操作流程和相机视野；再补齐“正式采集前必须完成的开发”后启动批量采集。

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

## 9. 当前不要执行的操作

- 不要直接运行 `shelf_e2e_batch.sh` 或 `shelf_e2e_multigpu_collect.py` 采集 Sorting Roll；它们使用双物料专家和双物料成功门。
- 不要用 `shelf_e2e_source.py` 或 `shelf_e2e_build_v2.py` 验收/构建 Sorting Roll；任务版本、阶段和质量契约不匹配。
- 不要从 `scripts/archive/` 选择旧转换脚本作为新正式流程，除非先逐字段验证并显式改造成 Sorting Roll 专用实现。
- 不要把人工看到“差不多放进去了”当作成功标签；正式标签必须来自 `sorting_roll_task.py` 的完整稳定判据。

## 10. 常见问题

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

## 11. 当前推荐决策

现在先完成场景检查并人工录制少量 pilot，用它们回答三个问题：机器人能否稳定完成双臂抓取与放置、五路相机是否覆盖所有关键阶段、操作过程中最常见的失败模式是什么。pilot 通过后，优先实现“成功判定接入录制器 + Sorting Roll 元数据/验收器”，再确定批量规模和正式 LeRobot 构建方案。
