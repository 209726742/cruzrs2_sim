# `scripts/` 目录导航

> 更新日期：2026-08-16
> 当前主任务：CRUZR S2 双物料线（立柱放中层、条料放顶层）  
> 整理结果：原顶层 71 个执行/测试文件已完成可逆物理分类；现新增 4 个柔性参数/场景工具和 4 个测试文件，共 79 个，没有永久删除源码。

## 先看这里

- 当前正式采集代码只在 `collection/` 与 `core/`；常用入口从 `collection/` 运行。
- `diagnostics/` 只做 NOREC 回归和几何探针，不发布训练数据。
- `tests/` 是当前链的 21 个测试文件，现有测试总数为 100。
- `archive/` 保存旧 ECU、RL/DAgger 和历史专家，不得与当前双物料数据混用；归档代码不承诺可原地运行。
- 当前正式覆盖率门为至少 90%（24/26），v24 实测为 16/26；加上仅可见 1 张 GPU，仍禁止正式多卡采集。单条 episode 的任务、SDK、motion、terminal-hold 和源验收硬门没有放宽；柔性 synthetic 参数只用于隔离开发，当前状态见[双物料线训练进度](../../docs/current/双物料线训练进度_0820.md)。

目录结构：

```text
scripts/
├── collection/       # 当前双物料采集、验收、构建、视频和 rollout（10）
├── core/             # 当前共享契约、控制、质量、对象与柔性拟合工具（14）
├── diagnostics/      # 当前 NOREC/几何诊断（3）
├── tests/            # 当前回归测试文件（19，合计 88 项测试）
└── archive/
    ├── ecu/          # 旧 ECU/纸箱流水线（18）
    ├── rl/           # 旧单立柱 RL/DAgger 及专属测试（13）
    └── history/      # 单物料与 .orig.py 历史专家（2）
```

## 当前常用入口

| 目标 | 使用文件 | 说明 |
|---|---|---|
| 4/8 卡计划或受保护启动 | [`collection/shelf_e2e_multigpu_collect.py`](collection/shelf_e2e_multigpu_collect.py) | 默认 plan-only；最终 70% 常规 / 20% 单轴边界 / 10% 受控恢复 |
| 生成 4/8 卡不可变预检报告 | [`collection/shelf_e2e_collection_preflight.py`](collection/shelf_e2e_collection_preflight.py) | 只检查，不启动 worker |
| 单卡/单分片调试 | [`collection/shelf_e2e_batch.sh`](collection/shelf_e2e_batch.sh) | 含锁、resume、源验收、失败保留和原子发布 |
| 少量 seed 无录制回归 | [`diagnostics/shelf_e2e_sweep.py`](diagnostics/shelf_e2e_sweep.py) | 只用于任务行为诊断 |
| 单条 episode 硬验收 | [`collection/shelf_e2e_source.py`](collection/shelf_e2e_source.py) | 统一检查任务、相机、SDK、motion、diversity/layout |
| 构建 LeRobot v2.1 | [`collection/shelf_e2e_build_v2.py`](collection/shelf_e2e_build_v2.py) | 只读取已通过源验收的 episode |
| 生成三路/宫格视频 | [`collection/shelf_e2e_make_videos.py`](collection/shelf_e2e_make_videos.py) | 编码已录制的策略相机帧 |
| 生成第三视角视频 | [`collection/shelf_e2e_replay_3rd.py`](collection/shelf_e2e_replay_3rd.py) | 第三视角只用于诊断，不进入 π0.5 输入 |

命令应从 `cruzr_mujoco_sim/` 根目录运行：

```bash
# 查看多卡入口；不会采集
../envs/mjx/bin/python scripts/collection/shelf_e2e_multigpu_collect.py --help

# 运行当前 88 项测试
../envs/mjx/bin/python -m unittest discover -s scripts/tests -p 'test_*.py' -v
```

## A. 当前双物料正式链（10）

| 文件 | 角色 | 是否建议直接运行 |
|---|---|---|
| [`shelf_e2e_multigpu_collect.py`](collection/shelf_e2e_multigpu_collect.py) | 4/8 GPU 总调度、三层配额、master manifest | 是；默认 plan-only |
| [`shelf_e2e_collection_preflight.py`](collection/shelf_e2e_collection_preflight.py) | GPU/任务/相机/时序/seed/目录/存储预检 | 是；只生成报告 |
| [`shelf_e2e_batch.sh`](collection/shelf_e2e_batch.sh) | 单 GPU 安全分片 worker | 仅调试；多卡由总调度器调用 |
| [`shelf_e2e_dual_expert.py`](collection/shelf_e2e_dual_expert.py) | 双物料专家、任务硬门、布局/恢复多样性 | 仅专家/NOREC 调试 |
| [`shelf_e2e_sdk_capture_smoke.py`](collection/shelf_e2e_sdk_capture_smoke.py) | SDK 三相机与原始时间戳短录制 | 是；输出不可训练 |
| [`shelf_e2e_source.py`](collection/shelf_e2e_source.py) | 录制源硬验收 | 是；只读验证 |
| [`shelf_e2e_build_v2.py`](collection/shelf_e2e_build_v2.py) | 通过源构建 LeRobot v2.1 | 是；必须使用全新 OUT |
| [`shelf_e2e_make_videos.py`](collection/shelf_e2e_make_videos.py) | 策略相机 MP4/宫格 | 是；后处理 |
| [`shelf_e2e_replay_3rd.py`](collection/shelf_e2e_replay_3rd.py) | 第三视角重建 | 是；诊断用 |
| [`shelf_e2e_rollout.py`](collection/shelf_e2e_rollout.py) | π0.5 闭环仿真评测 | 训练后使用 |

## B. 当前共享核心（14）

这些文件主要供正式链导入，不是操作员的第一入口。

| 文件 | 作用 |
|---|---|
| [`action_quality.py`](core/action_quality.py) | action 连续性与 state/action tracking 审计 |
| [`cruzr_s2_sdk_contract.py`](core/cruzr_s2_sdk_contract.py) | SDK 关节、速度、底盘、相机和时间戳契约 |
| [`cruzr_teleop.py`](core/cruzr_teleop.py) | MuJoCo 控制、共享状态与 episode recorder |
| [`shelf_e2e_contract.py`](core/shelf_e2e_contract.py) | 部署 observation/action/camera 契约 |
| [`shelf_e2e_flex_state.py`](core/shelf_e2e_flex_state.py) | 柔性条料内部状态 sidecar/回放契约 |
| [`shelf_e2e_objects.py`](core/shelf_e2e_objects.py) | 刚体/多体对象拓扑工具 |
| [`shelf_e2e_profiles.py`](core/shelf_e2e_profiles.py) | `strict_v1` / `sdk_recovery_v1` 相机配置 |
| [`strip_cable_damping_fit.py`](core/strip_cable_damping_fit.py) | 固定协议的隔离自由衰减与 joint damping 拟合 |
| [`strip_cable_isolated.py`](core/strip_cable_isolated.py) | 拟合参数的独立 cable XML 与零应力稳定性验证 |
| [`strip_cable_scene.py`](core/strip_cable_scene.py) | 独立柔性双物料模板生成、契约与重力落稳验证 |
| [`strip_cable_structure.py`](core/strip_cable_structure.py) | 柔性弧形条料结构编译探针 |
| [`strip_material_fit.py`](core/strip_material_fit.py) | 实物/假设曲线到 MuJoCo cable 参数候选的可审计拟合 |
| [`strip_measurement_check.py`](core/strip_measurement_check.py) | 实物测量完整性和模型选择门 |
| [`teleop_timing.py`](core/teleop_timing.py) | 无累计漂移的控制子步时钟 |

## C. 当前诊断（3）

| 文件 | 用途 |
|---|---|
| [`shelf_e2e_sweep.py`](diagnostics/shelf_e2e_sweep.py) | 少量 seed NOREC 扫描、报告和差分 |
| [`shelf_e2e_grasp_probe.py`](diagnostics/shelf_e2e_grasp_probe.py) | 抓取帧与夹爪接触定位 |
| [`shelf_e2e_reach_probe.py`](diagnostics/shelf_e2e_reach_probe.py) | 几何可达性和 fully-on-shelf 探针 |

## D. 当前测试（19）

| 文件 | 覆盖对象 |
|---|---|
| [`test_action_quality.py`](tests/test_action_quality.py) | action/tracking 质量门 |
| [`test_cruzr_s2_sdk_contract.py`](tests/test_cruzr_s2_sdk_contract.py) | SDK 契约、限位、相机/时间戳 |
| [`test_shelf_e2e_build_v2.py`](tests/test_shelf_e2e_build_v2.py) | 数据构建、split、源去歧义 |
| [`test_shelf_e2e_collection_preflight.py`](tests/test_shelf_e2e_collection_preflight.py) | 4/8 卡预检分片和 seed |
| [`test_shelf_e2e_contract.py`](tests/test_shelf_e2e_contract.py) | state/action/camera 契约 |
| [`test_shelf_e2e_diversity.py`](tests/test_shelf_e2e_diversity.py) | clean/recovery 与 random/boundary 元数据 |
| [`test_shelf_e2e_flex_state.py`](tests/test_shelf_e2e_flex_state.py) | 柔性内部状态 sidecar/回放 |
| [`test_shelf_e2e_make_videos.py`](tests/test_shelf_e2e_make_videos.py) | 元数据驱动的多相机预览 |
| [`test_shelf_e2e_multigpu_collect.py`](tests/test_shelf_e2e_multigpu_collect.py) | 多卡计划、预检绑定和 seed |
| [`test_shelf_e2e_objects.py`](tests/test_shelf_e2e_objects.py) | 刚体/铰接对象拓扑 |
| [`test_shelf_e2e_profiles.py`](tests/test_shelf_e2e_profiles.py) | strict/SDK profile 隔离 |
| [`test_shelf_e2e_source_sdk.py`](tests/test_shelf_e2e_source_sdk.py) | SDK 录制源端到端验收 |
| [`test_strip_cable_damping_fit.py`](tests/test_strip_cable_damping_fit.py) | 衰减协议、参数溯源和 damping 拟合门 |
| [`test_strip_cable_isolated.py`](tests/test_strip_cable_isolated.py) | 独立 cable 编译、provenance 与静态稳定性门 |
| [`test_strip_cable_scene.py`](tests/test_strip_cable_scene.py) | 柔性双物料模板、接触参数、NOREC 与落稳门 |
| [`test_strip_cable_structure.py`](tests/test_strip_cable_structure.py) | 柔性结构编译探针 |
| [`test_strip_material_fit.py`](tests/test_strip_material_fit.py) | cable 参数单位、几何/物理一致性与 provenance |
| [`test_strip_measurement_check.py`](tests/test_strip_measurement_check.py) | 实物测量和模型选择门 |
| [`test_teleop_timing.py`](tests/test_teleop_timing.py) | 控制子步时钟 |

## E. 可逆归档（33）

归档规则和恢复注意事项见 [`archive/README.md`](archive/README.md)。

### 旧 ECU / 纸箱流水线（18）

[`audit_staged_dataset.py`](archive/ecu/audit_staged_dataset.py)、
[`build_carton_lerobot.py`](archive/ecu/build_carton_lerobot.py)、
[`convert_and_train.sh`](archive/ecu/convert_and_train.sh)、
[`ecu_expert_record.py`](archive/ecu/ecu_expert_record.py)、
[`ecu_expert_batch.sh`](archive/ecu/ecu_expert_batch.sh)、
[`ecu_expert_batch45.sh`](archive/ecu/ecu_expert_batch45.sh)、
[`ecu_full_batch_night.sh`](archive/ecu/ecu_full_batch_night.sh)、
[`ecu_full_batch_v3.sh`](archive/ecu/ecu_full_batch_v3.sh)、
[`ecu_grasp_batch.sh`](archive/ecu/ecu_grasp_batch.sh)、
[`ecu_grasp_batch_v3.sh`](archive/ecu/ecu_grasp_batch_v3.sh)、
[`ecu_hybrid_rollout.py`](archive/ecu/ecu_hybrid_rollout.py)、
[`ecu_policy_rollout.py`](archive/ecu/ecu_policy_rollout.py)、
[`factory_assets.py`](archive/ecu/factory_assets.py)、
[`fast_norm_stats_cruzr.py`](archive/ecu/fast_norm_stats_cruzr.py)、
[`multi_batch.sh`](archive/ecu/multi_batch.sh)、
[`retrain_s2b.sh`](archive/ecu/retrain_s2b.sh)、
[`stage_split.py`](archive/ecu/stage_split.py)、
[`v5_batch.sh`](archive/ecu/v5_batch.sh)。

历史说明见 [`ECU旧流程手册_0820.md`](../../docs/archive/ECU旧流程手册_0820.md)；其中旧命令不代表当前双物料入口。

### 旧单立柱 RL / DAgger（13）

[`pi05_flow_sde.py`](archive/rl/pi05_flow_sde.py)、
[`shelf_dagger_append.py`](archive/rl/shelf_dagger_append.py)、
[`shelf_e2e_dagger.py`](archive/rl/shelf_e2e_dagger.py)、
[`shelf_e2e_reward.py`](archive/rl/shelf_e2e_reward.py)、
[`shelf_e2e_rl_train.py`](archive/rl/shelf_e2e_rl_train.py)、
[`shelf_e2e_rl_validate.py`](archive/rl/shelf_e2e_rl_validate.py)、
[`shelf_e2e_rl_worker.py`](archive/rl/shelf_e2e_rl_worker.py)、
[`shelf_e2e_rlenv.py`](archive/rl/shelf_e2e_rlenv.py)、
[`shelf_e2e_rlhook.py`](archive/rl/shelf_e2e_rlhook.py)、
[`rl_reward_agreement.sh`](archive/rl/rl_reward_agreement.sh)、
[`rl_snap_batch.sh`](archive/rl/rl_snap_batch.sh)、
[`test_pi05_flow_sde.py`](archive/rl/test_pi05_flow_sde.py)、
[`test_shelf_e2e_reward.py`](archive/rl/test_shelf_e2e_reward.py)。

### 历史专家（2）

- [`shelf_e2e_expert.py`](archive/history/shelf_e2e_expert.py)：旧单物料立柱专家；
- [`shelf_e2e_dual_expert.orig.py`](archive/history/shelf_e2e_dual_expert.orig.py)：双物料历史 A/B 备份。

## 归档边界

- 归档是保留，不是当前支持：文件仍存在，但内部旧路径、旧相机和旧数据契约没有纳入当前 88 项测试。
- 如需恢复某条历史线，应先在独立分支或副本中修复路径并补齐测试，不能直接把归档数据并入当前训练集。
- 当前正式链没有兼容 wrapper；请使用上表的新路径，不再使用旧的 `scripts/<file>` 命令。
