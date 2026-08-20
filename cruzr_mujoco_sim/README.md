# CRUZR S2 —— MuJoCo 仿真环境（立柱取放线 + ECU 装配线）

> 2026-08-15 整理说明：本文大部分内容是旧单立柱、ECU 和 online-RL 的历史手册，相关源码现保存在 `scripts/archive/`，不属于当前双物料正式采集链，也不保证可在归档位置直接运行。当前入口、目录和测试命令以 [`scripts/README.md`](scripts/README.md) 为准。

同一台 CRUZR S2 双臂移动机器人、同一套 MuJoCo 运行时下的**两条任务线**：

- **A. 立柱线（pillar）**：导航 → 抓取钢制立柱 → 搬运 → 放到料车第二层货架。
  含脚本专家、奖励函数与 PPO(Flow-SDE) 在线 RL 栈。
- **B. ECU 线（ecu）**：ECU 台架抓取 → 夹具 U 口搭桥 → 重抓 → 插入 3×3 料架。
  含 4 段脚本专家录制器、分段数据流水线、以及 FSM+分段 π0.5 的混合 rollout。

两条线共用 `scripts/core/cruzr_teleop.py` 运行时（同一套 qtgt / grip_cmd / base_vel / control_step 接口），
所以录出来的 state/action 与人工遥操走的是**同一条代码路径**。

打包目的：让共同做这些任务的成员在**同一套仿真环境**下训练/测试模型。

---

## 0. 这个包里有什么 / 没有什么

有：
- **两条线的场景与 3D 资产**，均可独立加载（见 §3 自检）
- 两条线的脚本专家（IK + 闭环导航），能稳定产出成功轨迹
- 立柱线：奖励函数 `shelf_e2e_reward.py`（核心设计物）+ 三层验证 + Flow-SDE/PPO 训练器
  + 114 个课程学习快照
- ECU 线：4 段录制器、段切分 `stage_split.py`、数据质量审计 `audit_staged_dataset.py`、
  混合 FSM rollout `ecu_hybrid_rollout.py`、LeRobot 数据集构建与训练编排脚本

**没有**（体积/许可原因，需自备）：
- π0.5 模型权重与 openpi 代码本身（见 §2）
- 训练数据集、录制视频、日志
- 与这两条任务无关的其它产线资产（`assets/1`、`assets/2`、`lingjian`、`factory`(v1) 等）

---

## 1. 目录结构

```
cruzr_mujoco_sim/
├── assets/
│   ├── cruzr_pgc140.xml           机器人本体（CRUZR S2 + PGC-140 夹爪）——两条线共用
│   ├── cruzr_meshes/              46 个机器人 STL
│   ├── pgc140_meshes/             3 个夹爪 STL（base_link / finger1 / finger2）
│   │  ── 立柱线 ──
│   ├── shelf/meshes/              KitCartAGV.obj（料车）/ SteelPillar.obj（立柱）/ RubberStrip.obj
│   ├── e2e/template_pillar_v1.xml 场景模板（专家按 seed 复制成 e2e_scene_<seed>.xml）
│   │  ── ECU 线 ──
│   ├── cruzr_newusd_scene.xml     ECU 车间场景（jig/ECU 直接 author 在里面）
│   ├── extracted/meshes_model{,_lowpoly}/  113 个车间 OBJ（从 new.usd 抽出）
│   ├── factory_v2/                5 个道具：ecu_module / precision_machine_part /
│   │                              modular_server_drive_rack / kitchen_base_cabinet /
│   │                              checkout_counter（visual OBJ + 贴图 + coacd 凸分解碰撞体）
│   └── pi05_cruzr_ecu_lora/       ECU 训练配置的 norm_stats 目录
├── scripts/                       37 个 .py/.sh，见 §4（立柱线）与 §5（ECU 线）
├── sim/cruzr_grip_control.py      夹爪开合换算（原本在 safe_vla_factory 根下，已随包附上）
└── out/                           详见 out/README.md
    ├── teleop/shelf_e2e/          立柱线专家采集 episodes
    ├── teleop/demos/pillar_v{2..6}_refined/  立柱专家轨迹母本
    ├── teleop/ecu/                ECU 线采集
    ├── logs/                      批产 / smoke 日志
    ├── rollout/                   推理视频
    ├── rl/snap/                   立柱线课程快照
    └── smoke/                     smoke 产物（ready_pose 等）
```

两条线的场景入口不同：
- **立柱线**：`TELEOP_SCENE_XML=<按 seed 生成的 e2e_scene_*.xml>`，绕开道具注入。
- **ECU 线**：不设 `TELEOP_SCENE_XML`，走 `factory_assets.load_model()` →
  `cruzr_newusd_scene.xml` + `inject_v2()` 注入 `factory_v2` 道具。

### ⚠️ 场景 XML 必须写在 `assets/` 里

`cruzr_pgc140.xml` 的 `meshdir="cruzr_meshes/"` 是**全局**的，且货架 OBJ 用
`../shelf/meshes/*.obj` 相对它解析。如果把生成的场景 XML 放到别处（例如 /tmp），
会直接报 `XML Error: Error opening file 'cruzr_pgc140.xml'`。
所有脚本默认 `E2E_SCENE_DIR=<pkg>/assets`，**不要改到包外**。

---

## 2. 两个 Python 解释器（这是本栈最容易踩的坑）

| 角色 | 环境 | 原因 |
|---|---|---|
| **仿真 env** | conda env 带 **MuJoCo ≥ 3.9** | 场景用了 `actuatorfrcrange`，旧版 MuJoCo 直接拒绝加载 |
| **策略 actor** | openpi 的 venv（JAX + π0.5） | 该 venv 里的 mujoco 太旧；我们**不去改动共享的 openpi venv** |

因此 RL 训练时 env 是**独立进程**，通过 UNIX socket 与 actor 通信。
两个解释器的 numpy 主版本不同（2.x vs 1.x），直接 pickle ndarray 会报
`No module named numpy._core` —— 所以 `shelf_e2e_rl_train.py` / `shelf_e2e_rl_worker.py`
自带 `enc`/`dec` 编解码器（把数组编成 dtype+shape+raw bytes）。**不要用普通 pickle 绕过它。**

只跑仿真/专家/数据采集的话，只需要 MuJoCo 那个环境，不需要 openpi。

### 依赖

```bash
# 仿真侧（本包在 Python 3.11 + mujoco 3.9.0 上验证通过）
conda create -n mjx python=3.11 && conda activate mjx
pip install "mujoco>=3.9" numpy scipy imageio imageio-ffmpeg
# 渲染用 EGL 无头模式：export MUJOCO_GL=egl

# 策略侧：按 openpi 官方说明安装（https://github.com/Physical-Intelligence/openpi）
```

### 站点相关路径 → 环境变量覆盖

包里保留了本机的默认值，**换机器只需设这些变量**：

| 变量 | 默认值（本机） | 用途 |
|---|---|---|
| `RL_MJX_PY` | `/data1/hsr/tools/miniconda3/envs/mjx/bin/python` | 仿真解释器 |
| `OPENPI_CLIENT_SRC` | `/data1/hsr/openpi-main/packages/openpi-client/src` | openpi-client 源码 |
| `RL_CKPT` | `.../checkpoints/pi05_cruzr_e2e_v2/cruzr_shelf_e2e_v2full/159999` | BC 权重（orbax） |
| `RL_CONFIG` | `pi05_cruzr_e2e_v2` | openpi 训练配置名 |
| `E2E_SCENE_DIR` | `<pkg>/assets` | 场景生成目录（**别改到包外**） |
| `RL_SNAP_DIR` | `<pkg>/out/rl/snap` | 课程快照 |

`shelf_e2e_build_v2.py` / `shelf_dagger_append.py` 里的 LeRobot 数据集路径（`SRC`/`OUT`/`CORR`）
也是本机路径，只有在你要重建数据集时才需要改。

---

`cruzr_teleop.py` 会把**包根**加进 `sys.path`（优先于原来的 safe_vla_factory 根），
所以随包附带的 `sim/` 会生效；`teleop_timing.py` / `factory_assets.py` 也已一并附上。

`factory_assets.py` 同时支持 v1（`assets/factory/`）和 v2（`assets/factory_v2/`）两套道具。
当前默认路径 `load_model()` 只走 **v2**，所以 v1 的 `assets/factory/`（966 MB）**没有随包**，
缺失不影响任何一条线；`factory_assets.py` 注释里提到它只是历史遗留。

ECU 线还需要这些环境变量（训练侧）：

| 变量 | 默认值（本机） | 用途 |
|---|---|---|
| `OPENPI_PY` | `.../envs/openpi_env/bin/python` | openpi 的解释器 |
| `OPENPI_ROOT` | `/data1/hsr/openpi-main` | openpi 代码树（训练/norm-stats 要 `cd` 进去） |

---

## 3. 三十秒自检

```bash
export PKG=/path/to/cruzr_mujoco_sim

# A. 立柱线场景
$RL_MJX_PY - <<'PY'
import os, mujoco
S = os.environ["PKG"]
scene = os.path.join(S, "assets", "_selftest.xml")
open(scene, "w").write(open(os.path.join(S, "assets/e2e/template_pillar_v1.xml")).read())
m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m)
for _ in range(200): mujoco.mj_step(m, d)
print("pillar OK", m.nbody, m.nmesh, d.time); os.remove(scene)
PY
# 期望：pillar OK 49 49 0.2

# B. ECU 线场景（必须在包根下跑：factory_assets 按相对路径找 assets/）
cd $PKG && $RL_MJX_PY - <<'PY'
import os, sys; sys.path.insert(0, "scripts"); os.environ["MUJOCO_GL"] = "egl"
import mujoco, factory_assets as FA
m = FA.load_model(free=False); d = mujoco.MjData(m)
for _ in range(200): mujoco.mj_step(m, d)
print("ecu OK", m.nbody, m.nmesh, m.ngeom, d.time)
PY
# 期望：ecu OK 52 297 339 0.2   ← nmesh/ngeom 必须完全对上，少了说明资产没解压全

$RL_MJX_PY $PKG/scripts/archive/rl/test_shelf_e2e_reward.py    # 历史归档测试
```

### 打包时在解压副本上实测过的结果（不是推断）

| 线 | 检查 | 结果 |
|---|---|---|
| 立柱 | 场景独立加载并步进 200 步 | `nbody=49 nmesh=49 nu=19`（46 机器人 + 3 夹爪 STL） |
| 立柱 | 奖励纯逻辑单测 | 14/14 passed |
| 立柱 | 专家 seed=7 全流程 | `grip_firm PASS` / `placed PASS` / **EPISODE PASS**，4504 帧 |
| 立柱 | 同 seed 挂 RL 钩子打分 | `expert_return=+65.03 term=success`，5 个 latch 全亮（包内记录值 63.89，差异是接触求解的非确定性） |
| 立柱 | 专家 seed=9 | EPISODE FAIL —— 该 seed 的**真实**表现（原批次同样 DROP，包内无 snap_000009），不是打包缺陷 |
| ECU | 场景加载并步进 200 步 | `nbody=52 nmesh=297 ngeom=339 nu=19`，与原始工程**完全一致** |
| ECU | 专家 `grasp_only` seed=1 bay(0,1) | grasp_ik / approach_undisturbed / grasp_contacts / lift 全 PASS，**EPISODE PASS**，257 帧 |
| ECU | 专家 `multi` seed=48 bay(2,2) | S1 PASS / S2 PASS(213 帧) / S3 PASS(290 帧) / **S4 FAIL**（preinsert_check y=0.300 超出 (0.256,0.284)） |

关于最后一行：我拿**原始工程**跑了同一个 seed/bay 做对照，得到**逐位相同**的失败
（`y=0.300 yaw=-1.7 tilt=33.2`，各段帧数 213/290 也完全一致）。所以这是该 seed 上专家自身的
S4 失败，不是打包缺陷；同时也说明**这个包能精确复现原环境的物理行为**。

即：**两条线的专家都 ≠ 100% 成功**。立柱线 seeds 1–200 里 114 个成功，这是它的真实通过率；
ECU 线 S4 插入本来就是最弱的一段，批量录制时要按 bay 轮转并筛选。

---

## 4. 立柱线怎么跑

### 4.1 专家（生成一条成功轨迹 + 视频）

```bash
cd $PKG
SEED=1 EXPERT_OUT=out/teleop/shelf_e2e/demo_1 MUJOCO_GL=egl $RL_MJX_PY scripts/archive/history/shelf_e2e_expert.py
```
关键环境变量：`SEED`(必需)、`EXPERT_OUT`、`E2E_KICKS`(纠偏次数, 默认2)、
`E2E_NOREC=1`(不落 85MB/回合的视频)、`E2E_RLHOOK=1`(挂 RL 钩子, 打分并存快照)。

当前双物料 batch 位于 `scripts/collection/shelf_e2e_batch.sh`；本节旧单立柱批量说明不再适用。

### 4.2 奖励函数（本项目的核心）

`scripts/archive/rl/shelf_e2e_reward.py` —— 纯逻辑 `PillarReward` + `MujocoSensor` 历史适配器（不 import MuJoCo，
所以可以单测）。历史设计见 [`立柱线Online_RL奖励设计_0820.md`](../docs/archive/立柱线Online_RL奖励设计_0820.md)。三条硬骨架：

1. **稠密项全部 potential-based**（Ng et al. 1999）：`Φ(s)=k(s)+φ_k(s)`，k 是 latch 的单调阶段号。
   `gamma_shape=1.0` 与 RL 的 `γ=0.995` **解耦** → 任何闭环轨迹 shaping 累积**恰好为 0**，
   "来回刷分"这一整类 hack 在形式上不存在。
2. **稀疏事件严格有序且 latch**：touch +2 → grasp +5 → lift +10 → arrive +5 → place +40（终止），
   合计 62。跳过任一阶段则后面一分不给。
3. **反作弊闸门**：lift 要求立柱-货架接触力 < 1 N（堵住"撬着架子抬"）；place 要求**松爪 + 货架支撑力
   持续 2 s**（堵住"悬停在区域里"）；place 还要求 lift 已 latch（堵住"把立柱推进区域"）。

其中 "掉落" 的定义是 **无支撑且正在下落**（用 `obj_vz` + 支撑力），不是"脱手"——真实的放置动作会
先松一只手、让立柱沿货架边沿滑落约 1.6 s，期间接触力时断时续。按"脱手"判会把成功放置误判成掉落。

三层验证（跑法见 §4.3）：14 个纯逻辑单测 + 4 个在环退化 agent + 专家在环实测。
实测判别力：专家成功 **+63.9…+65.2**，专家真掉落 +13.4/+14.7，原地打转 −7.34，
静止 −1.20，直撞 −10.64，张爪 −10.08，20 次闭环 shaping = 0.0000。

### 4.3 验证

```bash
$RL_MJX_PY scripts/archive/rl/test_shelf_e2e_reward.py
$RL_MJX_PY scripts/archive/rl/shelf_e2e_rl_validate.py
scripts/archive/rl/rl_reward_agreement.sh 13 17 42
python scripts/archive/rl/test_pi05_flow_sde.py
```
`rl_reward_agreement.sh` 是最有用的一个——它抓出过两个真 bug（受控放置误判为掉落、
碰撞力点采样漏掉力尖峰）。改奖励之后**一定**要重跑它。

### 4.4 课程快照

```bash
scripts/archive/rl/rl_snap_batch.sh 1 200 6 0,2
```
只保留奖励判定 `term=success` 的 seed。本包已含跑完的结果：**seeds 1–200 → 114 个可用快照**，
expert_return 63.12–65.18（均值 64.22）。每个 npz 含 `pre_grasp` / `post_lift` / `pre_place`
三个复位点的 `qpos/qvel/qtgt/grip/base_tgt/t`，外加 `reach_offset`、`region_center`、
838×9 的 `trace`。**读取必须 `np.load(..., allow_pickle=True)`**（`anchors` 是 object 数组）。

### 4.5 在线 RL（PPO + Flow-SDE）—— 见 §5 的已知问题

```bash
cd $PKG
RL_PHASE=A RL_ENVS=8 RL_ITERS=500 \
  <openpi_venv>/bin/python scripts/archive/rl/shelf_e2e_rl_train.py
```
课程三阶段（gate 是评估标准，改之前先和团队商量）：
- **A**：从 `pre_grasp` 复位，学 reach+grasp+lift —— gate: grasp 成功率 ≥ 60%
- **B**：从 `post_lift` 复位，学 transport+place —— gate: place 成功率 ≥ 50%
- **C**：全任务，专家状态复位概率从 0.5 退火到 0

设计要点：
- **Flow-SDE**：π0.5 的确定性 Euler flow 是个没有密度的 Dirac 策略。把每步改写成
  `N(x_k + dt·v_θ, (σ√|dt|)²I)`，整条去噪路径就有了可算的密度。RL 的"动作"是**整条路径**，
  env 只执行最终的 x_0 动作块。诚实声明：**没有加 `(σ²/2)·score` 漂移修正**，所以边缘分布与
  BC 的 ODE 不完全一致；PPO 只需要一个可采样可打分的密度，这在实用上够了。
- **非对称 actor-critic**：critic 用特权特征（state22 + 阶段 one-hot + 5 个 latch），
  actor 保持纯视觉。特权信息只进奖励和 critic，**绝不进 observation**。
- **骨干选型**：JAX 原生自实现，**不用 RLinf** —— 热启动必须来自我们 LoRA 微调出来的 orbax 权重，
  JAX→torch 移植是另一个研究级工程。openpi 本身**未被修改**，本包只读它。

### 4.6 BC 动作契约（写自定义 env/rollout 时必须对齐）

一个动作块里的动作对应 **2 个控制步**，每 **8 个动作**重新推理一次
→ 一个决策 = 16 个控制步 = **0.272 s**；`action_horizon=16`。
`shelf_e2e_rlenv.py` 里对应 `ACT_PER_DECISION=8`、`SUBSTEPS_PER_ACT=2`。
夹爪通道是**张开度**：`a[14]=1` 是**张到最大**，不是夹紧。

---

## 5. ECU 线怎么跑

任务：ECU 台架抓取 → 夹具 U 口**搭桥**（ECU 像盖子一样横跨 U 口，不是塞进去）→ 重抓 →
插入 3×3 料架某个 bay。共 4 段（S1 pick-from-stand / S2 place-on-fixture /
S3 pick-from-fixture / S4 insert-into-rack）。

### 5.1 专家录制

```bash
cd $PKG
# 只录第一段（最快，验证环境用）
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 TELEOP_RECORD_GPU=2 TELEOP_HOME=droop \
  EXPERT_STAGE=grasp_only EXPERT_SEED=1 EXPERT_BAY=0,1 EXPERT_OUT=ecu_demo_1 \
  $RL_MJX_PY scripts/archive/ecu/ecu_expert_record.py

# 全链一次跑出 _st2/_st3/_st4 三段子回合（转场不渲染，省约 4 倍时长）
EXPERT_STAGE=multi EXPERT_SEED=48 EXPERT_BAY=2,2 EXPERT_OUT=ecu_v6_s48 ...同上
```
关键变量：`EXPERT_SEED`（0 = 无扰动，>0 自动加 `EXPERT_NOISE=0.0035` 抖动）、
`EXPERT_BAY="row,col"`（row 0/1/2 = 下/中/上层，col 0/1/2 = 南/中/北）、
`EXPERT_GRIP`（夹持深度，默认 0.025）、`EXPERT_STAGE=multi|grasp_only`。

批量：`bash scripts/archive/ecu/multi_batch.sh <seed_lo> <seed_hi> [workers]`
（bay 轮转，**跳过 (1,1)**——那格被 jig2 占着）。`v5_batch.sh` 是五路相机版本。

### 5.2 数据流水线

```bash
$RL_MJX_PY scripts/archive/ecu/stage_split.py
$RL_MJX_PY scripts/archive/ecu/audit_staged_dataset.py
bash scripts/archive/ecu/convert_and_train.sh
```
`stage_split.py` 按右夹爪开合度 + 底盘速度切段，每段留 6 帧重叠头覆盖转场，
并对底盘转场帧做 1/3 降采样。

**审计门有个已知教训**：早期的录制审计只验 ECU 结果（未被扰动/接触/抬升），
**没验手臂动力学**——手腕/前臂刮蹭结构件"卡死-弹通"时 ECU 没动，于是全部漏网，
污染了一整版 S1 数据并让 M1 段成功率从 ~77% 崩到 ~31%。
**新审计门必须加：任意关节单帧 state 跳变 > 100 mrad 即拒收。**

### 5.3 混合 FSM + 分段 π0.5 rollout

```bash
POLICY_PORT=8735 ROLLOUT_SPAWN_SEED=48 ROLLOUT_BAY=2,2 ROLLOUT_OUT=out/hybrid_s48 \
  $RL_MJX_PY scripts/archive/ecu/ecu_hybrid_rollout.py
```
架构 = **脚本 NavTo 导航 + 4 段 VLA 操作**。支持 `POLICY_PORT_M1..M4` 分段选不同 checkpoint。
相机契约由 `ROLLOUT_CAMS` 切换，默认 `stereo_left,stereo_right,waist_front`（真机三路，无腕相机）。
`ROLLOUT_LAYOUT_RAND=1` 开布局随机化（±8cm/±8°）检验动态导航。

**这条线最重要的方法论**：每个 FSM↔策略接缝都必须置位到**下一段数据的起点分布**。
段间掰姿态要用**该段自己的**携带位（各段不同），掰错会把 ECU 甩回夹具。
另一个反复踩的坑：固定坐标泊车会让 M2 系统性 0/6——数据里的泊车是**对物件闭环**的，
必须改成物件相对泊车。

### 5.4 ECU 线运行注意

- `pkill`/`pgrep -f` 会匹配到自己的命令行文本（这里自杀过三次），一律用 `pgrep -f "xxx[x]"` 括号写法。
- 起新 policy server 前必须等端口真正释放（循环查 `ss -tln`）；训练前先杀干净残留 server 防 OOM。
- π0.5 是流采样，server RNG 随请求推进，**同 seed 不可复现**——评估只能看比率 + 置信区间，
  单轮 n=16 往往不够。

---

## 6. 已知未解决问题（接手前务必先读）—— 立柱线在线 RL

1. **PPO ratio 在 bf16 下不可靠 —— 每次更新都被早停，`pg/kl/vf` 恒为 0。**
   路径 log-prob 量级 ~3e3；rollout(batch=1) 与更新(minibatch=4) 的 XLA 融合方式不同，
   bf16 相对误差 1e-4 就放大成绝对差 0.33 → ratio 1.39 → k3 KL 0.06 超阈值。
   **正确修法**：不要存 rollout 时的 logp，而是在每轮更新开始时，用与 loss 完全相同的函数、
   完全相同的 batch 形状、在旧参数下重算 `logp_old`（bf16 噪声在 ratio 里精确抵消，起点严格为 1）。
   **不要**通过调大 `target_kl` 来"解决"，那是掩盖。
2. **吞吐约 26–31 s/决策步**，主要是 XLA 反复编译（6 次 transformer 前向 + 每个 minibatch 形状重编）。
   需要给 `sample_path` / `path_logprob` 套 `jit`（形状固定后缓存）。
3. 因此 `shelf_e2e_rl_train.py` 的 smoke **尚未通过**（能跑完但学不到东西）。
   奖励函数、env、Flow-SDE 三块是验证过的；卡住的只有 PPO 更新这一环。

另外两个数值上的"这不是 bug"：
- σ→0 与 BC ODE 的一致性只能到 **8 ulp of bf16**（bf16 eps = 2⁻⁸ = 3.9e-3）。要求 1e-3 原理上做不到。
- 对全参数求梯度会把 24 GB 显卡打爆；必须只对 LoRA 叶子求导
  （`nnx.DiffState(0, config.trainable_filter)`）并对去噪步做 `jax.checkpoint` remat。

---

## 7. 背景（立柱线）

模仿学习全家族（ACT / Diffusion Policy / π0.5 × 各数据版本 × DAgger）闭环 **0/6 全灭**之后，
在线 RL 是剩下的路径。BC 最典型的失败模式是原地打转（yaw 累积 6–55 rad）——奖励里那条
**路径相关**的 yaw 惩罚（每步 −0.15·|Δyaw|，累积到 4π 终止并 −5）就是专门针对它的：
转回来**不能**退钱。
