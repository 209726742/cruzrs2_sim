# 在线 RL 奖励函数设计 — CRUZR S2 立柱端到端取放

> 归档状态：旧单立柱 Online RL 设计，仅与 `scripts/archive/rl/` 中的历史实现配套，不用于当前双物料 π0.5 主线。
>
> 归档说明（2026-08-15）：这是一条旧单立柱 online-RL 设计线，实现在 `scripts/archive/rl/`，旧专家在 `scripts/archive/history/`。下文的 `scripts/<文件>` 是移动前的历史路径，不属于当前双物料 π0.5 数据采集入口。

版本 v1（设计稿，待评审后实现）
基线策略：π0.5 LoRA `pi05_cruzr_e2e_v2full/159999`（BC 闭环 0/6，开环 MAE 0.003）
环境基础：`scripts/shelf_e2e_rollout.py`（纯视觉、随机布局、已含全部物理判据）

---

## 0. 设计原则（先定死，再填公式）

1. **稠密项必须是 potential-based shaping**（Ng et al. 1999）：`F = γΦ(s') − Φ(s)`。
   这在数学上保证 (a) 最优策略不变；(b) **任何闭环轨迹的 shaping 累积恰好为 0**，
   即"来回刷"这一整类 reward hacking 从形式上被排除。上一轮 BC 观察到的底盘打转
   如果只靠"加个惩罚"去压，是补丁；用势函数是从结构上不给它收益。
2. **稀疏项必须 latch（一次性）且有序**：`touch → grasp → lift → arrive → place`，
   后一级只有在前一级 latch 置位后才可能兑现。防止"不抓直接把柱子推进区域"这类跳级。
3. **每条惩罚必须对应一个已观测到或物理上可预期的失败模式**，不设想象中的惩罚。
4. **奖励要先离线验证再上线**：用现有的专家 PASS 轨迹和 DAgger 失败轨迹回放打分，
   专家分显著高于失败分才允许开训（见 §6）。

---

## 1. MDP 定义

| 项 | 取值 |
|---|---|
| 决策频率 | 每 `REPLAN=8` 个 control step 查询一次策略（≈3.75 Hz），一次 chunk 为一个宏动作 |
| 观测 | 与 BC 完全一致：3 路 224×224 相机 + state22（不加特权信息） |
| 动作 | action18（14 臂关节目标 + 2 夹爪 + v_fwd/wz），裁剪同 `apply_action` |
| 折扣 | γ = 0.995（决策级；有效视界 ≈200 步） |
| 最大长度 | 300 决策步（=2400 control step ≈80 s）。原 7200 太长，RL 采样不起 |
| 奖励计算 | 势函数在 chunk 边界求；惩罚在 chunk 内 8 步上取和/取极值 |

**特权信息只进奖励，不进观测**——这是 RL 相对 BC 的唯一"外挂"，也是合法的。

---

## 2. 阶段机（latched，单调不回退）

```
k=0 APPROACH  尚未停在抓取工位
k=1 REACH     底盘已入抓取工位（|Δxy|<0.15 且 |Δyaw|<0.15）
k=2 GRASP     双爪与柱子均有接触
k=3 LIFT      双手 ≥1N 且 z_o > Z0+0.06 且 柱子-货架接触力 <1N（真离架）
k=4 TRANSPORT 持物且底盘朝小车工位移动
k=5 PLACE     in_region() 且已松爪且柱子静置
```

`k` 只增不减。掉物触发终止（§4.2），不是回退。

---

## 3. 稠密项：分段势函数

记 `p_h` = 双手 mount 中点（`d.xpos[ct.L.mount]`/`ct.R.mount`），
`p_g` = 抓取点 = `obj_pos() + [0,0,h_grip]`（h_grip 取专家 demo 抓取时的手-柱相对高度），
`b` = `ct.base_pose()`，`b*_grasp/b*_place` = 专家里的 `park_grasp/park_place`（随 seed 平移，
env 已能算），`p_R` = REGION 中心 + 架高。

各阶段内势 φ_k ∈ [0,1]：

| k | φ_k |
|---|---|
| 0 | `0.7·(1−tanh(‖b_xy−b*_g,xy‖/0.5)) + 0.3·(1−|wrap(yaw−yaw*_g)|/π)` |
| 1 | `1−tanh(‖p_h−p_g‖/0.25)` |
| 2 | `min(1, (f_l+f_r)/(2·F_hold))`，F_hold=3 N |
| 3 | `clip((z_o−Z0)/0.10, 0, 1)` |
| 4 | `0.7·(1−tanh(‖b_xy−b*_p,xy‖/1.0)) + 0.3·(1−|wrap(yaw−yaw*_p)|/π)` |
| 5 | `1−tanh(‖p_o−p_R‖/0.4)` |

全局势 **Φ(s) = k(s) + φ_{k(s)}(s)**，值域 [0,6]，跨阶段边界连续单调。

```
r_shape(t) = λ_s · ( γ·Φ(s_{t+1}) − Φ(s_t) ),   λ_s = 1.0
```

整条 episode 的 shaping 累积上界 ≈ Φ_max − Φ_0 ≈ 6 —— 与稀疏总奖励（§4.1，62）
同量级但更小，保证 shaping 是"引路"而不是"目标"。

> 为什么用 `tanh` 而不是 `−距离`：tanh 有界，远处梯度平缓，近处（<0.25 m）梯度陡，
> 正好把学习压力压在"最后几厘米"这个 BC 失败的地方。

---

## 4. 稀疏项

### 4.1 事件奖励（latch，一次性，须按序）

| 事件 | 判据（沿用 rollout 里已有的物理量） | 奖励 |
|---|---|---|
| touch | 双爪 pad 与 OBJ_GEOMS 均有接触 | +2 |
| grasp | `grip_force('l')≥1 且 ('r')≥1` **持续 0.5 s** | +5 |
| lift | grasp 保持 且 `z_o>Z0+0.06` 且 柱-架接触力<1 N，持续 0.5 s | +10 |
| arrive | lift 已置位 且 `‖b_xy−b*_p,xy‖<0.25` 且仍持物 | +5 |
| place | `in_region()` 且双爪松开(f<0.5 N) 且 柱子与架板有支撑接触，持续 2 s | **+40，终止** |

合计 62。lift 里加"柱-架接触力<1 N"是为了堵住**靠架子撬起来**的假抬升。
place 要求"已松爪 + 有支撑接触"是为了堵住**举着柱子悬停在区域内**骗成功。

### 4.2 惩罚（每条都对应一个真实失败模式）

| # | 失败模式 | 惩罚 |
|---|---|---|
| P1 | **底盘打转**（已观测：yaw 累积 6–55 rad） | `−0.15·|Δyaw|` 每步；累积 `|Δyaw|>4π` 立即终止 `−5` |
| P2 | 掉物 | lift 后双手力<0.5 N 且 z 较持物峰值跌 0.05 → `−10` 终止 |
| P3 | 撞翻/推掉柱子 | 抓前柱子倾角>45° 或 z 跌 0.05 → `−5` 终止 |
| P4 | 硬碰撞 | 非 pad 机体-环境接触力 >50 N：`−0.5/步`；>200 N：`−10` 终止 |
| P5 | 抖动 | `−0.02·‖a_arm(t)−a_arm(t−1)‖²` |
| P6 | 空耗 | `−0.002/决策步`（300 步共 −0.6） |
| P7 | 顶关节限位 | 每个被 clip 的关节 `−0.02/步` |
| P8 | 能耗/急动 | `−0.01·(v²+ω²)` |

P1 标定说明：完成任务所需转角约 ≤3.5 rad → 代价 0.52，远小于 R_grasp=5；
而 55 rad 的打转代价 8.25 且早已被 4π 终止砍掉。**该项是唯一一个非势函数的
路径依赖惩罚，是刻意的**——打转必须付真金白银，不能靠"转回来"退款。

---

## 5. 训练回路（πRL / Flow-SDE 风格）

- **算法**：PPO（clip 0.2）+ GAE(λ=0.95)。flow-matching 动作的 log-prob 走 Flow-SDE：
  把去噪 ODE 改成等价 SDE，注入噪声后每步可算 log-prob，整条去噪链的和作为 chunk 的 log-prob。
- **warm start**：actor = BC ckpt `v2full/159999`；critic 新建（state22+视觉特征 → V），先用
  BC 轨迹回归 λ-return 预热 2k 步再联合更新。
- **BC 锚**：loss 里加 `β·KL(π‖π_BC)`，β=0.02，防止 RL 早期把 BC 的全部结构打烂。
- **课程（关键，不做必失败）**：BC 闭环 0/6 意味着从头探索抓取几乎不可能采到正样本。
  用专家轨迹做 reset 分布：
  - Phase A：从专家的 pre-grasp 状态起（底盘已停好），horizon 100 步，只学 grasp+lift。
    门槛：grasp 成功率 ≥60% 才进 B。
  - Phase B：加入 place（从 lift 后状态起）。门槛：place ≥50%。
  - Phase C：全流程从随机初始态起，reset 状态按 `p=0.5` 从专家状态采样，
    随训练线性退火到 0（DemoStart 退火）。
- **并行**：MuJoCo 无渲染部分可并行，但本任务观测是渲染图像 → 每 env 一个 Renderer，
  显存约 0.6 G/env，单卡 4090 开 8 env，2 卡采样 + 1 卡训练。
- **回报归一化**：running return std 归一化（PPO 标配），γ 已定，不再手调 scale。

---

## 6. 上线前的离线校验（必须先做，成本几分钟）

把奖励函数写成独立模块 `shelf_e2e_reward.py`（纯函数：`(m, d, latches, prev) → (r, info)`），
然后**回放已有轨迹打分**：

| 轨迹集 | 期望 |
|---|---|
| 专家 PASS episode（`out/teleop/shelf_e2e/shelf_e2e_pillar_*`，success=True） | 总回报 ≈ +55 ~ +62 |
| 专家 FAIL episode | 明显更低，且 info 里 latch 停在正确阶段 |
| DAgger 6 个失败 rollout（打转的那批） | 显著为负（P1 主导），且 stage 停在 k=0 |
| 人造 hack：原地打转 300 步 | 强负 |
| 人造 hack：把柱子推下架子滑进区域 | 不给 place 奖励（lift latch 未置位） |

这 5 条全过，才认为奖励函数没有明显漏洞，再开训。

---

## 7. 待定/需要用户拍板的点

1. `R_place=+40` vs 惩罚上限：如果希望"宁可慢也别撞"，P4 可以调到 −20 终止。
2. Phase A 的 60% / Phase B 的 50% 门槛是我拟的，属于 evaluation criteria，按惯例需要你确认。
3. 是否允许 reward 里用特权信息（物体位姿/接触力）—— 我默认允许（仅奖励，不进观测），
   因为这是仿真 RL 的标准做法；若要求纯视觉奖励则整套要重做。

---

# v2 — 经离线+在环校验后的修订（2026-07-28）

设计稿（上文）实现为 `scripts/shelf_e2e_reward.py`，校验为 `scripts/test_shelf_e2e_reward.py`
（13 个纯逻辑单测，不依赖 MuJoCo）和 `scripts/shelf_e2e_rl_validate.py`（4 个在环退化智能体）。
§6 全部通过后才动训练。**校验抓出 4 个真缺陷，都是设计层面的，不是代码笔误：**

| # | 缺陷 | 现象 | 修正 |
|---|---|---|---|
| B1 | 掉落判据把"放下"当"掉了" | 专家实际 `placed PASS`，奖励却给 term=drop、+5.88 | 专家是在架上方 11 cm 松爪让柱子落定的。drop 改为"脱手且**不在目标区内**并持续 1 s"，或 z 跌破架面 0.25 m 立即判 | 
| B2 | γ=0.995 的 shaping 折扣泄漏 | 838 步 episode 里 (γ−1)ΣΦ ≈ −12，比 62 分事件预算还狠，能干的专家反而低分 | shaping 内部用 `gamma_shape=1.0`（与 RL 的 γ 解耦）。闭环累积从"≤0"变成"恰好 0"，抗刷属性更强 |
| B3 | 课程热启动时掉落检测是死的 | Phase B 从 post_lift 恢复后立刻松爪，居然 timeout 而非 drop | `_z_peak` 只在"持物"时更新，热启动那一刻已不持物 → 永远是 −inf。改为 lift latch 已置位时用当前 z 播种 |
| B4 | dt 写错 | 按 8/30 s 算，实际是 8×17×1 ms | dt=0.136 s；hold 计时全部随之修正 |

## 校验结果（实测，非推演）

| 场景 | 结果 |
|---|---|
| 专家成功 episode（真实 MuJoCo，seed 1/18/19…） | **+63.9 ~ +64.1，term=success**，五个 latch 全置位 |
| 专家失败 episode（seed 17，运输中真掉落） | **+13.39，term=drop** ——与成功档拉开 50 分 |
| 原地打转（BC 的实际失败模式） | −7.72，term=spin，stage 停在 APPROACH |
| 原地不动 | −2.00，term=timeout（= 1000 步 × k_time） |
| 全速撞货架 | −10.66，term=hard_collision，无任何奖励 latch |
| 从 post_lift 直接松爪扔掉 | −10.17，term=drop |
| 撬柱子（不离架的假抬升） | 给 grasp 不给 lift |
| 举着柱子悬停在区域内（不松爪） | 给 arrive 不给 place |
| 不抓、推柱子进区域 | term=knock，place latch 不置位 |
| 20 圈闭环轨迹的 shaping 累积 | **恰好 0.0000** |

## 落地的组件

- `scripts/shelf_e2e_reward.py` — 奖励逻辑（纯函数，`Sense` 结构体）+ `MujocoSensor` 适配器
- `scripts/shelf_e2e_rlhook.py` — 挂进专家的钩子：课程快照 + 在环打分（`E2E_RLHOOK=1`）
- `scripts/shelf_e2e_rlenv.py` — RL 环境，观测/动作契约与 `shelf_e2e_rollout.py` 完全一致；
  `reset(phase="A"/"B"/"C")` 分别从 pre_grasp / post_lift / 随机初始态起
- `scripts/rl_snap_batch.sh` — 批量生成课程快照（只留专家 term=success 的 seed）
- `scripts/pi05_flow_sde.py` — π0.5 的 Flow-SDE 包装：把确定性流 ODE 变成有密度的高斯链，
  PPO 才有 ratio 可算；动作=去噪路径，环境只执行末端 chunk

## 骨干选型：JAX 原生，不用 RLinf

热启动必须来自我们这个 LoRA 微调的 orbax 权重（v2full/159999）。RLinf 是 PyTorch 栈，把一个
自定义 LoRA 的 pi05 从 JAX/orbax 移植到 torch 本身就是研究级工程，风险远大于自己实现 Flow-SDE
+ PPO。openpi 侧一行不改，全部新代码放在 mujoco_teleop/scripts。

Flow-SDE 的诚实偏差：严格版会在漂移里加 (σ²/2)·score 以保持与 ODE 同边际；我们不加。PPO 只需
一个可采样可打分的密度，不需要边际守恒；偏离由 σ 和 loss 里的 KL-to-BC 锚控制。已写进模块注释。
