# π0.5 30K checkpoint 策略推理与视频录制工作说明

## 这份文档要解决什么问题

当前已经完成训练的目录是：

```text
<PROJECT_ROOT>/cruzr_mujoco_sim/out/training/
pi05_fullft_formal300_h100x4_b16_30k_20260818
```

下一步的目标不是继续训练，而是把最终 checkpoint 放回与训练数据相同的 Cruzr S2 双物料 MuJoCo 任务中，让策略根据三路相机和机器人状态持续生成动作，闭环控制机器人完成“先把钢制立柱放到料车中层，再把柔性条料放到顶层”的任务，同时录制第三人称 MP4，并保存机器可读的成功判定结果。整个流程不应该使用 `/workspace/GlobalHumanoidRobotChallenge_2026_Baseline/run_a_policy_inference.sh`，因为 `/workspace` 中那套现成入口绑定的是 `walker_s2_sim + Part_Sorting`，机器人、场景和观测语义都与这里训练的 Cruzr 双物料策略不同。

当前仓库已经具备完成这件事所需的主要代码，因此不需要重新训练、转换权重，也不需要把 checkpoint 复制到 `/workspace`。真正要做的工作，是先启动一个负责加载大模型的策略服务，再启动一个负责运行 MuJoCo、请求策略动作并写视频的 rollout 进程。完成一次短流程联调之后，再运行完整时长和多 seed 评测。

## 当前已经确认的事实

训练目录中的 `checkpoints/last` 已经指向 `030000`。`030000/pretrained_model` 内存在约 9.35 GB 的 `model.safetensors`、`config.json`、训练配置、策略预处理器、策略后处理器以及对应的归一化参数。训练状态中的 step 也是 30000，模型权重、优化器状态和 RNG 状态均能用 safetensors 正常打开，因此这是一个完整 checkpoint，而不是训练中断留下的半成品。

该模型对应的训练数据是 `formal/cruzr_shelf_v24_300source`，数据集任务版本为 `dual_two_trip_v1`，采集配置为 `sdk_recovery_v1`。策略真实接收 18 维状态，包括 14 个手臂关节、左右夹爪状态以及底盘的前向速度和角速度；输出也是 18 维，包括 14 个关节目标、两个夹爪命令和两个底盘速度命令。三路策略图像严格依次对应 `stereo_left`、`waist_front` 和 `chassis_front`，分辨率为 224×224。训练语言指令是：

```text
move the steel pillar to the middle shelf of the cart, then move the rubber strip to the top shelf
```

checkpoint 的 `config.json` 中会看到 `observation.state.shape = [32]`。这并不表示 rollout 应该凭空构造 32 维状态，也不应该手工在 state18 后面补零。32 是 π0.5 模型内部允许的最大状态维度，训练和部署的真实状态仍然是 18 维，模型内部会完成 padding。已经使用该 30K checkpoint 做过完整的假观测推理验证：state18 可以正常通过已保存的预处理器，模型能够在当前 RTX 4090 上加载并返回形状为 `(50, 18)`、类型为 `float32` 且全部为有限值的动作序列。因此当前模型与通用策略服务之间的基本契约已经验证通过。

模型冷启动并不快。当前机器从共享存储读取 9.35 GB 权重、构建模型并移动到 GPU，需要预留大约三到五分钟。模型进入 GPU 后观察到约 9.5 GB 显存占用，单次完整推理已在当前 24 GB RTX 4090 上通过。启动策略服务后应耐心等待端口真正进入监听状态，不能因为几十秒没有新输出就误认为它卡死。

## 正确的运行结构

这里采用两个进程，是因为策略模型和 MuJoCo 仿真依赖不同的 Python 环境。策略服务使用训练时的 `/isaac-sim/python.sh`，它负责加载 checkpoint、执行预处理、调用 `predict_action_chunk()` 并反归一化动作；MuJoCo rollout 使用项目内的 `envs/mjx/bin/python`，它负责创建随机场景、渲染三路策略相机、构造 state18、通过本机 WebSocket 请求动作、执行动作，并把第三人称画面编码为 MP4。

策略服务入口是：

```text
cruzr_mujoco_sim/scripts/collection/lerobot_policy_server.py
```

仿真和录制入口是：

```text
cruzr_mujoco_sim/scripts/collection/shelf_e2e_rollout.py
```

不要使用 `smoke/pillar_lerobot_policy_server.py` 跑这个 30K checkpoint。该 smoke 服务会把 checkpoint 配置里显示的状态形状强制校验为 18，而本 checkpoint 为了模型内部 padding 显示为 32，会被它提前拒绝。`scripts/collection/lerobot_policy_server.py` 才是本次应使用的入口，它既使用正确的 SDK 三相机映射，也已经验证能把真实 state18 送进这个 checkpoint。

同样重要的是，运行 rollout 时必须显式设置：

```text
E2E_COLLECTION_PROFILE=sdk_recovery_v1
```

如果漏掉这个变量，rollout 会采用默认的 `strict_v1` 相机组合，即便图像张量形状相同，物理视角也会与训练数据不一致，最终效果没有评估意义。

## 第一阶段：启动 30K 策略服务

建议打开第一个终端，进入项目根目录后执行下面的命令。这里使用显式的 `030000` 路径，便于记录实验所用权重；使用 `checkpoints/last/pretrained_model` 在当前状态下也是等价的。

```bash
cd /path/to/cruzr_sim

PROJECT_ROOT="$(pwd -P)"
CHECKPOINT="$PROJECT_ROOT/cruzr_mujoco_sim/out/training/pi05_fullft_formal300_h100x4_b16_30k_20260818/checkpoints/030000/pretrained_model"
CLIENT_WHEEL="$PROJECT_ROOT/cruzr_mujoco_sim/smoke/openpi_client-0.1.2-py3-none-any.whl"
MJX_PACKAGES="$PROJECT_ROOT/envs/mjx/lib/python3.11/site-packages"

mkdir -p "$PROJECT_ROOT/cruzr_mujoco_sim/out/logs/rollout"

PYTHONPATH="$PROJECT_ROOT:$CLIENT_WHEEL:$MJX_PACKAGES" \
HF_HOME="$PROJECT_ROOT/smoke_data/hf_cache" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
/isaac-sim/python.sh \
  "$PROJECT_ROOT/cruzr_mujoco_sim/scripts/collection/lerobot_policy_server.py" \
  --checkpoint "$CHECKPOINT" \
  --host 127.0.0.1 \
  --port 8731 \
  2>&1 | tee "$PROJECT_ROOT/cruzr_mujoco_sim/out/logs/rollout/pi05_30k_policy_server.log"
```

这条命令中较长的 `PYTHONPATH` 不是多余配置。训练用的 Isaac Python 具备 PyTorch、Transformers 和 safetensors，但当前环境没有单独安装 `openpi_client` 和 `msgpack`；仓库自带的 wheel 提供 `openpi_client`，项目的 mjx 环境提供已经验证可用的 `msgpack`。这种方式不会修改 `/isaac-sim` 的系统 Python，也不会因为安装依赖而意外升级或降级 Isaac 使用的 NumPy。离线 tokenizer 则从已经存在的 `smoke_data/hf_cache` 中加载。

终端停在服务进程中是正常现象。首次启动要等待模型加载完成，看到类似下面的日志后，才能在第二个终端启动 rollout：

```text
serving .../checkpoints/030000/pretrained_model on ws://127.0.0.1:8731
```

如果出现 `No module named openpi_client`，说明 `CLIENT_WHEEL` 没有包含在 `PYTHONPATH` 中；如果出现 `No module named msgpack`，说明 `MJX_PACKAGES` 没有包含在 `PYTHONPATH` 中；如果 tokenizer 提示离线找不到 `google/paligemma-3b-pt-224`，应检查 `HF_HOME` 是否准确指向项目内现有的 `smoke_data/hf_cache`。这些问题都不需要重新训练模型。

## 第二阶段：先做 300 步短录制

策略服务就绪后，在第二个终端先运行 300 步 smoke。这个阶段的目标不是判断任务成功率，而是验证真实仿真图像、state18、WebSocket、动作执行和 MP4 编码能够首尾贯通。输出目录应使用独立名称，避免后续完整评测覆盖短测结果。

```bash
cd /path/to/cruzr_sim

PROJECT_ROOT="$(pwd -P)"
mkdir -p "$PROJECT_ROOT/cruzr_mujoco_sim/out/logs/rollout"

MUJOCO_GL=egl \
MUJOCO_EGL_DEVICE_ID=0 \
E2E_COLLECTION_PROFILE=sdk_recovery_v1 \
POLICY_HOST=127.0.0.1 \
POLICY_PORT=8731 \
SEED=10001 \
ROLLOUT_STEPS=300 \
ROLLOUT_REPLAN=50 \
ROLLOUT_OUT=out/rollout/pi05_30k_smoke_seed10001 \
"$PROJECT_ROOT/envs/mjx/bin/python" \
  "$PROJECT_ROOT/cruzr_mujoco_sim/scripts/collection/shelf_e2e_rollout.py" \
  2>&1 | tee "$PROJECT_ROOT/cruzr_mujoco_sim/out/logs/rollout/pi05_30k_smoke_seed10001.log"
```

短测结束后，预期至少出现下面两个文件：

```text
cruzr_mujoco_sim/out/rollout/pi05_30k_smoke_seed10001/e2e_3rd.mp4
cruzr_mujoco_sim/out/rollout/pi05_30k_smoke_seed10001/result.json
```

`e2e_3rd.mp4` 是第三人称最终效果视频。`result.json` 会记录 seed、两种物料是否被抓起、是否分别放到指定层、最终坐标、实际相机配置以及总成功标志。300 步短测大概率没有足够时间完成整个双物料任务，因此 `success=false` 本身不代表模型失败；短测的通过条件是服务端持续返回动作、rollout 没有状态或相机契约错误、视频能够播放、`result.json` 能正常生成，并且日志中没有 NaN、连接中断或动作形状错误。

这里暂时使用 `SEED=10001` 作为新评测范围的起点。正式报告中若要严格声称“unseen seed”，还应把评测 seed 与原始 300 条源 episode 的 seed 清单做一次去重。当前构建后的 LeRobot 数据集元数据记录了源 episode 数量，但没有直接保留完整源 seed 清单，因此在完成来源核对之前，文档只把 10001 当作新的候选评测 seed，而不把它当作已经证明未见的 seed。

## 第三阶段：运行完整时长并查看最终效果

当 300 步 smoke 通过后，保持第一个终端中的策略服务不退出，在第二个终端把 `ROLLOUT_STEPS` 改为默认完整时长 7200，并换一个新的输出目录。脚本注释将 7200 步定义为约 240 秒仿真任务时长；实际墙钟时间还会受到每次 π0.5 推理耗时影响，因此通常明显长于视频时长。

```bash
cd /path/to/cruzr_sim

PROJECT_ROOT="$(pwd -P)"

MUJOCO_GL=egl \
MUJOCO_EGL_DEVICE_ID=0 \
E2E_COLLECTION_PROFILE=sdk_recovery_v1 \
POLICY_HOST=127.0.0.1 \
POLICY_PORT=8731 \
SEED=10001 \
ROLLOUT_STEPS=7200 \
ROLLOUT_REPLAN=50 \
ROLLOUT_OUT=out/rollout/pi05_30k_full_seed10001 \
"$PROJECT_ROOT/envs/mjx/bin/python" \
  "$PROJECT_ROOT/cruzr_mujoco_sim/scripts/collection/shelf_e2e_rollout.py" \
  2>&1 | tee "$PROJECT_ROOT/cruzr_mujoco_sim/out/logs/rollout/pi05_30k_full_seed10001.log"
```

完整结果位于：

```text
cruzr_mujoco_sim/out/rollout/pi05_30k_full_seed10001/e2e_3rd.mp4
cruzr_mujoco_sim/out/rollout/pi05_30k_full_seed10001/result.json
```

查看视频时不应只看机器人最终是否碰巧把物体留在货架附近，还要结合 `result.json` 的物理成功判据。脚本要求物体中心位于对应层范围内、物体底部接近货架支撑面、货架接触力达到阈值，并且左右夹爪均已释放，满足这些条件后才会把 `placed` 和 `success` 标记为真。因此 `result.json` 比人工看一帧截图更适合作为批量统计依据，而 MP4 主要用于判断策略行为是否合理、是否发生抖动、碰撞、错误抓取或反复重规划。

## 第四阶段：从“录一个视频”升级为可信评测

单个 seed 的视频只能证明流水线能运行，不能代表 30K checkpoint 的真实成功率。π0.5 推理本身带有流采样随机性，同一场景 seed 的多次运行也可能得到不同轨迹。建议在首个完整 episode 跑通后，再选一组与训练源去重的布局 seed，每个 seed 至少重复若干次，并为每次运行使用唯一输出目录。最终汇总每个 `result.json` 中的 `success`、`placed` 和 `grasped`，同时保留对应视频用于失败分类。

一个实用的最小评测规模可以先从 10 个布局 seed 开始，每个 seed 跑 2 到 3 次；若只是早期诊断，也可以先跑 3 个 seed。等确认模型确实有完成任务的行为后，再扩大样本量并计算成功率和置信区间。批量运行前最好新增一个很薄的 shell 包装器，它只负责生成唯一目录、顺序启动多个 seed、保存日志和汇总 `result.json`，不应该复制或改写 rollout 的控制逻辑。这样可以避免手工命令中最常见的输出覆盖、漏设相机 profile 和 seed 记录错误。

## 当前视频能力与可能需要补的代码

现有 `shelf_e2e_rollout.py` 已经能够直接生成第三人称 `e2e_3rd.mp4`，这足以观看最终任务效果；仓库中也已经存在早期 pilot checkpoint 通过同一条录制链生成的 MP4，说明编码器和场景资产在当前机器上可用。当前 30K checkpoint 尚未生成对应的 rollout 视频，因此需要按本文先短测、再完整运行。

现有 rollout 会实时渲染三路策略输入相机，但不会把这三路输入分别写成 MP4。如果后续希望同时观看“策略到底看到了什么”，还需要对 rollout 做一个小范围代码改动：为 `stereo_left`、`waist_front`、`chassis_front` 分别增加视频 writer，或者生成一个三相机宫格视频。这个功能不影响第一轮第三人称效果录制，建议在基本闭环确认后再补，避免同时改变推理和录制两部分。

另一个值得补的便利功能，是一个专门面向正式 checkpoint 的一键评测脚本。现有 `smoke/run_pillar_smoke.sh` 把 checkpoint 写死在 smoke 训练目录中，而且使用了会拒绝当前 32 维内部配置的 smoke server，因此不应直接改几个环境变量就用于这次 30K 模型。更稳妥的做法是后续新增一个独立、很短的 launcher，调用本文已经验证的 `scripts/collection/lerobot_policy_server.py` 和 `shelf_e2e_rollout.py`，并把 checkpoint、seed、步数和输出目录做成显式参数。

## 建议的验收顺序

当前最合理的工作顺序是：先确认策略服务能够在三到五分钟后进入监听状态；然后跑 300 步短测，确认 WebSocket、相机映射、state18、动作输出和视频编码全部正常；再跑一个 7200 步完整 episode，人工观看视频并核对 `result.json`；最后才开始多 seed、多次重复的成功率评测。如果完整 episode 表现异常，应优先检查相机 profile、任务 prompt、动作频率、抓取和底盘行为，而不是立即重新训练。

完成第一轮工作的最低验收标准是：使用明确记录的 `030000` checkpoint；日志中显示策略服务成功加载；rollout 明确使用 `sdk_recovery_v1` 和三路正确相机；生成非空且可播放的 `e2e_3rd.mp4`；生成结构完整的 `result.json`；服务器与 rollout 均没有 NaN、动作形状、状态形状、相机键或 WebSocket 错误。在满足这些条件之后，才进入模型效果本身的判断阶段。

## 实际执行记录

后续工作已于 2026-08-19 17:30 UTC 开始。启动前检查显示当前机器只有一张 NVIDIA GeForce RTX 4090，显存总量 24564 MiB，检查时仅使用 1 MiB；没有残留的训练、策略服务或 rollout 进程，8731 和 8735 端口均未被占用。Git 工作区中与本任务有关的唯一新增文件是本说明文档，尚未修改任何现有 Python 或 shell 源码，因此这一阶段不需要源码备份。

策略服务已按本文命令启动。30K 权重完整加载，日志最终给出 `All keys loaded successfully!`；GPU 稳定占用约 9544 MiB。由于加载权重之前产生的 warning 已经提前初始化了 Python logging，服务脚本后续的 INFO 级 `serving` 文本没有显示出来，但本机 8731 端口已经进入监听，并且独立 WebSocket 客户端成功读取到服务元数据：策略类型为 `pi05`，checkpoint 明确指向 `030000/pretrained_model`，动作块形状为 `[50, 18]`。因此策略服务阶段按协议验收通过。

300 步 smoke 已使用 `SEED=10001`、`E2E_COLLECTION_PROFILE=sdk_recovery_v1`、`ROLLOUT_REPLAN=8` 完成，墙钟时间约 24 秒，进程退出码为 0。rollout 成功连接持续运行的 30K 策略服务，日志没有出现 traceback、NaN、动作形状错误、状态形状错误、相机键错误或 WebSocket 断连。`result.json` 明确记录三路相机为 `stereo_left`、`waist_front`、`chassis_front`。短测没有抓起或放置物料，`success=false`，这符合短测只验链路、不评价完整任务成功率的定位。

短测生成的 `e2e_3rd.mp4` 为有效 H.264 文件，大小 76550 bytes，分辨率 960×544，20 FPS，共 100 帧，视频时长 5 秒；首、中、末帧均可正确解码并显示完整场景。视频和结果分别位于 `cruzr_mujoco_sim/out/rollout/pi05_30k_smoke_seed10001/e2e_3rd.mp4` 与同目录的 `result.json`，运行日志位于 `cruzr_mujoco_sim/out/logs/rollout/pi05_30k_smoke_seed10001.log`。视频编码器把原始 960×540 画面自动补到 960×544 以满足 H.264 宏块兼容性，这只是四行像素的编码尺寸调整，不影响策略输入，因为策略三路 224×224 图像并不经过该视频 writer。

7200 步完整 rollout 已使用同一 `SEED=10001` 自然运行到终点，进程退出码为 0，未出现推理、WebSocket、相机、状态、动作或视频编码错误。输出 MP4 大小为 1083667 bytes，是有效 H.264 文件，分辨率 960×544、20 FPS、2400 帧，播放时长为 120 秒。脚本的动作循环日志覆盖 0 到 230 秒，并在 7200 步结束，因此 120 秒视频是在 20 FPS 下记录每三个动作步一帧的结果，播放时间约为脚本标称 240 秒仿真进度的一半；这不表示 rollout 提前终止。

终局 `result.json` 判定 `success=false`，立柱和条料的 `grasped`、`placed` 全部为 false。日志显示底盘从初始 `(x=0.08, y=0.02, yaw=0.07)` 在前 30 秒左右主要原地转到约 `yaw=-0.94`，此后直到结束都在该朝向附近轻微振荡，最终平移仅约数厘米。两件物料的终局坐标与初始坐标在毫米精度上相同。对视频首帧、四分之一、中点、四分之三和末帧的视觉抽查也确认机器人没有形成有效导航或抓取行为。因此这是一次推理和录制链路完整、但模型行为失败的有效评测结果，不能把失败归因于 MP4 损坏或服务断连。

完整视频位于 `cruzr_mujoco_sim/out/rollout/pi05_30k_full_seed10001/e2e_3rd.mp4`，终局结果位于同目录的 `result.json`，日志位于 `cruzr_mujoco_sim/out/logs/rollout/pi05_30k_full_seed10001.log`。在这一执行阶段策略服务保持运行以供后续诊断；全部 checkpoint 对照完成后，服务已正常停止并释放 GPU。

当前执行状态如下：策略服务、300 步 smoke 和首个 7200 步完整 episode 均已完成技术验收。由于首个完整 episode 表现为系统性原地转向，下一步先检查真实首批预测动作、state/action 顺序、归一化统计和训练分布，不立即批量运行更多 seed。到目前为止仍未修改任何现有 Python 或 shell 源码，因此没有触发源码备份流程。

针对原地转向失败已经完成第一轮输入与动作诊断。`SEED=10001` 的真实初始 state18 与训练数据 episode 0 的初始 state18 几乎逐元素一致，手臂下垂姿态并不是训练分布之外的状态；state/action 的左右臂、夹爪和底盘通道顺序也与数据集元信息一致。失败 rollout 第一次请求得到的 50 步动作块中，底盘角速度从约 `-0.007` 逐步下降到约 `-0.277 rad/s`，前 8 步和完整动作块的平均角速度分别约为 `-0.131` 与持续负值，因此视频最开始向负方向旋转是模型真实输出，不是 `apply_action()` 符号翻转造成的。

三路策略画面也已经与训练视频做了直接实图对照。rollout 的 `stereo_left`、`waist_front`、`chassis_front` 顺序、上下方向和画面内容与训练数据一致，正式专家采集脚本本身也把 `REC_WH` 覆盖为 224×224，所以这里不存在训练先用 640×480、推理直接用 224×224造成的新视场差异。三路相机在起始姿态下主要看到地面和机器人结构，任务物体不可见；沿训练 episode 0 抽查第 0、300、900、1500 和 1953 帧后，可以看到机器人接近料车时相机逐渐获得货架与机械臂画面。因此相机并非全程失效，但起始导航阶段的视觉信息很弱，模型必须依赖有限的场景纹理和训练布局规律决定转向。

为了区分 checkpoint 损坏与泛化失败，已经把训练数据 episode 0 的真实三路首帧、真实 state18 和相同语言指令原样送入正在运行的 30K 策略服务，并连续独立采样五次。训练真值 50 步的平均底盘角速度为 `+0.144 rad/s`；模型五次采样的平均角速度均为正，约在 `+0.136` 到 `+0.176 rad/s` 之间。预测动作与对应 50 步训练真值在全部 18 个通道上的平均绝对误差约为 `0.00177` 到 `0.00334`，底盘两通道误差约为 `0.00354` 到 `0.01662`。这证明 checkpoint 能拟合已见训练观测，保存的预处理器、后处理器和动作语义没有整体损坏；同一个模型在新 seed 图像上输出相反方向，当前证据更支持视觉/布局泛化不足或闭环分布漂移。

正式数据的 episode 元数据进一步确认，episode 0 来自 `source_seed=2`、`clean`、`random` 的训练源，跨度为源帧 `[0, 1954]`。使用当前 rollout 代码重建 `SEED=2` 后，三路初始画面与训练压缩视频的平均像素绝对差分别约为 9.38、8.96 和 3.04 灰度级；在这个重建画面上做三次策略采样，完整 50 步的平均角速度仍全部为正，约为 `+0.069` 到 `+0.095 rad/s`。下一项工作因此改为先运行训练内 seed 2 的闭环短测和完整对照：若训练内场景也在后续阶段崩溃，说明行为克隆存在明显闭环累积误差；若训练内场景能够完成而新 seed 失败，则应把主要结论定为数据多样性和泛化不足，而不是继续修改已经对齐的推理相机接口。

以上诊断只读取 checkpoint、数据集与现有代码，并在 `/tmp` 生成临时对照图；仓库内仍只更新了本文档，没有修改任何现有 Python 或 shell 源码，也没有产生需要回滚的代码改动，因此源码备份流程仍未触发。

训练内 `SEED=2` 随后完成了 300 步和 900 步闭环探针。使用原先文档中的 `ROLLOUT_REPLAN=8` 时，底盘在 10 秒仅转到约 `yaw=+0.26`，20 秒转到约 `+0.81`，之后趋于停滞；延长探针时，70–80 秒仍停留在 `yaw≈+0.91`、平移不足 7 cm，因此已主动中止无信息增益的 7200 步运行并保留部分视频。原始 seed2 专家文件仍保存在项目中，它证明示范在 10 秒已经到达约 `(-0.046,+0.096,+0.760)`，20 秒已经回转到 `yaw≈+0.013`，40–50 秒则开始向料车方向快速前进。当前策略只进入了第一段转向，未完成后续状态切换。

对训练 episode 0 的八个真实关键观测又做了逐点开环回放。在 0、10、20、30、40、45、50、60 秒分别送入真实 state18 和真实三路图像时，模型在 10 秒正确给出约 `-0.42～-0.44 rad/s` 回转，在 30–40 秒正确给出约 `-0.20 rad/s`，在 45–50 秒正确给出约 `+0.20 m/s` 前进，在 60 秒也正确给出约 `-0.03 m/s` 短退。除 20 秒附近的长静止片段外，多数关键点的 50 步全通道平均绝对误差小于 `0.002`。因此模型在已见图像上记住了完整阶段序列，但闭环产生少量位姿和像素偏差后，不能稳定到达下一个已学状态。

checkpoint 配置明确保存 `chunk_size=50` 和 `n_action_steps=50`。原 rollout 使用 `ROLLOUT_REPLAN=8`，每次只执行动作块前 8 步便丢弃剩余 42 步；而训练起始动作包含从静止逐渐加速的过程，频繁重规划会让策略反复回到动作块起点。A/B 测试把执行窗口改为 50 后，seed2 在 10 秒达到 `yaw=+0.76`，与专家轨迹基本一致，证实 8 步窗口确实会显著放大停滞。不过 seed2 到 20 秒仍停在约 `yaw=+0.87`，说明动作块长度修正只能解决第一层时序问题，无法消除后续闭环分布漂移。

同样的 `ROLLOUT_REPLAN=50` 已在新场景 `SEED=10001` 上运行 1800 步。10 秒底盘转到约 `yaw=-0.86`，之后 20–50 秒主要在 `-0.65～-0.96` 之间摆动，总平移约 5 cm；立柱和条料均未移动，`grasped` 与 `placed` 全为 false。这与 8 步窗口的失败方向相同，说明 50 步设置更忠实于 checkpoint，但不足以使当前 30K 模型在新场景完成导航。

还使用训练时的低层 EGL `EpisodeRecorder` 与 rollout 的高层 `mujoco.Renderer` 在同一 seed2 场景上做了像素对照，二者平均只差约 0.5–0.9 灰度级，排除了渲染 API 是主要根因。当前重建首帧与原始训练 JPEG 的差异在三路相机上约为 2.2–8.8 灰度级，而原始 JPEG 到训练 H.264 解码帧仅约 1.1–1.8；扫描 0–200 个光照随机数偏移也无法还原训练首帧。这说明训练源与当前场景代码之间还存在未被元数据完整记录的历史渲染版本差异。现阶段仍未修改任何 Python 或 shell 源码；接下来用较正确的 50 步执行窗口完成一次正式长录制，再决定是否需要备份并做最小代码改动。

使用 `ROLLOUT_REPLAN=50` 的正式 7200 步运行现已自然结束，进程退出码为 0，策略服务、WebSocket、状态、动作、相机与编码链路均无异常。最终 MP4 为 H.264，大小 1745636 bytes，分辨率 960×544、20 FPS、2400 帧、播放时长 120 秒；对应 240 秒仿真过程按每三个动作步录一帧，因此视频仍是约二倍速展示。视频位于 `cruzr_mujoco_sim/out/rollout/pi05_30k_full_seed10001_replan50/e2e_3rd.mp4`，结果位于同目录 `result.json`，日志位于 `cruzr_mujoco_sim/out/logs/rollout/pi05_30k_full_seed10001_replan50.log`。

这次较正确动作窗口下的最终结果仍为 `success=false`，立柱与条料的 `grasped`、`placed` 全为 false，终局物体坐标与初始坐标一致。底盘在 240 秒内只从约 `(0.08,0.02)` 漂移到约 `(0.04,-0.12)`，朝向在大范围负角度之间反复摆动，未接近任一物体。对首帧、四分之一、中点、四分之三和末帧的视觉检查确认，机器人没有形成导航、抓取或运输行为。与 8 步完整结果相比，50 步执行把小幅固定振荡变成更大幅随机转向，但没有改善任务完成度。

因此，当前 30K checkpoint 已经完成了从权重加载、真实 MuJoCo 闭环、两种动作执行窗口到完整 MP4 的最终验证。可以确定的是：录制代码和推理链路可用，视频文件有效，checkpoint 也能在训练帧上高度拟合动作；但该 checkpoint 在闭环中无法稳定跨越导航阶段，即使对训练内 seed 也会因轻微观测偏差停滞，对新 seed 更无法完成任务。继续修改 WebSocket、动作符号、相机顺序或视频 writer 没有证据基础，也不应把失败包装成录制问题。

文档前面的推荐命令已经把 `ROLLOUT_REPLAN` 从 8 改为 50，使未来复现与 checkpoint 保存的 `n_action_steps=50` 一致；实际执行记录仍保留最初 8 步实验，便于理解 A/B 差异。

中间 checkpoint 筛选也已经完成。为避免场景随机性干扰，10K、20K、30K 三个服务都使用完全相同的训练 episode 0 首帧、相同的 seed10001 首帧、相同 state18 和语言指令，各独立采样八次。30K 在训练首帧上的全通道误差最低，约为 `0.0018～0.0043`，20K 约为 `0.0024～0.0067`，10K 约为 `0.0046～0.0151`，说明训练继续推进确实提高了已见数据拟合。可是在 seed10001 新首帧上，三者的 50 步平均角速度八次全部为负：30K 约 `-0.102～-0.138`，20K 约 `-0.103～-0.156`，10K 约 `-0.049～-0.155 rad/s`。早期权重没有显示出方向或稳定性优势，因此没有浪费时间再为 10K/20K 录制长视频。

综合所有证据，当前问题不是单纯的 30K 后期过拟合，也没有一个现成中间 checkpoint 可以直接替代。更合理的后续训练工作是：固定并记录与数据采集一致的场景/渲染版本；改善起始阶段的可观测性，因为三路相机初始看不到任务物体；增加真正由策略偏离后再纠正的闭环 recovery 数据，而不是只靠专家轨迹切片；避免同一长任务在缺少阶段信息时产生难以辨认的状态切换；并在正式长训前先用未见 seed 的短闭环成功门槛筛选 canary。是否把底盘位姿、阶段子指令或更合适的前视相机加入策略观测，会改变真实部署契约，不能在本轮推理工作中擅自决定。

截至这里没有修改任何现有 Python 或 shell 源码，因此没有需要备份的代码文件。所有新增内容仅为本文档、被 `.gitignore` 管理的 rollout 视频/JSON/日志和 `/tmp` 诊断图。用户要求的备份约束一直得到遵守：如果下一阶段明确选择修改 rollout 诊断输出或重新设计数据生成代码，应先建立带 UTC 时间戳和校验值的源码备份，再应用最小补丁并运行回归测试。

## 官方推理入口等价性审计

为了进一步排除项目自定义策略服务与 LeRobot 官方动作选择入口之间存在隐藏差异，2026-08-20 又完成了一次固定输入、固定随机数的确定性等价性审计。审计没有使用全零假观测，而是从正式数据集 `formal300_v24_lerobot_v30_20260817` 中读取 episode 0、frame 0 的真实 state18 和三路 H.264 首帧，语言指令也使用训练时的完整英文任务文本。输入快照、每个数组的 SHA256、数据集帧位置和相机来源均已保存。模型使用 `030000/pretrained_model`，随机种子固定为 `20260820`；checkpoint 配置仍为 `chunk_size=50`、`n_action_steps=50` 和 `num_inference_steps=10`。

审计在同一个已成功严格加载的 `PI05Policy` 实例上依次执行三条路径。第一条是当前项目实际使用的 `LeRobotPolicyAdapter.infer()`；第二条绕过 adapter 的 `infer()` 方法，直接执行 checkpoint preprocessor、`PI05Policy.predict_action_chunk()` 和整块 postprocessor；第三条按照 LeRobot 官方同步控制语义，每个控制 tick 执行 preprocessor、`PI05Policy.select_action()` 和单步 postprocessor，连续取满内部队列中的 50 个动作。每条路径开始前都重置 policy、preprocessor 和 postprocessor，并重新设置完全相同的 Python、NumPy、PyTorch 和 CUDA 随机种子，因此比较的是同一次流采样条件下的纯实现差异。

结果为严格通过。三条路径都返回形状 `(50, 18)` 的有限 `float32` 动作；当前 adapter 对直接 `predict_action_chunk()`、当前 adapter 对 50 次 `select_action()`、以及直接 chunk 对 `select_action()` 的逐元素最大绝对误差、平均绝对误差和 RMS 误差全部为 `0.0`，三份输出也通过 `numpy.array_equal`。三份动作数组的 SHA256 完全相同，均为 `449629f67c84196283e454391ce8b747fb5eab7e8ec284519c949716deee6bf0`。这同时证明，对于当前 checkpoint 保存的“分位数反归一化 + CPU device”后处理链，把整个三维动作块一次性送入 postprocessor 与官方异步服务逐动作后处理在数值上完全一致。

这项审计的结论是：当前自定义 WebSocket adapter 没有改变 π0.5 的预测数值，直接改用 LeRobot `select_action()` 也不会改善已经观察到的闭环行为。它只证明当前软件版本、当前 checkpoint 和这份固定真实观测下的推理入口等价，并不证明 checkpoint 在未见场景中具有闭环泛化能力，也没有比较经过格式转换后的 OpenPI checkpoint 或新版本 RTC 的实时调度效果。结合前面的训练帧拟合、训练内 seed 闭环和新 seed 长视频证据，可以更有把握地把后续工作集中在训练数据、策略偏离后的 recovery、任务阶段信息、初始视觉可观测性和历史场景渲染一致性，而不是继续替换等价的推理包装代码。

完整机器可读结果位于 `cruzr_mujoco_sim/out/diagnostics/pi05_30k_official_equivalence_20260820/equivalence_result.json`；固定输入位于同目录的 `fixed_observation.npz` 和 `input_manifest.json`；三份原始动作位于 `action_outputs.npz`；完整控制台输出位于 `equivalence.log`。这些诊断产物由 `.gitignore` 管理，不会提交权重、数据帧或机器私有路径。本轮仍未修改任何 Python 或 shell 源码，因此没有触发源码备份；唯一的受版本控制改动是本说明文档新增了审计记录。
