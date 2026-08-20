# CRUZR S2 仿真与 π0.5 训练源码

本仓库保存 CRUZR S2 双物料任务线的仿真、数据处理、训练脚本和测试代码，当前主线包括立柱放置与 ECU 装配相关流程。仓库定位是公开的源码仓库，方便团队成员查看代码、复现实验流程和继续开发。

## 仓库内容与外部依赖

仓库中包含：

- `src/`：LeRobot 相关源码、策略、数据集处理和训练入口；
- `cruzr_mujoco_sim/scripts/collection/`：当前双物料数据采集、验收、构建和 rollout 脚本；
- `cruzr_mujoco_sim/scripts/core/`：仿真控制、任务契约、动作质量和对象工具；
- `cruzr_mujoco_sim/scripts/tests/`：当前测试；
- `cruzr_mujoco_sim/scripts/training/`：正式 π0.5 训练包装脚本；
- `pi05_train.sh`：通用 π0.5 训练启动器，包含配置检查、训练、续训和状态查看。

为了避免上传私有资料、超大文件和本地运行产物，以下内容不会进入 GitHub：训练数据集、模型权重、Python/Conda 环境、日志、训练输出、`CruzrS2/` 私有资料、`cruzr_mujoco_sim/assets/` 仿真资源以及视频和数据文件。具体规则见 [`.gitignore`](.gitignore)。因此，克隆本仓库后不能仅凭 GitHub 内容直接运行完整仿真或训练，还需要另外准备仿真资源、LeRobot 数据集、π0.5 基础模型以及相应的 Python/Isaac Sim/openpi 环境。

仿真环境和历史流程说明见 [`cruzr_mujoco_sim/README.md`](cruzr_mujoco_sim/README.md)，脚本目录导航见 [`cruzr_mujoco_sim/scripts/README.md`](cruzr_mujoco_sim/scripts/README.md)。`scripts/archive/` 中的内容是历史代码，不是当前双物料训练链的首选入口。

## 当前推荐训练流程

当前正式训练使用 LeRobot v3.0 数据集和 π0.5 基础策略，建议先运行短 canary 验证硬件、数据、模型和多卡配置，再启动正式训练。正式训练包装脚本是 [`cruzr_mujoco_sim/scripts/training/pi05_formal300_train.sh`](cruzr_mujoco_sim/scripts/training/pi05_formal300_train.sh)。它会检查数据集元数据、GPU、checkpoint 完整性，并支持断点续训。

训练脚本默认值针对当前 4 卡机器：4 个进程、GPU `0,1,2,3`、每张 GPU 的 batch size 为 `1`、每张 GPU 使用 `2` 个 DataLoader worker。正式目标为 `10000` steps，每 `500` steps 保存一个 checkpoint；策略使用 `bfloat16`、开启 gradient checkpointing、默认只训练 expert 参数。换机器时必须根据实际 GPU 数量调整 `GPU_IDS`、`NUM_PROCESSES`、`BATCH_SIZE` 和 `NUM_WORKERS`，不要直接照搬当前服务器路径。

训练前需要准备以下三个外部路径：

1. LeRobot v3.0 数据集目录，至少包含 `meta/info.json` 和 `meta/stats.json`，并且包含 `observation.state`、`action` 以及图像/视频特征；
2. π0.5 基础策略目录，例如包含 `config.json` 和 `model.safetensors` 的 `pretrained_model` 目录；
3. 可执行的 Isaac Sim Python 入口，以及与当前源码匹配的 PyTorch、Transformers、LeRobot 和 CUDA 环境。

先用 canary 做检查和短训练：

```bash
cd /path/to/cruzr_sim

DATASET_ROOT=/path/to/formal300_v24_lerobot_v30 \
BASE_POLICY=/path/to/pi05_base_remapped \
ISAAC_PY=/path/to/isaac-sim/python.sh \
GPU_IDS=0,1,2,3 \
NUM_PROCESSES=4 \
BATCH_SIZE=1 \
NUM_WORKERS=2 \
bash cruzr_mujoco_sim/scripts/training/pi05_formal300_train.sh canary
```

canary 成功后启动正式训练：

```bash
DATASET_ROOT=/path/to/formal300_v24_lerobot_v30 \
BASE_POLICY=/path/to/pi05_base_remapped \
ISAAC_PY=/path/to/isaac-sim/python.sh \
GPU_IDS=0,1,2,3 \
NUM_PROCESSES=4 \
BATCH_SIZE=1 \
NUM_WORKERS=2 \
bash cruzr_mujoco_sim/scripts/training/pi05_formal300_train.sh start
```

服务器重启或训练进程中断后，使用同一套路径恢复：

```bash
DATASET_ROOT=/path/to/formal300_v24_lerobot_v30 \
BASE_POLICY=/path/to/pi05_base_remapped \
ISAAC_PY=/path/to/isaac-sim/python.sh \
GPU_IDS=0,1,2,3 \
NUM_PROCESSES=4 \
NUM_WORKERS=2 \
bash cruzr_mujoco_sim/scripts/training/pi05_formal300_train.sh resume
```

查看任务状态：

```bash
bash cruzr_mujoco_sim/scripts/training/pi05_formal300_train.sh status
```

## 通用训练启动器

如果不是使用固定的 formal300 数据集，使用根目录的 [`pi05_train.sh`](pi05_train.sh)。它支持通过命令行调整数据集、episode、基础模型、输出目录、GPU、训练步数、保存频率、学习率和 scheduler 等参数。先查看完整帮助：

```bash
bash pi05_train.sh help
```

推荐先执行 dry-run：

```bash
bash pi05_train.sh dry-run \
  --dataset-root /path/to/lerobot_v30 \
  --repo-id local/my_dataset \
  --episodes train \
  --base-policy /path/to/pi05_base_remapped \
  --isaac-python /path/to/isaac-sim/python.sh \
  --gpu-ids 0,1,2,3 \
  --num-processes 4 \
  --batch-size 1 \
  --num-workers 2 \
  --steps 10000 \
  --save-freq 500
```

常用参数包括：

| 参数 | 当前默认值 | 作用 |
| --- | ---: | --- |
| `--dataset-root` | 本机默认路径 | LeRobot v3.0 数据集目录 |
| `--episodes` | `train` | 使用 `train`、`val`、`test`、`all` 或自定义 episode |
| `--base-policy` | 本机默认路径 | π0.5 基础模型或 checkpoint 的 `pretrained_model` |
| `--gpu-ids` | `0,1,2,3` | 使用的 GPU 编号 |
| `--num-processes` | `4` | DDP 进程数，通常与 GPU 数量一致 |
| `--batch-size` | `1` | 每个 GPU 的 batch size |
| `--num-workers` | `2` | 每个进程的 DataLoader worker 数量 |
| `--steps` | `10000` | 目标训练步数 |
| `--save-freq` | `500` | checkpoint 保存间隔 |
| `--learning-rate` | `2.5e-5` | 学习率 |
| `--weight-decay` | `0.01` | 权重衰减 |
| `--warmup-steps` | `1000` | scheduler warmup 步数 |
| `--decay-steps` | `30000` | scheduler decay 步数 |
| `--n-action-steps` | `50` | action chunk 长度 |
| `--num-inference-steps` | `10` | 推理采样步数 |

首次换 GPU、数据集或 batch size 时，建议先用较短的 canary 验证显存、数据读取、checkpoint 保存和恢复，再进行长训练。不要仅根据 loss 下降判断任务是否成功，训练后还需要使用仿真 rollout 和未见过的 seed 做闭环评测。

## 训练输出与续训规则

训练输出默认写入 `cruzr_mujoco_sim/out/training/`，日志写入 `log/`；这些目录属于本地产物，不提交到 GitHub。一个完整 checkpoint 应同时包含模型、训练配置、优化器、scheduler 和随机状态。`resume` 会优先读取 checkpoint 中保存的数据集、batch、worker 和 scheduler 配置，避免服务器重启后静默使用错误参数；如果更换了 GPU 编号或进程数，需要在恢复命令中重新提供对应环境变量。

## 代码入口

- 当前双物料采集和数据构建：[`cruzr_mujoco_sim/scripts/collection/`](cruzr_mujoco_sim/scripts/collection/)
- 共享仿真与任务契约：[`cruzr_mujoco_sim/scripts/core/`](cruzr_mujoco_sim/scripts/core/)
- 当前测试：[`cruzr_mujoco_sim/scripts/tests/`](cruzr_mujoco_sim/scripts/tests/)
- π0.5 训练入口：[`src/lerobot/scripts/lerobot_train.py`](src/lerobot/scripts/lerobot_train.py)
- 项目文档总索引：[`docs/README.md`](docs/README.md)
- 当前训练状态和实验记录：[`docs/current/双物料线训练进度.md`](docs/current/双物料线训练进度.md)

运行当前脚本测试：

```bash
cd cruzr_mujoco_sim
python -m unittest discover -s scripts/tests -p 'test_*.py' -v
```

## 重要限制

当前仓库是源码和流程仓库，不保证在没有外部仿真资源、训练数据、基础模型和正确运行环境的机器上直接启动训练。`cruzr_mujoco_sim/scripts/archive/` 中的旧 RL、DAgger 和 ECU 脚本仅供历史参考；开始新实验前，应优先使用本 README 指向的正式训练脚本，并保留 canary、checkpoint 和闭环评测记录。
