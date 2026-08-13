# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replays the actions of an episode from a dataset on a robot.

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=<USER>/record-test \
    --dataset.episode=0
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np

from src.lerobot.configs import parser
from src.lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.lerobot.processor import (
    make_default_robot_action_processor,
)
from src.lerobot.robots import (  # noqa: F401
    unitree_g1,
    walker_s2_sim,
)
from src.lerobot.utils.control_utils import init_keyboard_listener, is_headless
from src.lerobot.utils.constants import ACTION
from src.lerobot.utils.import_utils import register_third_party_plugins
from src.lerobot.utils.robot_utils import precise_sleep
from src.lerobot.utils.utils import (
    init_logging,
    log_say,
)
from src.lerobot.robots import Robot, RobotConfig, bi_openarm_follower, bi_so_follower, earthrover_mini_plus, hope_jr, koch_follower, make_robot_from_config, omx_follower, openarm_follower, reachy2, so_follower


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path'). If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Task name for simulation environments (e.g., "Part_Sorting", "Foam_Inlaying")
    # Only used by walker_s2_sim robot. If not provided, defaults to "Foam_Inlaying"
    task: str | None = None

    def __post_init__(self):
        # Load task configuration for walker_s2_sim robot
        if hasattr(self.robot, "task_name") and self.task:
            self.robot.task_name = self.task
            # Load task configuration from yaml
            if hasattr(self.robot, "load_from_yaml"):
                try:
                    self.robot.load_from_yaml(self.task)
                except FileNotFoundError as e:
                    logging.warning(f"Failed to load task config: {e}")


def _step_robot_if_supported(robot: Robot, *, render: bool) -> None:
    """仅在机器人实现了 step() 时推进一帧仿真。"""
    step_fn = getattr(robot, "step", None)
    if callable(step_fn):
        step_fn(render=render)


def _get_episode_initial_environment_state(episode_frames: Any, robot: Robot | None = None) -> np.ndarray | None:
    """从 episode 首帧的 observation.state 中提取环境状态向量。

    observation.state 结构：
    - 前 20 维：机器人状态（14 臂关节 + 4 手指关节 + 2 夹持器）
    - 后面 N 维：环境物体位姿（每个物体 7 维：x, y, z, qx, qy, qz, qw）

    Args:
        episode_frames: 数据集的帧
        robot: 机器人实例，用于获取 env_state_dim

    Returns:
        环境状态向量（仅物体位姿部分），shape (env_state_dim,)
    """
    if "observation.state" not in episode_frames.column_names:
        return None

    # 获取首帧的 observation.state
    state = episode_frames[0]["observation.state"]
    if state is None:
        return None

    state_np = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_np.size < 20:
        return None

    # 前 20 维是机器人状态，后面的是环境物体位姿
    env_state = state_np[20:]

    if env_state.size == 0:
        return None

    # 如果提供了 robot，验证维度是否匹配
    if robot is not None and hasattr(robot, 'env_state_dim'):
        expected_dim = robot.env_state_dim
        if expected_dim > 0 and env_state.size != expected_dim:
            logging.warning(
                f"[Replay] 环境状态维度不匹配：期望 {expected_dim}, 实际 {env_state.size}。"
                f"数据集可能使用了不同的任务配置。"
            )

    return env_state


def _restore_replay_environment(robot: Robot, episode_frames: Any) -> None:
    """在 replay 前将仿真环境恢复到数据集首帧的物体位姿。"""
    env_state = _get_episode_initial_environment_state(episode_frames, robot)
    if env_state is None:
        logging.info("[Replay] 未找到环境状态数据，跳过环境位姿恢复")
        return
    robot.set_environment_state(env_state)

    # 推进物理仿真让物体稳定
    for _ in range(10):
        robot.step(render=True)
    logging.info("[Replay] 物体初始位姿恢复完成并已稳定")


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    listener, events = init_keyboard_listener()
    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    # Filter dataset to only include frames from the specified episode since episodes are chunked in dataset V3.0
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.episode)
    actions = episode_frames.select_columns(ACTION)

    robot.connect()
    _restore_replay_environment(robot, episode_frames)
    log_flag = True
    while not events["start_record"]:
        robot.step(render=True)
        if log_flag:
            log_say("调整好按 Enter 键开始回放...", cfg.play_sounds)
            log_flag = False

    try:
        log_say("Replaying episode", cfg.play_sounds, blocking=True)
        for idx in range(len(episode_frames)):
            start_episode_t = time.perf_counter()

            action_array = actions[idx][ACTION]
            action = {}
            for i, name in enumerate(dataset.features[ACTION]["names"]):
                action[name] = action_array[i]

            robot_obs = robot.get_observation()

            processed_action = robot_action_processor((action, robot_obs))

            _ = robot.send_action(processed_action)

            dt_s = time.perf_counter() - start_episode_t
            precise_sleep(max(1 / dataset.fps - dt_s, 0.0))
    finally:
        if robot.is_connected:
            robot.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
