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
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

Example teleoperation with walker_s2_sim and task:

```shell
lerobot-teleoperate \
    --robot.type=walker_s2_sim \
    --robot.headless=false \
    --teleop.type=walker_s2_keyboard \
    --task=packing_box \
    --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat

import rerun as rr

from src.lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from src.lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from src.lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from src.lerobot.configs import parser
from src.lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from src.lerobot.robots import (  # noqa: F401
    unitree_g1 as unitree_g1_robot,
)
from src.lerobot.teleoperators import (  # noqa: F401
    unitree_g1,
)
from src.lerobot.utils.control_utils import (
    init_keyboard_listener,
)   
from src.lerobot.utils.import_utils import register_third_party_plugins
from src.lerobot.utils.robot_utils import precise_sleep
from src.lerobot.utils.utils import init_logging, move_cursor_up, log_say
from src.lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from src.lerobot.robots import Robot, RobotConfig, bi_openarm_follower, bi_so_follower, earthrover_mini_plus, hope_jr, koch_follower, make_robot_from_config, omx_follower, openarm_follower, reachy2, so_follower
from src.lerobot.teleoperators import Teleoperator, TeleoperatorConfig, bi_openarm_leader, bi_so_leader, gamepad, homunculus, keyboard, koch_leader, make_teleoperator_from_config, omx_leader, openarm_leader, openarm_mini, reachy2_teleoperator, so_leader


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
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


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        obs = robot.get_observation()

        if robot.name == "unitree_g1":
            teleop.send_feedback(obs)

        # Get teleop action
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))


        if robot.name == "walkerS2":
            robot_action_to_send = None
        # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
        _ = robot.send_action(robot_action_to_send)

        # 遥操作中实时打印双臂末端姿态（每30帧打印一次，约0.5s间隔）
        if robot.name == "walkerS2" and hasattr(robot, "log_ee_poses"):
            loop_count = getattr(teleop_loop, "_ee_log_counter", 0) + 1
            teleop_loop._ee_log_counter = loop_count  # type: ignore[attr-defined]
            if loop_count % 30 == 0:
                robot.log_ee_poses()

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()
    if (robot.name == "walkerS2" and teleop is not None):
        robot.attach_teleop(teleop)
        logging.info("Attached teleop to walker_s2_sim for callback-driven keyboard control")
    listener, events = init_keyboard_listener()
    log_flag = True
    while not events["start_record"]:
        robot.step(render=True) 
        if log_flag:
            log_say("调整好按Enter键开始控制...", play_sounds=False)
            log_flag = False
    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()
        if listener:
            listener.stop()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
