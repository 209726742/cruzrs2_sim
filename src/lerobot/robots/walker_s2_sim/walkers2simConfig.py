#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Walker S2 仿真机器人配置模块。

本模块定义了 Walker S2 双臂机器人在 Isaac Sim 仿真环境中的配置类。
该机器人通过 RobotArticulation API 在 Isaac Sim 中控制，而非真实电机总线。
具有 14 自由度（每臂 7 关节）和 4 个相机。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

from src.lerobot.cameras.configs import CameraConfig
from src.lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from src.lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("walker_s2_sim")
@dataclass
class WalkerS2Config(RobotConfig):
    """Walker S2 双臂机器人在 Isaac Sim 仿真环境中的配置类。

    该机器人通过 RobotArticulation API 在 Isaac Sim 中控制，而非真实电机总线。
    具有 14 自由度（每臂 7 关节）和 4 个相机。
    """

    # 必填字段：机器人类型标识符
    type: str = "walker_s2_sim"

    # 必填字段：机器人唯一标识符
    id: str | None = "walker_s2_default"

    # 必填字段：校准文件存储目录
    calibration_dir: Path | None = None

    # 任务配置映射
    TASK_CONFIGS: dict = field(
        default_factory=lambda: {
            "Part_Sorting": {
                "config_path": "config/Part_Sorting.yaml",
                "desc": "Part_Sorting",
            },
            "Conveyor_Sorting": {
                "config_path": "config/Conveyor_Sorting.yaml",
                "desc": "Conveyor_Sorting",
            },
            "Foam_Inlaying": {
                "config_path": "config/Foam_Inlaying.yaml",
                "desc": "Foam_Inlaying",
            },
            "Packing_Box": {
                "config_path": "config/Packing_Box.yaml",
                "desc": "Packing_Box",
            },
        }
    )

    # Isaac Sim 特定设置
    task_name: str = "Foam_Inlaying"  # 默认任务名称
    task_cfg: dict = field(default_factory=dict)
    root_path: str = "Ubtech_sim"
    task_cfg_path: str = str(Path("Ubtech_sim") / "config" / "Packing_Box.yaml")

    # 机器人配置
    prim_path: str = "/Root/Ref_Xform/Ref"  
    robot_name: str = "walkerS2"
    urdf_path: str = str(Path("assets") / "resources" / "s2.urdf")

    # Isaac Sim 配置
    headless: bool = False
    sim_width: int = 1280
    sim_height: int = 720
    physics_dt: float = 1.0 / 200.0
    rendering_dt: float = 1.0 / 20.0

    # 控制设置
    # 键盘控制的速度级别 (rad/step)
    speed_levels: list[float] = field(default_factory=lambda: [0.010, 0.035, 0.15])
    default_speed_index: int = 1  
    tracking_interp_steps: int = 100


    teleop_time_s: Optional[float] = None  # None 表示无限时长

    # ROS2 / Pico4 遥操作（可选，需 rclpy；未安装时自动跳过）
    enable_ros2_teleop: bool = True
    ros2_joint_commands_topic: str = "/isaac/joint_position_commands"

    # 本地 OpenCV 多相机可视化
    head_viz_enabled: bool = True
    head_viz_window_name: str = "walker_s2_cameras"
    head_viz_scale: float = 1.0
    head_viz_every_n: int = 10
    head_viz_window_x: int = 40
    head_viz_window_y: int = 40
    head_viz_show_labels: bool = True

    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # dummy camera - 添加两个额外视角方便观察
    # 位置和朝向可以根据实际场景调整
    dummy_cameras_cfg: dict = field(
        default_factory=lambda: {
            "dummy_camera_top": {
                "translation": [-2.54145, -0.06363, 2.4821],
                "orientation": [0.942732, -0.008441, 0.333388, 0.006151],
                "prim_path": "//Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/dummy_camera_top",
            },
            "dummy_camera_side": {
                "translation": [2.06555, -0.02631, 0.95453],
                "orientation": [0.942732, -0.008441, 0.333388, 0.006151],
                "prim_path": "/Replicator/Ref_Xform/Ref/dummy_camera_side",
            },
        }
    )

    # Walker S2 有 4 个相机：head_left, head_right, wrist_left, wrist_right
    # 在仿真中，我们使用 OpenCV 相机
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "head_left": OpenCVCameraConfig(
                index_or_path=0,  
                fps=30,
                width=640,
                height=480,
            ),
            "head_right": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist_left": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist_right": OpenCVCameraConfig(
                index_or_path=3,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # 手臂关节配置（共 14 个关节：7 左 + 7 右）
    # 这些是 Isaac Sim 中期望的关节名称
    left_arm_joint_names: list[str] = field(
        default_factory=lambda: [
            "L_shoulder_pitch_joint",
            "L_shoulder_roll_joint",
            "L_shoulder_yaw_joint",
            "L_elbow_roll_joint",
            "L_elbow_yaw_joint",
            "L_wrist_pitch_joint",
            "L_wrist_roll_joint",
        ]
    )
    right_arm_joint_names: list[str] = field(
        default_factory=lambda: [
            "R_shoulder_pitch_joint",
            "R_shoulder_roll_joint",
            "R_shoulder_yaw_joint",
            "R_elbow_roll_joint",
            "R_elbow_yaw_joint",
            "R_wrist_pitch_joint",
            "R_wrist_roll_joint",
        ]
    )

    # 键盘控制映射占位符
    teleop_keymap: dict = field(default_factory=dict)
    mock: bool = False

    def load_from_yaml(self, task: Optional[str] = None) -> dict:
        """加载任务 yaml 配置并更新 task_cfg/task_cfg_path。

        Args:
            task: 任务名称，如果不指定或不在 TASK_CONFIGS 中，则使用默认任务 "Foam_Inlaying"

        Returns:
            从 yaml 文件加载的配置字典

        Raises:
            FileNotFoundError: 如果配置文件不存在
        """
        effective_task = task if task in self.TASK_CONFIGS else "Foam_Inlaying"
        task_cfg = self.TASK_CONFIGS[effective_task]

        full_config_path = Path(self.root_path) / task_cfg["config_path"]
        if not full_config_path.is_absolute():
            # 获取项目根目录
            project_root = Path(__file__).parent.parent.parent.parent.parent
            full_config_path = project_root / full_config_path

        config_path = Path(full_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        yaml_dir = config_path.parent
        if "root_path" in cfg:
            cfg["root_path"] = str((yaml_dir / cfg["root_path"]).resolve())

        self.task_cfg_path = str(config_path)
        self.task_cfg = cfg
        self.task_name = effective_task
        return cfg

    def __post_init__(self):
        # 验证关节配置
        if len(self.left_arm_joint_names) != 7 or len(self.right_arm_joint_names) != 7:
            raise ValueError(
                f"Walker S2 robot requires exactly 7 joints per arm. "
                f"Got {len(self.left_arm_joint_names)} left and "
                f"{len(self.right_arm_joint_names)} right joints."
            )

        # 合并所有手臂关节名称供参考
        self.all_arm_joint_names = self.left_arm_joint_names + self.right_arm_joint_names

        # 在仿真环境中，mock 默认为 True（不使用真实硬件）
        if not self.mock:
            # 在仿真中，我们不使用真实电机，所以这实际上是 mock 模式
            # 对于底层设备层，但机器人本身在仿真中是"真实"的
            pass

        # 更新相机名称以匹配 mobile_manipulator.py 中的预期格式
        # 确保相机配置具有正确的名称
        for cam_name, cam_config in self.cameras.items():
            if hasattr(cam_config, "name"):
                cam_config.name = cam_name


# Backward compatibility for historical references.
WalkerS2SimRobotConfig = WalkerS2Config
