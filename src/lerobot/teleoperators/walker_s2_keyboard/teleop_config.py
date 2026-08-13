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

"""Walker S2 键盘遥操作配置模块。

本模块定义了 Walker S2 键盘遥操作器的配置类，用于 LeRobot 0.5.1 框架。
包含按键映射、速度级别、动作空间等配置。
"""

from dataclasses import dataclass, field
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("walker_s2_keyboard")
@dataclass
class WalkerS2KeyboardTeleopConfig(TeleoperatorConfig):
    """Walker S2 键盘遥操作配置类。

    继承自 TeleoperatorConfig，用于配置键盘遥操作器的各项参数。
    支持双臂控制，包含按键映射、速度等级、动作空间等配置。

    Attributes:
        type: 遥操作器类型标识符，固定为 "walker_s2_keyboard"
        id: 遥操作器唯一标识符，默认为 "walker_s2_teleop"
        calibration_dir: 校准文件存储目录路径
        speed_levels: 速度级别列表，定义不同的运动速度
        default_speed_index: 默认速度级别索引
        initial_control_arm: 初始控制的机械臂（"left" 或 "right"）
        motion_axes: 运动轴元组，定义六个自由度
        bimanual_mirror_signs: 双臂镜像符号元组，用于镜像控制
        toggle_arm_key: 切换控制臂的按键
        toggle_bimanual_key: 切换双臂模式的按键
        keymap: 按键映射字典，将键盘按键映射到动作
        pressed_keys_template: 按键状态模板字典
        action_shape: 动作空间形状
        action_names: 动作名称字典，映射动作名到索引
    """

    # 必填字段：遥操作器类型标识符
    type: str = "walker_s2_keyboard"

    # 必填字段：遥操作器唯一标识符
    id: str | None = "walker_s2_teleop"

    # 必填字段：校准文件存储目录
    calibration_dir: Path | None = None

    # 速度级别列表，与 mobile_manipulator 默认行为保持一致
    speed_levels: list[float] = field(default_factory=lambda: [0.010, 0.035])

    # 默认速度级别索引（1 表示使用较快速度）
    default_speed_index: int = 0

    # 初始控制的机械臂
    initial_control_arm: str = "left"

    # 运动轴：x, y, z 平移和 rx, ry, rz 旋转
    motion_axes: tuple[str, ...] = ("x", "y", "z", "rx", "ry", "rz")

    # 双臂镜像符号：用于镜像右侧机械臂到左侧
    bimanual_mirror_signs: tuple[float, ...] = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
    # bimanual_mirror_signs: tuple[float, ...] = (-1.0, 1.0, 1.0, 1.0, 1.0, -1.0) # FIXME: 需要根据实际测试调整

    # 切换控制臂的按键
    toggle_arm_key: str = "o"

    # 切换双臂模式的按键（0 键）
    toggle_bimanual_key: str = "0"

    # 退出和速度调整按键
    quit_key: str = "q"
    speed_up_key: str = "+"
    speed_down_key: str = "-"

    # 回到初始位置按键
    go_home_key: str = "h"

    # evdev 输入设备路径；为空时自动扫描所有候选键盘设备
    evdev_device_path: str | None = None

    # 按键映射字典：将数字键和字母键映射到具体动作
    keymap: dict[str, str] = field(
        default_factory=lambda: {
            "1": "x_up",
            "3": "x_down",
            "4": "y_up",
            "6": "y_down",
            "7": "z_up",
            "9": "z_down",
            "y": "rx_up",
            "u": "rx_down",
            "v": "ry_up",
            "b": "ry_down",
            "n": "rz_up",
            "m": "rz_down",
            "k": "gripper_open",
            "l": "gripper_close",
        }
    )

    # 按键状态模板字典：用于跟踪按键按下状态
    pressed_keys_template: dict[str, bool] = field(
        default_factory=lambda: {
            "x_up": False,
            "x_down": False,
            "y_up": False,
            "y_down": False,
            "z_up": False,
            "z_down": False,
            "rx_up": False,
            "rx_down": False,
            "ry_up": False,
            "ry_down": False,
            "rz_up": False,
            "rz_down": False,
            "gripper_open": False,
            "gripper_close": False,
            "quit": False,
            "go_home": False,
        }
    )

    # 动作空间形状（14 维：左右臂各 6 维 + 左右夹爪各 1 维）
    action_shape: tuple[int, ...] = (14,)

    # 动作名称字典：映射动作名称到索引位置
    action_names: dict[str, int] = field(
        default_factory=lambda: {
            "left_delta_x": 0,
            "left_delta_y": 1,
            "left_delta_z": 2,
            "left_delta_rx": 3,
            "left_delta_ry": 4,
            "left_delta_rz": 5,
            "right_delta_x": 6,
            "right_delta_y": 7,
            "right_delta_z": 8,
            "right_delta_rx": 9,
            "right_delta_ry": 10,
            "right_delta_rz": 11,
            "left_gripper": 12,
            "right_gripper": 13,
        }
    )
