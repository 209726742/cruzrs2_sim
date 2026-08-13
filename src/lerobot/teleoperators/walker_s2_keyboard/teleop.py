#!/usr/bin/env python
"""
Walker S2 键盘遥操作 - LeRobot 0.5.1 实现
"""

import collections
import logging
import select
import threading
from typing import Any, Callable, Optional

import numpy as np
import torch

PYNPUT_AVAILABLE = False
keyboard = None
try:
    from pynput import keyboard as pynput_keyboard
    keyboard = pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    pass

from src.lerobot.processor import RobotAction
from src.lerobot.teleoperators.config import TeleoperatorConfig
from src.lerobot.teleoperators.teleoperator import Teleoperator
from src.lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .teleop_config import WalkerS2KeyboardTeleopConfig

logger = logging.getLogger(__name__)

EVDEV_AVAILABLE = False
try:
    import evdev
    EVDEV_AVAILABLE = True
except ImportError:
    pass


class EvdevKeyboardListener:
    """evdev 键盘监听器，适用于 Docker/远程桌面环境"""

    _CODE_TO_CHAR: dict[int, str] = {
        2: "1", 3: "2", 4: "3", 5: "4", 6: "5",
        7: "6", 8: "7", 9: "8", 10: "9", 11: "0",
        12: "-", 13: "=",
        16: "q", 17: "w", 18: "e", 19: "r", 20: "t",
        21: "y", 22: "u", 23: "i", 24: "o", 25: "p",
        30: "a", 31: "s", 32: "d", 33: "f", 34: "g",
        35: "h", 36: "j", 37: "k", 38: "l",
        44: "z", 45: "x", 46: "c", 47: "v", 48: "b",
        49: "n", 50: "m",
        52: ".",
        71: "7", 72: "8", 73: "9", 74: "-",
        75: "4", 76: "5", 77: "6",
        78: "+",
        79: "1", 80: "2", 81: "3", 82: "0", 83: ".",
    }

    def __init__(
        self,
        on_press: Optional[Callable] = None,
        on_release: Optional[Callable] = None,
        device_path: str | None = None,
    ):
        self._on_press = on_press
        self._on_release = on_release
        self._device_path = device_path
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._devices = []

    def start(self) -> None:
        self._stop_event.clear()
        self._devices = self._find_keyboard_devices()
        if not self._devices:
            raise RuntimeError("未找到可用的 evdev 键盘设备")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        for dev in self._devices:
            try:
                dev.close()
            except Exception:
                pass
        self._devices = []

    def _find_keyboard_devices(self):
        """自动寻找所有包含遥操作按键的输入设备。"""
        if self._device_path:
            try:
                dev = evdev.InputDevice(self._device_path)
                logger.info(
                    "evdev 使用指定键盘设备: path=%s name=%s",
                    dev.path,
                    getattr(dev, "name", "unknown"),
                )
                return [dev]
            except Exception as e:
                raise RuntimeError(f"无法打开指定 evdev 设备 {self._device_path}: {e}") from e

        logger.info("开始扫描容器内输入设备，查找可用键盘...")
        scanned_devices = 0
        devices = []
        teleop_key_codes = set(self._CODE_TO_CHAR)
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                scanned_devices += 1
                caps = dev.capabilities(verbose=False)
                key_codes = set(caps.get(1, []))  # EV_KEY = 1
                matched_codes = sorted(key_codes & teleop_key_codes)
                logger.info(
                    "扫描到输入设备: path=%s name=%s phys=%s caps=%s key_count=%s matched_teleop_keys=%s",
                    dev.path,
                    getattr(dev, "name", "unknown"),
                    getattr(dev, "phys", "unknown"),
                    sorted(caps.keys()),
                    len(key_codes),
                    matched_codes,
                )
                if matched_codes:
                    devices.append(dev)
                    logger.info("evdev 监听键盘候选设备: path=%s name=%s", dev.path, getattr(dev, "name", "unknown"))
                else:
                    dev.close()
            except Exception as e:
                logger.debug(f"跳过输入设备 {path}: {e}")
        if not devices:
            logger.warning("evdev 扫描完成，共检查 %s 个设备，未找到可用键盘", scanned_devices)
        else:
            logger.info("evdev 扫描完成，共监听 %s/%s 个键盘候选设备", len(devices), scanned_devices)
        return devices

    def _run(self) -> None:
        """evdev 事件循环"""
        if not self._devices:
            logger.warning("未找到键盘设备，evdev 键盘控制不可用")
            return

        logger.info(
            "evdev 使用设备: %s",
            ", ".join(f"{dev.path} ({dev.name})" for dev in self._devices),
        )

        try:
            while not self._stop_event.is_set():
                active_devices = [dev for dev in self._devices if getattr(dev, "fd", -1) >= 0]
                if not active_devices:
                    logger.warning("evdev 所有键盘设备都已关闭")
                    return

                readable, _, _ = select.select([dev.fd for dev in active_devices], [], [], 0.1)
                if not readable:
                    continue

                readable_fds = set(readable)
                for dev in active_devices:
                    if dev.fd not in readable_fds:
                        continue
                    try:
                        events = dev.read()
                    except OSError as e:
                        logger.warning("evdev 设备读取失败，停止监听该设备: path=%s error=%s", dev.path, e)
                        self._devices.remove(dev)
                        try:
                            dev.close()
                        except Exception:
                            pass
                        continue

                    for event in events:
                        if event.type != 1:  # EV_KEY
                            continue

                        char = self._CODE_TO_CHAR.get(event.code)
                        if char is None:
                            continue

                        key_obj = type(
                            "EvdevKey",
                            (),
                            {"char": char, "name": None},
                        )()

                        if event.value in (1, 2):  # press or key repeat
                            if self._on_press:
                                self._on_press(key_obj)
                        elif event.value == 0:  # release
                            if self._on_release:
                                self._on_release(key_obj)

        except Exception as e:
            logger.warning(f"evdev 监听错误：{e}")
        finally:
            for dev in self._devices:
                try:
                    dev.close()
                except Exception:
                    pass
            self._devices = []


class WalkerS2KeyboardTeleop(Teleoperator):
    """Walker S2 键盘遥操作实现"""

    name: str = "walker_s2_keyboard"
    config_class: type[TeleoperatorConfig] = WalkerS2KeyboardTeleopConfig

    def __init__(self, config: WalkerS2KeyboardTeleopConfig | None = None):
        if config is None:
            config = WalkerS2KeyboardTeleopConfig()

        super().__init__(config)
        self.config = config

        self.current_control_arm: str = config.initial_control_arm
        self.bimanual_control_enabled: bool = False
        self._speed_index: int = config.default_speed_index
        self.speed_levels: list[float] = config.speed_levels

        self._pressed_keys: dict[str, bool] = dict(config.pressed_keys_template)
        self._keyboard_listener: Optional[Any] = None

        self.MOTION_AXES: tuple[str, ...] = config.motion_axes
        self.BIMANUAL_MIRROR_SIGNS: torch.Tensor = torch.tensor(
            config.bimanual_mirror_signs, dtype=torch.float32
        )

        self._keyboard_cmd_queue: collections.deque[dict[str, bool]] = collections.deque(maxlen=256)
        self._current_frame_keys: Optional[dict[str, bool]] = None
        self._last_keyboard_frame_id: int = -1
        self._callback_mode: bool = False

        self._go_home_key_was_pressed: bool = False

    @property
    def is_connected(self) -> bool:
        return self._keyboard_listener is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": self.config.action_shape,
            "names": self.config.action_names,
        }

    @property
    def feedback_features(self) -> dict[str, Any]:
        return {}

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if EVDEV_AVAILABLE:
            try:
                listener = EvdevKeyboardListener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                    device_path=self.config.evdev_device_path,
                )
                listener.start()
                self._keyboard_listener = listener
                logger.info("键盘监听已启动 (evdev)")
                return
            except Exception as e:
                logger.warning(f"evdev 启动失败：{e}，尝试 pynput")

        if PYNPUT_AVAILABLE:
            try:
                self._keyboard_listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                )
                self._keyboard_listener.start()
                logger.info("键盘监听已启动 (pynput)")
                return
            except Exception as e:
                logger.warning(f"pynput 启动失败：{e}")

        logger.warning("未能启动可用的键盘监听后端")

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._keyboard_listener:
            if hasattr(self._keyboard_listener, "stop"):
                self._keyboard_listener.stop()
            self._keyboard_listener = None
            logger.info("键盘监听已断开")

    def _resolve_key_action(self, char: str | None) -> str | None:
        if char is None:
            return None
        return self.config.keymap.get(char.lower())

    def _on_key_press(self, key) -> bool | None:
        char = getattr(key, "char", None)
        char = char.lower() if char is not None else None
        key_name = getattr(key, "name", None)

        go_home_key = str(self.config.go_home_key).lower()

        if key_name == go_home_key or char == go_home_key:
            if not self._go_home_key_was_pressed:
                self._go_home_key_was_pressed = True
                self._pressed_keys["go_home"] = True
                logger.info(f"按键 '{self.config.go_home_key}': 切换到 home 位置")
                self._enqueue_keyboard_snapshot()
            return None

        if char == self.config.toggle_arm_key:
            self.current_control_arm = (
                "right" if self.current_control_arm == "left" else "left"
            )
            logger.info(f"切换控制臂：{self.current_control_arm}")
            self._enqueue_keyboard_snapshot()
            return None

        if char == self.config.toggle_bimanual_key:
            self.bimanual_control_enabled = not self.bimanual_control_enabled
            mode = "双臂同步" if self.bimanual_control_enabled else "单臂"
            logger.info(f"控制模式：{mode}")
            self._enqueue_keyboard_snapshot()
            return None

        if char in (self.config.speed_up_key, "=", "+"):
            self._speed_index = min(self._speed_index + 1, len(self.speed_levels) - 1)
            logger.info(f"速度：{self.speed_levels[self._speed_index]:.3f}")
            self._enqueue_keyboard_snapshot()
            return None

        if char == self.config.speed_down_key:
            self._speed_index = max(self._speed_index - 1, 0)
            logger.info(f"速度：{self.speed_levels[self._speed_index]:.3f}")
            self._enqueue_keyboard_snapshot()
            return None

        if char == self.config.quit_key:
            self._pressed_keys["quit"] = True
            self._enqueue_keyboard_snapshot()
            logger.info("退出键被按下")
            return False

        action_name = self._resolve_key_action(char)
        if action_name:
            self._pressed_keys[action_name] = True
            self._enqueue_keyboard_snapshot()

        return None

    def _on_key_release(self, key) -> None:
        char = getattr(key, "char", None)
        char = char.lower() if char is not None else None
        key_name = getattr(key, "name", None)

        go_home_key = str(self.config.go_home_key).lower()

        if key_name == go_home_key or char == go_home_key:
            self._go_home_key_was_pressed = False
            self._pressed_keys["go_home"] = False
            self._enqueue_keyboard_snapshot()
            return

        action_name = self._resolve_key_action(char)
        if action_name:
            self._pressed_keys[action_name] = False
            self._enqueue_keyboard_snapshot()

    def _enqueue_keyboard_snapshot(self) -> None:
        self._keyboard_cmd_queue.append(dict(self._pressed_keys))

    def enable_callback_mode(self) -> None:
        self._callback_mode = True
        self._keyboard_cmd_queue.clear()
        self._last_keyboard_frame_id = -1
        self._current_frame_keys = None

    def disable_callback_mode(self) -> None:
        self._callback_mode = False

    def get_action(self) -> RobotAction:
        step = float(self.speed_levels[self._speed_index])

        active_delta = torch.zeros(len(self.MOTION_AXES), dtype=torch.float32)
        active_gripper = 0.0

        for index, axis in enumerate(self.MOTION_AXES):
            if self._pressed_keys.get(f"{axis}_up"):
                active_delta[index] += step
            if self._pressed_keys.get(f"{axis}_down"):
                active_delta[index] -= step

        if self._pressed_keys.get("gripper_open"):
            active_gripper = 1.0
        if self._pressed_keys.get("gripper_close"):
            active_gripper = -1.0

        # # URDF sixforce_link 的 yaw=-π/2，将世界坐标 delta 转换到 EE frame（X↔Y 交换）FIXME
        # active_delta[[0, 1]] = active_delta[[1, 0]]
        # active_delta[[3, 4]] = active_delta[[4, 3]]

        left_delta = torch.zeros(len(self.MOTION_AXES), dtype=torch.float32)
        right_delta = torch.zeros(len(self.MOTION_AXES), dtype=torch.float32)
        left_gripper = torch.tensor([0.0], dtype=torch.float32)
        right_gripper = torch.tensor([0.0], dtype=torch.float32)

        if self.current_control_arm == "left":
            left_delta = active_delta.clone()
            left_gripper[0] = active_gripper
            if self.bimanual_control_enabled:
                right_delta = active_delta * self.BIMANUAL_MIRROR_SIGNS
                right_gripper[0] = active_gripper
        else:
            right_delta = active_delta.clone()
            right_gripper[0] = active_gripper
            if self.bimanual_control_enabled:
                left_delta = active_delta * self.BIMANUAL_MIRROR_SIGNS
                left_gripper[0] = active_gripper

        action: RobotAction = {
            "left_delta_x": left_delta[0].item(),
            "left_delta_y": left_delta[1].item(),
            "left_delta_z": left_delta[2].item(),
            "left_delta_rx": left_delta[3].item(),
            "left_delta_ry": left_delta[4].item(),
            "left_delta_rz": left_delta[5].item(),
            "right_delta_x": right_delta[0].item(),
            "right_delta_y": right_delta[1].item(),
            "right_delta_z": right_delta[2].item(),
            "right_delta_rx": right_delta[3].item(),
            "right_delta_ry": right_delta[4].item(),
            "right_delta_rz": right_delta[5].item(),
            "left_gripper": left_gripper[0].item(),
            "right_gripper": right_gripper[0].item(),
        }

        return action

    def get_action_numpy(self, frame_id: int = 0) -> tuple[np.ndarray, np.ndarray, float, float]:
        if frame_id != self._last_keyboard_frame_id:
            self._last_keyboard_frame_id = frame_id

            key_snapshot = {}
            while self._keyboard_cmd_queue:
                snap = self._keyboard_cmd_queue.popleft()
                for k, v in snap.items():
                    if v:
                        key_snapshot[k] = True

            for k, v in self._pressed_keys.items():
                if v:
                    key_snapshot[k] = True

            self._current_frame_keys = key_snapshot
        else:
            key_snapshot = self._current_frame_keys or {}

        step = self.speed_levels[self._speed_index]
        active_delta = np.zeros(6, dtype=np.float32)
        active_gripper = 0.0

        for index, axis in enumerate(self.MOTION_AXES):
            if key_snapshot.get(f"{axis}_up"):
                active_delta[index] += step
            if key_snapshot.get(f"{axis}_down"):
                active_delta[index] -= step

        if key_snapshot.get("gripper_open"):
            active_gripper = 1.0
        if key_snapshot.get("gripper_close"):
            active_gripper = -1.0

        # URDF sixforce_link 的 yaw=-π/2，将世界坐标 delta 转换到 EE frame（X↔Y 交换）
        # active_delta[[0, 1]] = active_delta[[1, 0]]
        # active_delta[[3, 4]] = active_delta[[4, 3]]
        mirror_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        left_delta = np.zeros(6, dtype=np.float32)
        right_delta = np.zeros(6, dtype=np.float32)
        left_gripper = 0.0
        right_gripper = 0.0

        if self.current_control_arm == "left":
            left_delta = active_delta.copy()
            left_gripper = active_gripper
            if self.bimanual_control_enabled:
                right_delta = active_delta * mirror_signs
                right_gripper = active_gripper
        else:
            right_delta = active_delta.copy()
            right_gripper = active_gripper
            if self.bimanual_control_enabled:
                left_delta = active_delta * mirror_signs
                left_gripper = active_gripper

        return left_delta, right_delta, left_gripper, right_gripper

    def get_keyboard_state(self) -> dict[str, bool]:
        if self._callback_mode and self._current_frame_keys is not None:
            return self._current_frame_keys.copy()
        return self._pressed_keys.copy()

    def reset(self) -> None:
        self._keyboard_cmd_queue.clear()
        self._last_keyboard_frame_id = -1
        self._current_frame_keys = None
        self._go_home_key_was_pressed = False
        for key in self._pressed_keys:
            self._pressed_keys[key] = False

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass
