"""ROS2 JointState teleoperation subscriber (optional Pico4 / redis2isaac_bridge link).

ROS2_AVAILABLE is False when rclpy is not installed, safe to import in main program.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

ROS2_AVAILABLE = False
try:
    import rclpy  # noqa: F401

    ROS2_AVAILABLE = True
except ImportError:
    rclpy = None  # type: ignore[misc, assignment]


class ROS2TeleopSubscriber:
    """
    Subscribe to ROS2 JointState, spin in background thread, main thread gets latest commands via get_latest.

    数据流示例: Pico4  -> ... -> redis2isaac_bridge -> ROS2 -> this class
    """

    def __init__(
        self,
        arm_joint_names: list[str],
        finger_joint_names: list[str],
        topic: str = "/isaac/joint_position_commands",
    ):
        self._arm_joint_names = list(arm_joint_names)
        self._finger_joint_names = list(finger_joint_names)
        self._topic = topic

        self._lock = threading.Lock()
        self._latest_arm_positions: Optional[np.ndarray] = None
        self._latest_finger_positions: Optional[np.ndarray] = None
        self._latest_timestamp: Optional[float] = None

        self._node = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> bool:
        if not ROS2_AVAILABLE:
            logger.warning("rclpy not available, ROS2 teleoperation subscriber cannot start")
            return False

        try:
            import rclpy as _rclpy

            if not _rclpy.ok():
                _rclpy.init()
        except Exception:
            import rclpy as _rclpy

            _rclpy.init()

        import rclpy as _rclpy
        from sensor_msgs.msg import JointState as _JointState

        self._node = _rclpy.create_node("isaac_pico4_teleop_subscriber")
        self._node.create_subscription(
            _JointState,
            self._topic,
            self._on_joint_state,
            10,
        )
        self._running = True
        self._thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._thread.start()
        logger.info(f"[ROS2Teleop] subscription started: {self._topic}")
        return True

    def _spin_loop(self) -> None:
        import rclpy as _rclpy

        while self._running and _rclpy.ok():
            _rclpy.spin_once(self._node, timeout_sec=0.005)

    def _on_joint_state(self, msg: Any) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))

        arm_positions = []
        for jname in self._arm_joint_names:
            if jname not in name_to_pos:
                return
            arm_positions.append(name_to_pos[jname])

        finger_positions: list[float] = []
        fingers_found = True
        for jname in self._finger_joint_names:
            if jname not in name_to_pos:
                fingers_found = False
                break
            finger_positions.append(name_to_pos[jname])

        with self._lock:
            self._latest_arm_positions = np.array(arm_positions, dtype=np.float32)
            self._latest_finger_positions = (
                np.array(finger_positions, dtype=np.float32) if fingers_found else None
            )
            self._latest_timestamp = time.time()

    def get_latest(self, max_age_s: float = 0.5) -> Optional[Dict[str, Optional[np.ndarray]]]:
        with self._lock:
            if self._latest_arm_positions is None:
                return None
            if self._latest_timestamp is not None:
                age = time.time() - self._latest_timestamp
                if age > max_age_s:
                    return None
            return {
                "arm_positions": self._latest_arm_positions.copy(),
                "finger_positions": (
                    self._latest_finger_positions.copy()
                    if self._latest_finger_positions is not None
                    else None
                ),
            }

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        try:
            import rclpy as _rclpy

            if _rclpy.ok():
                _rclpy.shutdown()
        except Exception:
            pass
        logger.info("[ROS2Teleop] Stopped")


def connect_ros2_teleop_if_enabled(
    config: Any,
    arm_joint_names: list[str],
    finger_joint_names: list[str],
) -> Optional[ROS2TeleopSubscriber]:
    """Create and start ROS2 subscriber based on WalkerS2Config."""
    if not getattr(config, "enable_ros2_teleop", True):
        return None
    if not ROS2_AVAILABLE:
        logger.info("[ROS2Teleop] rclpy not installed, using pure keyboard/evdev teleoperation")
        return None

    topic = getattr(
        config,
        "ros2_joint_commands_topic",
        "/isaac/joint_position_commands",
    )
    sub = ROS2TeleopSubscriber(
        arm_joint_names=arm_joint_names,
        finger_joint_names=finger_joint_names,
        topic=topic,
    )
    if sub.start():
        logger.info(f"[ROS2Teleop] Enabled, subscribed to {topic}")
        return sub
    logger.warning("[ROS2Teleop] Start failed, using pure keyboard/evdev teleoperation")
    return None


def stop_ros2_teleop(sub: Optional[ROS2TeleopSubscriber]) -> None:
    if sub is not None:
        sub.stop()
