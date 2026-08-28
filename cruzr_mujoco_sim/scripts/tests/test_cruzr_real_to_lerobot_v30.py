#!/usr/bin/env python3

from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

import numpy as np


COLLECTION_DIR = Path(__file__).resolve().parents[1] / "collection"
sys.path.insert(0, str(COLLECTION_DIR))

import cruzr_real_to_lerobot_v30 as converter  # noqa: E402


def synthetic_motion(equal_wheels=True):
    joint_state = {
        name: float(index) / 10
        for index, name in enumerate(
            converter.ARM_JOINTS + ("waist_yaw_joint",) + converter.LIFTER_JOINTS
        )
    }
    left_command = {name: 1.0 + index for index, name in enumerate(converter.LEFT_ARM_JOINTS)}
    right_command = {name: 8.0 + index for index, name in enumerate(converter.RIGHT_ARM_JOINTS)}
    actuator = {
        converter.WHEEL_MOTORS[0]: {"velocity": 0.4, "cmd_vel": 0.5},
        converter.WHEEL_MOTORS[1]: {
            "velocity": 0.6,
            "cmd_vel": 0.5 if equal_wheels else -0.5,
        },
    }
    for joint, motor, sign in zip(
        converter.LIFTER_JOINTS,
        converter.LIFTER_MOTORS,
        converter.LIFTER_MOTOR_TO_JOINT_SIGNS,
        strict=True,
    ):
        actuator[motor] = {
            "position": sign * joint_state[joint],
            "cmd_pos": sign * (joint_state[joint] + 0.25),
        }
    return {
        converter.TOPIC_JOINT_STATE: [joint_state],
        converter.TOPIC_LEFT_ARM: [left_command],
        converter.TOPIC_RIGHT_ARM: [right_command],
        converter.TOPIC_WAIST: [{"waist_yaw_joint": 0.75}],
        converter.TOPIC_ACTUATOR: [actuator],
        converter.TOPIC_LEFT_GRIP_COMMAND: [0.025],
        converter.TOPIC_RIGHT_GRIP_COMMAND: [0.05],
        converter.TOPIC_LEFT_GRIP_STATE: [0.01],
        converter.TOPIC_RIGHT_GRIP_STATE: [0.04],
    }


class CruzrRealToLeRobotV30Test(unittest.TestCase):
    def test_cdr_alignment_is_relative_to_encapsulation(self):
        payload = bytearray(20)
        payload[4] = 2
        struct.pack_into("<d", payload, 12, 1.25)
        reader = converter.CdrReader(payload)
        self.assertEqual(reader.int8(), 2)
        self.assertEqual(reader.float64(), 1.25)
        self.assertEqual(reader.offset, 20)

    def test_motion_records_are_sorted_per_topic(self):
        messages = []
        for topic in converter.REQUIRED_TOPICS:
            messages.extend([(topic, 30, b"c"), (topic, 10, b"a"), (topic, 20, b"b")])
        with patch.object(converter, "iter_selected_messages", return_value=iter(messages)):
            records = converter.load_motion_records(Path("unused.mcap"))
        for items in records.values():
            self.assertEqual([item[0] for item in items], [10, 20, 30])

    def test_camera_selection_holds_latest_nonfuture_frame(self):
        paths = [Path("10.jpg"), Path("20.jpg"), Path("40.jpg")]
        selected, report = converter.select_camera_paths(
            paths,
            np.asarray([10, 20, 40]),
            np.asarray([20, 30, 40]),
        )
        self.assertEqual(selected, [paths[1], paths[1], paths[2]])
        self.assertEqual(report["selected_unique_frames"], 2)
        self.assertEqual(report["repeated_output_frames"], 1)

    def test_state_action_contract_and_transmission_signs(self):
        motion = synthetic_motion()
        state, action, report = converter.build_state_action(motion)
        self.assertEqual(state.shape, (1, 22))
        self.assertEqual(action.shape, (1, 22))
        np.testing.assert_allclose(state[0, :14], np.arange(14) / 10)
        np.testing.assert_allclose(state[0, 14:16], [0.2, 0.8])
        np.testing.assert_allclose(action[0, 14:16], [0.5, 1.0])
        self.assertEqual(state[0, 16], 0.0)
        self.assertEqual(action[0, 16], 0.0)
        self.assertAlmostEqual(
            state[0, 17], converter.BASE_WZ_PER_WHEEL_MOTOR_RAD_S * 0.5
        )
        self.assertAlmostEqual(
            action[0, 17], converter.BASE_WZ_PER_WHEEL_MOTOR_RAD_S * 0.5
        )
        np.testing.assert_allclose(action[0, 19:22], state[0, 19:22] + 0.25)
        self.assertEqual([item["sign"] for item in report], [1.0, -1.0, 1.0])

    def test_forward_wheel_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "forward motion"):
            converter.build_state_action(synthetic_motion(equal_wheels=False))


if __name__ == "__main__":
    unittest.main()
