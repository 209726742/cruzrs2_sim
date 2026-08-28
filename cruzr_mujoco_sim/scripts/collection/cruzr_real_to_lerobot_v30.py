#!/usr/bin/env python3
"""Convert paired CRUZR S2 real-robot recordings to LeRobot v3.0."""

import argparse
import json
from pathlib import Path
import struct
import subprocess
import time

import numpy as np


FPS = 30
IMAGE_SHAPE = (224, 224, 3)
GRIPPER_MAX_M = 0.05
# Identified from odom->base_footprint yaw and equal wheel-motor commands in
# episode4. The recordings contain rotation only, so no forward mapping can be
# identified from this batch.
BASE_WZ_PER_WHEEL_MOTOR_RAD_S = -0.47764
START_END_MARGIN_NS = 100_000_000

LEFT_ARM_JOINTS = (
    "L_shoulder_pitch_joint",
    "L_shoulder_roll_joint",
    "L_shoulder_yaw_joint",
    "L_elbow_roll_joint",
    "L_elbow_yaw_joint",
    "L_wrist_pitch_joint",
    "L_wrist_roll_joint",
)
RIGHT_ARM_JOINTS = tuple(name.replace("L_", "R_", 1) for name in LEFT_ARM_JOINTS)
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
LIFTER_JOINTS = tuple(f"lifter_pitch_{index}_joint" for index in range(1, 4))
LIFTER_MOTORS = tuple(f"lifter_pitch_{index}_motor" for index in range(1, 4))
LIFTER_MOTOR_TO_JOINT_SIGNS = (1.0, -1.0, 1.0)
WHEEL_MOTORS = ("driving_wheel_left_motor", "driving_wheel_right_motor")

STATE_NAMES = (
    *ARM_JOINTS,
    "left_gripper_open_fraction",
    "right_gripper_open_fraction",
    "base_linear_velocity_mps",
    "base_angular_velocity_rad_s",
    "waist_yaw_joint",
    *LIFTER_JOINTS,
)
ACTION_NAMES = (
    *(f"{name}_position_command" for name in ARM_JOINTS),
    "left_gripper_open_fraction_command",
    "right_gripper_open_fraction_command",
    "base_linear_velocity_command_mps",
    "base_angular_velocity_command_rad_s",
    "waist_yaw_joint_position_command",
    *(f"{name}_position_command" for name in LIFTER_JOINTS),
)

CAMERAS = {
    "observation.images.base_0_rgb": "sensor_camera_stereo_color_raw",
    "observation.images.left_wrist_0_rgb": "sensor_camera_wrist_left_color_raw",
    "observation.images.right_wrist_0_rgb": "sensor_camera_wrist_right_color_raw",
}

TOPIC_JOINT_STATE = "/mc/joint_states"
TOPIC_LEFT_ARM = "/mc/left_arm/joint_command"
TOPIC_RIGHT_ARM = "/mc/right_arm/joint_command"
TOPIC_WAIST = "/mc/waist/joint_command"
TOPIC_ACTUATOR = "/mc/actuator_state"
TOPIC_LEFT_GRIP_COMMAND = "/ecat/left_grip/cmd"
TOPIC_RIGHT_GRIP_COMMAND = "/ecat/right_grip/cmd"
TOPIC_LEFT_GRIP_STATE = "/ecat/left_grip/state"
TOPIC_RIGHT_GRIP_STATE = "/ecat/right_grip/state"
REQUIRED_TOPICS = (
    TOPIC_JOINT_STATE,
    TOPIC_LEFT_ARM,
    TOPIC_RIGHT_ARM,
    TOPIC_WAIST,
    TOPIC_ACTUATOR,
    TOPIC_LEFT_GRIP_COMMAND,
    TOPIC_RIGHT_GRIP_COMMAND,
    TOPIC_LEFT_GRIP_STATE,
    TOPIC_RIGHT_GRIP_STATE,
)


class CdrReader:
    def __init__(self, payload, encapsulated=True):
        self.data = memoryview(payload)
        self.base = 4 if encapsulated else 0
        self.offset = self.base

    def align(self, size):
        self.offset += (-(self.offset - self.base)) % size

    def unpack(self, fmt, alignment):
        self.align(alignment)
        value = struct.unpack_from("<" + fmt, self.data, self.offset)
        self.offset += struct.calcsize(fmt)
        return value[0] if len(value) == 1 else value

    def int8(self):
        return self.unpack("b", 1)

    def uint8(self):
        return self.unpack("B", 1)

    def uint16(self):
        return self.unpack("H", 2)

    def uint32(self):
        return self.unpack("I", 4)

    def int32(self):
        return self.unpack("i", 4)

    def int64(self):
        return self.unpack("q", 8)

    def float64(self):
        return self.unpack("d", 8)

    def string(self):
        size = self.uint32()
        value = bytes(self.data[self.offset : self.offset + size])
        self.offset += size
        if value.endswith(b"\0"):
            value = value[:-1]
        return value.decode("utf-8")

    def header(self):
        return self.int32(), self.uint32(), self.string()


def _read_uint16(data, offset):
    return struct.unpack_from("<H", data, offset)[0], offset + 2


def _read_uint32(data, offset):
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def _read_uint64(data, offset):
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def _read_string(data, offset):
    size, offset = _read_uint32(data, offset)
    value = bytes(data[offset : offset + size]).decode("utf-8")
    return value, offset + size


def iter_mcap_records(data):
    offset = 0
    while offset < len(data):
        if offset + 9 > len(data):
            raise ValueError(f"truncated MCAP record at byte {offset}")
        opcode = data[offset]
        size = struct.unpack_from("<Q", data, offset + 1)[0]
        start = offset + 9
        stop = start + size
        if stop > len(data):
            raise ValueError(f"MCAP record at byte {offset} exceeds input")
        yield opcode, memoryview(data)[start:stop]
        offset = stop


def _decompress_zstd(payload):
    from numcodecs import zstd

    return zstd.decompress(payload)


def _iter_messages(data, channels):
    for opcode, record in iter_mcap_records(data):
        if opcode == 4:  # Channel
            channel_id, offset = _read_uint16(record, 0)
            _, offset = _read_uint16(record, offset)
            topic, offset = _read_string(record, offset)
            channels[channel_id] = topic
        elif opcode == 5:  # Message
            channel_id, _ = _read_uint16(record, 0)
            topic = channels.get(channel_id)
            if topic in REQUIRED_TOPICS:
                log_time = struct.unpack_from("<Q", record, 6)[0]
                yield topic, log_time, bytes(record[22:])
        elif opcode == 6:  # Chunk
            offset = 28
            compression, offset = _read_string(record, offset)
            size, offset = _read_uint64(record, offset)
            chunk = bytes(record[offset : offset + size])
            if compression == "zstd":
                chunk = _decompress_zstd(chunk)
            elif compression:
                raise ValueError(f"unsupported MCAP chunk compression: {compression}")
            yield from _iter_messages(chunk, channels)


def iter_selected_messages(path):
    raw = path.read_bytes()
    magic = b"\x89MCAP0\r\n"
    if not raw.startswith(magic) or not raw.endswith(magic):
        raise ValueError(f"invalid MCAP magic: {path}")
    yield from _iter_messages(memoryview(raw)[len(magic) : -len(magic)], {})


def load_motion_records(path):
    records = {topic: [] for topic in REQUIRED_TOPICS}
    for topic, log_time_ns, payload in iter_selected_messages(path):
        records[topic].append((log_time_ns, payload))
    missing = [topic for topic, items in records.items() if not items]
    if missing:
        raise ValueError(f"missing motion topics in {path}: {missing}")
    for items in records.values():
        items.sort(key=lambda item: item[0])
    return records


def decode_robot_command(payload):
    reader = CdrReader(_decompress_zstd(payload))
    reader.header()
    commands = {}
    for _ in range(reader.uint32()):
        name = reader.string()
        control_mode = reader.int8()
        position = reader.float64()
        for _ in range(5):
            reader.float64()
        if control_mode != 2:
            raise ValueError(f"{name} has non-position control mode {control_mode}")
        commands[name] = position
    return commands


def decode_joint_state(payload):
    reader = CdrReader(_decompress_zstd(payload))
    reader.header()
    names = [reader.string() for _ in range(reader.uint32())]
    positions = [reader.float64() for _ in range(reader.uint32())]
    for _ in range(reader.uint32()):
        reader.float64()
    for _ in range(reader.uint32()):
        reader.float64()
    if len(names) != len(positions):
        raise ValueError("JointState name/position lengths differ")
    return dict(zip(names, positions, strict=True))


def _decode_gripper(payload, command):
    reader = CdrReader(_decompress_zstd(payload), encapsulated=False)
    reader.int32()
    reader.uint32()
    reader.offset += 256
    reader.uint8()
    for _ in range(5 if command else 4):
        reader.uint16()
    return reader.float64()


def decode_actuator_state(payload):
    reader = CdrReader(_decompress_zstd(payload))
    reader.header()
    selected = {}
    wanted = set(WHEEL_MOTORS + LIFTER_MOTORS)
    for _ in range(reader.uint32()):
        name = reader.string()
        reader.string()
        reader.uint32()
        reader.int32()
        reader.uint32()
        reader.float64()
        reader.float64()
        reader.int64()
        position = reader.float64()
        velocity = reader.float64()
        reader.float64()
        reader.float64()
        control_mode = reader.int8()
        cmd_pos = reader.float64()
        cmd_vel = reader.float64()
        for _ in range(4):
            reader.float64()
        reader.uint16()
        reader.uint16()
        if name in wanted:
            selected[name] = {
                "position": position,
                "velocity": velocity,
                "control_mode": control_mode,
                "cmd_pos": cmd_pos,
                "cmd_vel": cmd_vel,
            }
    missing = wanted - selected.keys()
    if missing:
        raise ValueError(f"ActuatorState missing {sorted(missing)}")
    return selected


DECODERS = {
    TOPIC_JOINT_STATE: decode_joint_state,
    TOPIC_LEFT_ARM: decode_robot_command,
    TOPIC_RIGHT_ARM: decode_robot_command,
    TOPIC_WAIST: decode_robot_command,
    TOPIC_ACTUATOR: decode_actuator_state,
    TOPIC_LEFT_GRIP_COMMAND: lambda payload: _decode_gripper(payload, True),
    TOPIC_RIGHT_GRIP_COMMAND: lambda payload: _decode_gripper(payload, True),
    TOPIC_LEFT_GRIP_STATE: lambda payload: _decode_gripper(payload, False),
    TOPIC_RIGHT_GRIP_STATE: lambda payload: _decode_gripper(payload, False),
}


def sample_motion(records, grid_ns):
    values = {}
    age_report = {}
    for topic, items in records.items():
        times = np.fromiter(
            (item[0] for item in items),
            dtype=np.int64,
            count=len(items),
        )
        indices = np.searchsorted(times, grid_ns, side="right") - 1
        if np.any(indices < 0):
            raise ValueError(f"{topic} starts after the requested time grid")
        decoded = {}
        sampled = []
        for index in indices:
            index = int(index)
            if index not in decoded:
                decoded[index] = DECODERS[topic](items[index][1])
            sampled.append(decoded[index])
        source_ns = times[indices]
        ages = grid_ns - source_ns
        if np.any(ages < 0):
            raise ValueError(f"{topic} sampled a future message")
        values[topic] = sampled
        age_report[topic] = {
            "max_age_ms": float(ages.max() / 1e6),
            "p99_age_ms": float(np.quantile(ages, 0.99) / 1e6),
        }
    return values, age_report


def image_paths_and_times(visual_dir, camera_dir):
    direct = visual_dir / "image_data" / camera_dir
    directories = [direct] if direct.is_dir() else list(
        visual_dir.glob(f"**/image_data/{camera_dir}")
    )
    if len(directories) != 1:
        raise ValueError(f"expected one {camera_dir} directory under {visual_dir}, got {directories}")
    paths = sorted(directories[0].glob("*.jpg"), key=lambda path: int(path.stem))
    if not paths:
        raise ValueError(f"no JPEG files in {directories[0]}")
    times = np.asarray([int(path.stem) for path in paths], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"non-increasing image timestamps in {directories[0]}")
    return paths, times


def make_time_grid(camera_times, motion_bounds, max_frames=None):
    start_ns = max(
        max(times[0] for times in camera_times.values()),
        max(value[0] for value in motion_bounds.values()),
    ) + START_END_MARGIN_NS
    stop_ns = min(
        min(times[-1] for times in camera_times.values()),
        min(value[1] for value in motion_bounds.values()),
    ) - START_END_MARGIN_NS
    period_ns = 1_000_000_000 // FPS
    start_ns = ((start_ns + period_ns - 1) // period_ns) * period_ns
    count = (stop_ns - start_ns) // period_ns + 1
    if count <= 0:
        raise ValueError("camera streams have no common time range")
    if max_frames is not None:
        count = min(count, max_frames)
    return start_ns + np.arange(count, dtype=np.int64) * period_ns


def select_camera_paths(paths, times, grid_ns):
    indices = np.searchsorted(times, grid_ns, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("camera stream starts after the requested time grid")
    ages = grid_ns - times[indices]
    return [paths[index] for index in indices], {
        "source_frames": len(paths),
        "selected_unique_frames": int(len(np.unique(indices))),
        "repeated_output_frames": int(len(indices) - len(np.unique(indices))),
        "max_age_ms": float(ages.max() / 1e6),
        "p99_age_ms": float(np.quantile(ages, 0.99) / 1e6),
    }


def resize_with_padding(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode JPEG: {path}")
    height, width = image.shape[:2]
    scale = min(IMAGE_SHAPE[1] / width, IMAGE_SHAPE[0] / height)
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    output = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    y = (IMAGE_SHAPE[0] - resized.shape[0]) // 2
    x = (IMAGE_SHAPE[1] - resized.shape[1]) // 2
    output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized[:, :, ::-1]
    return output


def _fraction(position_m):
    return float(np.clip(position_m / GRIPPER_MAX_M, 0.0, 1.0))


def infer_lifter_signs(motion):
    joints = motion[TOPIC_JOINT_STATE]
    actuators = motion[TOPIC_ACTUATOR]
    signs = []
    report = []
    for joint, motor, sign in zip(
        LIFTER_JOINTS,
        LIFTER_MOTORS,
        LIFTER_MOTOR_TO_JOINT_SIGNS,
        strict=True,
    ):
        joint_values = np.asarray([value[joint] for value in joints])
        motor_values = np.asarray([value[motor]["position"] for value in actuators])
        positive_error = float(np.median(np.abs(joint_values - motor_values)))
        negative_error = float(np.median(np.abs(joint_values + motor_values)))
        signs.append(sign)
        report.append(
            {
                "joint": joint,
                "motor": motor,
                "sign": sign,
                "positive_median_error": positive_error,
                "negative_median_error": negative_error,
            }
        )
    return np.asarray(signs), report


def build_state_action(motion):
    lifter_signs, lifter_report = infer_lifter_signs(motion)
    state_rows = []
    action_rows = []
    for index in range(len(motion[TOPIC_JOINT_STATE])):
        joint_state = motion[TOPIC_JOINT_STATE][index]
        actuator = motion[TOPIC_ACTUATOR][index]
        left_command = motion[TOPIC_LEFT_ARM][index]
        right_command = motion[TOPIC_RIGHT_ARM][index]
        waist_command = motion[TOPIC_WAIST][index]
        wheel_velocity = np.mean([actuator[name]["velocity"] for name in WHEEL_MOTORS])
        wheel_commands = [actuator[name]["cmd_vel"] for name in WHEEL_MOTORS]
        if abs(wheel_commands[0] - wheel_commands[1]) > 1e-6:
            raise ValueError(
                "wheel commands include unidentifiable forward motion; "
                "the rotation-only base mapping is not applicable"
            )
        wheel_command = np.mean(wheel_commands)
        lifter_commands = [
            sign * actuator[motor]["cmd_pos"]
            for sign, motor in zip(lifter_signs, LIFTER_MOTORS, strict=True)
        ]
        state_rows.append(
            [joint_state[name] for name in ARM_JOINTS]
            + [
                _fraction(motion[TOPIC_LEFT_GRIP_STATE][index]),
                _fraction(motion[TOPIC_RIGHT_GRIP_STATE][index]),
                0.0,
                BASE_WZ_PER_WHEEL_MOTOR_RAD_S * wheel_velocity,
                joint_state["waist_yaw_joint"],
            ]
            + [joint_state[name] for name in LIFTER_JOINTS]
        )
        action_rows.append(
            [left_command[name] for name in LEFT_ARM_JOINTS]
            + [right_command[name] for name in RIGHT_ARM_JOINTS]
            + [
                _fraction(motion[TOPIC_LEFT_GRIP_COMMAND][index]),
                _fraction(motion[TOPIC_RIGHT_GRIP_COMMAND][index]),
                0.0,
                BASE_WZ_PER_WHEEL_MOTOR_RAD_S * wheel_command,
                waist_command["waist_yaw_joint"],
            ]
            + lifter_commands
        )
    state = np.asarray(state_rows, dtype=np.float32)
    action = np.asarray(action_rows, dtype=np.float32)
    if state.shape != (len(state_rows), len(STATE_NAMES)):
        raise ValueError(f"invalid state shape {state.shape}")
    if action.shape != (len(action_rows), len(ACTION_NAMES)):
        raise ValueError(f"invalid action shape {action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("state/action contains NaN or Inf")
    return state, action, lifter_report


def episode_number(path):
    suffix = path.name.removeprefix("episode").split("_", 1)[0]
    return int(suffix)


def discover_sources(data_root):
    sources = []
    for episode_dir in data_root.glob("**/episode*"):
        if not episode_dir.is_dir() or episode_dir.name in {
            "episode3_fail",
            "episode11",
            "episode23_zhedie",
        }:
            continue
        motion = list(episode_dir.glob("*_m/bag/bag_0.mcap"))
        visual = [
            path
            for path in episode_dir.glob("*_v")
            if all(list(path.glob(f"**/image_data/{camera}")) for camera in CAMERAS.values())
        ]
        if len(motion) != 1 or len(visual) != 1:
            raise ValueError(
                f"{episode_dir}: expected one motion MCAP and visual directory, "
                f"got motion={motion}, visual={visual}"
            )
        sources.append(
            {
                "name": episode_dir.name,
                "episode_dir": episode_dir.resolve(),
                "motion_mcap": motion[0].resolve(),
                "visual_dir": visual[0].resolve(),
            }
        )
    sources.sort(key=lambda source: episode_number(Path(source["name"])))
    if len(sources) != 23:
        raise ValueError(f"expected 23 selected paired episodes, found {len(sources)}")
    return sources


def prepare_episode(source, max_frames=None):
    camera_paths = {}
    camera_times = {}
    for feature, directory in CAMERAS.items():
        paths, times = image_paths_and_times(source["visual_dir"], directory)
        camera_paths[feature] = paths
        camera_times[feature] = times
    records = load_motion_records(source["motion_mcap"])
    bounds = {topic: (items[0][0], items[-1][0]) for topic, items in records.items()}
    grid_ns = make_time_grid(camera_times, bounds, max_frames=max_frames)
    selected_paths = {}
    camera_report = {}
    for feature in CAMERAS:
        selected_paths[feature], camera_report[feature] = select_camera_paths(
            camera_paths[feature], camera_times[feature], grid_ns
        )
    motion, motion_age_report = sample_motion(records, grid_ns)
    state, action, lifter_report = build_state_action(motion)
    report = {
        "source": source["name"],
        "source_dir": str(source["episode_dir"]),
        "frames": len(grid_ns),
        "duration_s": len(grid_ns) / FPS,
        "start_time_ns": int(grid_ns[0]),
        "stop_time_ns": int(grid_ns[-1]),
        "camera_alignment": camera_report,
        "motion_alignment": motion_age_report,
        "lifter_motor_to_joint": lifter_report,
        "state_min": state.min(axis=0).tolist(),
        "state_max": state.max(axis=0).tolist(),
        "action_min": action.min(axis=0).tolist(),
        "action_max": action.max(axis=0).tolist(),
    }
    return selected_paths, state, action, report


def dataset_features():
    features = {
        key: {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        }
        for key in CAMERAS
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(STATE_NAMES),),
        "names": list(STATE_NAMES),
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (len(ACTION_NAMES),),
        "names": list(ACTION_NAMES),
    }
    return features


def open_dataset(args, create):
    from src.lerobot.datasets.lerobot_dataset import LeRobotDataset

    if create:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=args.output,
            fps=FPS,
            robot_type="cruzr_s2",
            features=dataset_features(),
            use_videos=True,
            metadata_buffer_size=1,
            vcodec="h264",
            streaming_encoding=True,
            encoder_queue_maxsize=3000,
            encoder_threads=args.encoder_threads,
        )
    else:
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.output,
            video_backend="pyav",
            vcodec="h264",
            streaming_encoding=True,
            encoder_queue_maxsize=3000,
            encoder_threads=args.encoder_threads,
        )
        dataset.meta.metadata_buffer_size = 1
    return dataset


def write_progress(path, progress):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    temporary.replace(path)


def add_info_metadata(dataset, source_count, task):
    from src.lerobot.datasets.utils import write_info

    dataset.meta.info.update(
        {
            "source_format": "cruzr_s2_ros2_mcap_and_timestamped_jpeg",
            "source_task": task,
            "total_source_episodes": source_count,
            "conversion_profile": "cruzr_real_full_body_22d_v1",
            "policy_image_map": {
                "observation/image": "observation.images.base_0_rgb",
                "observation/left_wrist_image": "observation.images.left_wrist_0_rgb",
                "observation/right_wrist_image": "observation.images.right_wrist_0_rgb",
            },
            "base_mapping": {
                "forward_velocity": "constant_zero_unobserved_in_source_batch",
                "angular_velocity_per_mean_wheel_motor_rad_s": BASE_WZ_PER_WHEEL_MOTOR_RAD_S,
                "identification_source": "episode4 odom_to_base_footprint yaw",
            },
            "state_names": list(STATE_NAMES),
            "action_names": list(ACTION_NAMES),
        }
    )
    write_info(dataset.meta.info, dataset.root)


def convert(args):
    sources = discover_sources(args.data_root)
    if args.max_episodes is not None:
        sources = sources[: args.max_episodes]
    source_order = [str(source["episode_dir"]) for source in sources]
    progress_path = args.output / "meta" / "conversion_progress.json"

    created = not args.output.exists()
    if not created:
        if not progress_path.is_file():
            raise ValueError(f"output exists without conversion progress: {args.output}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["repo_id"] != args.repo_id or progress["source_order"] != source_order:
            raise ValueError("existing conversion configuration does not match this run")
        dataset = open_dataset(args, create=False)
        completed = dataset.meta.total_episodes
        if completed > len(sources):
            raise ValueError("existing dataset has more episodes than selected sources")
        print(f"[resume] completed={completed}/{len(sources)}", flush=True)
        dataset.finalize()
    else:
        dataset = open_dataset(args, create=True)
        add_info_metadata(dataset, len(sources), args.task)
        progress = {
            "repo_id": args.repo_id,
            "task": args.task,
            "source_order": source_order,
            "episodes": [],
            "status": "running",
        }
        write_progress(progress_path, progress)
        completed = 0

    overall_start = time.monotonic()
    for episode_index, source in enumerate(sources[completed:], start=completed):
        started = time.monotonic()
        print(f"[episode {episode_index + 1}/{len(sources)}] prepare {source['name']}", flush=True)
        selected_paths, state, action, report = prepare_episode(
            source, max_frames=args.max_frames
        )
        if not (created and episode_index == 0):
            dataset = open_dataset(args, create=False)
        if dataset.meta.total_episodes != episode_index:
            raise ValueError("episode index changed while conversion was running")
        for frame_index in range(len(state)):
            frame = {
                "observation.state": state[frame_index],
                "action": action[frame_index],
                "task": args.task,
            }
            for feature in CAMERAS:
                frame[feature] = resize_with_padding(selected_paths[feature][frame_index])
            dataset.add_frame(frame)
        dataset.save_episode()
        dataset.finalize()
        report["conversion_seconds"] = time.monotonic() - started
        if len(progress["episodes"]) == episode_index:
            progress["episodes"].append(report)
        else:
            progress["episodes"][episode_index] = report
        write_progress(progress_path, progress)
        elapsed = time.monotonic() - overall_start
        average = elapsed / (episode_index - completed + 1)
        eta = average * (len(sources) - episode_index - 1)
        print(
            f"[episode {episode_index + 1}/{len(sources)}] saved frames={len(state)} "
            f"seconds={report['conversion_seconds']:.1f} eta_seconds={eta:.0f}",
            flush=True,
        )

    progress["status"] = "complete"
    progress["completed_episodes"] = len(sources)
    progress["completed_frames"] = sum(item["frames"] for item in progress["episodes"])
    write_progress(progress_path, progress)
    dataset = open_dataset(args, create=False)
    add_info_metadata(dataset, len(sources), args.task)
    dataset.finalize()
    print(
        f"[complete] episodes={dataset.meta.total_episodes} "
        f"frames={dataset.meta.total_frames} output={dataset.root}",
        flush=True,
    )


def audit(args):
    from src.lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.output, video_backend="pyav")
    progress = json.loads(
        (args.output / "meta" / "conversion_progress.json").read_text(encoding="utf-8")
    )
    if progress.get("status") != "complete":
        raise ValueError("conversion progress is not complete")
    if dataset.meta.total_episodes != progress["completed_episodes"]:
        raise ValueError("episode count differs between LeRobot metadata and conversion progress")
    if dataset.meta.total_frames != progress["completed_frames"]:
        raise ValueError("frame count differs between LeRobot metadata and conversion progress")
    if dataset.meta.shapes["observation.state"] != (len(STATE_NAMES),):
        raise ValueError("state shape mismatch")
    if dataset.meta.shapes["action"] != (len(ACTION_NAMES),):
        raise ValueError("action shape mismatch")
    for feature in ("observation.state", "action"):
        if not {"q01", "q99"}.issubset(dataset.meta.stats[feature]):
            raise ValueError(f"missing quantile stats for {feature}")
    boundary_indices = {0, len(dataset) - 1}
    for episode in dataset.meta.episodes:
        boundary_indices.add(episode["dataset_from_index"])
        boundary_indices.add(episode["dataset_to_index"] - 1)
    for index in sorted(boundary_indices):
        sample = dataset[index]
        if tuple(sample["observation.state"].shape) != (len(STATE_NAMES),):
            raise ValueError(f"state decode failed at frame {index}")
        if tuple(sample["action"].shape) != (len(ACTION_NAMES),):
            raise ValueError(f"action decode failed at frame {index}")
        for feature in CAMERAS:
            if tuple(sample[feature].shape) != (3, 224, 224):
                raise ValueError(f"camera decode failed at frame {index}: {feature}")
    for episode_index, episode in enumerate(dataset.meta.episodes):
        for feature in CAMERAS:
            path = dataset.root / dataset.meta.get_video_file_path(episode_index, feature)
            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=width,height,r_frame_rate,nb_read_frames",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            stream = json.loads(completed.stdout)["streams"][0]
            actual = (
                int(stream["width"]),
                int(stream["height"]),
                stream["r_frame_rate"],
                int(stream["nb_read_frames"]),
            )
            expected = (224, 224, "30/1", int(episode["length"]))
            if actual != expected:
                raise ValueError(f"video contract mismatch for {path}: {actual} != {expected}")
    print(
        f"[audit complete] episodes={dataset.meta.total_episodes} "
        f"frames={dataset.meta.total_frames} cameras={list(CAMERAS)}",
        flush=True,
    )


def parse_args():
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("convert", "audit"), nargs="?", default="convert")
    parser.add_argument("--data-root", type=Path, default=project_root / "real_data")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            project_root
            / "cruzr_mujoco_sim/out/datasets/cruzr_real_clamp_23ep_lerobot_v30_20260828"
        ),
    )
    parser.add_argument("--repo-id", default="local/cruzr_real_clamp_23ep")
    parser.add_argument("--task", default="Clamp the target object.")
    parser.add_argument("--encoder-threads", type=int, default=4)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.encoder_threads < 1:
        parser.error("--encoder-threads must be positive")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    return args


def main():
    args = parse_args()
    if args.action == "convert":
        convert(args)
    else:
        audit(args)


if __name__ == "__main__":
    main()
