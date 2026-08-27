#!/usr/bin/env python3
"""Build validated Sorting Roll sources as a LeRobot v2.1 dataset."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from sorting_roll_validate import (
    DIVERSE_TASK_VERSION,
    FPS,
    POLICY_CAMERAS,
    expand_episode_paths,
    validate_episode,
)
from sorting_roll_realsense_profile import POLICY_IMAGE_MAP
from sorting_roll_diversity import load_manifest, manifest_counts


IMAGE_SHAPE = (224, 224, 3)
STATE_NAMES = tuple(
    [f"j{index}" for index in range(14)]
    + ["grip_l", "grip_r", "base_v_fwd", "base_wz"]
)
ACTION_NAMES = tuple(
    [f"j{index}" for index in range(14)]
    + ["grip_l", "grip_r", "base_cmd_v_fwd", "base_cmd_wz"]
)
PROMPT = "Pick up the roll and place it stably in the integrated top shelf slot"
SPLIT_ORDER = ("train", "val", "test")
VIDEO_FILTER = (
    "scale=224:224:force_original_aspect_ratio=decrease,"
    "pad=224:224:(ow-iw)/2:(oh-ih)/2:black"
)


def policy_state_action(payload):
    state = np.concatenate(
        [
            np.asarray(payload["state"], dtype=np.float32),
            np.asarray(payload["base_velocity"], dtype=np.float32),
        ],
        axis=1,
    )
    action = np.concatenate(
        [
            np.asarray(payload["action"], dtype=np.float32),
            np.asarray(payload["base_action"], dtype=np.float32),
        ],
        axis=1,
    )
    if state.ndim != 2 or state.shape[1] != len(STATE_NAMES):
        raise ValueError(f"policy state has invalid shape {state.shape}")
    if action.ndim != 2 or action.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"policy action has invalid shape {action.shape}")
    if state.shape[0] != action.shape[0]:
        raise ValueError("policy state/action lengths differ")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("policy state/action contains NaN/Inf")
    return state, action


def sort_sources(sources):
    rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    return sorted(
        sources,
        key=lambda source: (
            rank[source["split"]],
            source["seed"],
            source["path"],
        ),
    )


def encode_video(frame_dir, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(frame_dir / "frame_%06d.jpg"),
            "-vf",
            VIDEO_FILTER,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if (
        completed.returncode != 0
        or not output_path.is_file()
        or output_path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"ffmpeg failed for {frame_dir}: {completed.stderr[:300]}"
        )


def validate_video(path, expected_frames):
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            (
                "stream=width,height,r_frame_rate,nb_read_frames:"
                "frame=best_effort_timestamp_time"
            ),
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr[:300]}")
    probe = json.loads(completed.stdout)
    stream = (probe.get("streams") or [{}])[0]
    actual = (
        stream.get("height"),
        stream.get("width"),
        stream.get("r_frame_rate"),
        int(stream.get("nb_read_frames", -1)),
    )
    expected = (IMAGE_SHAPE[0], IMAGE_SHAPE[1], f"{FPS}/1", expected_frames)
    if actual != expected:
        raise ValueError(f"{path}: video contract {actual} != {expected}")
    pts = np.asarray(
        [
            float(frame["best_effort_timestamp_time"])
            for frame in probe.get("frames", [])
        ]
    )
    expected_pts = np.arange(expected_frames) / FPS
    if pts.shape != expected_pts.shape or not np.allclose(
        pts, expected_pts, atol=1e-5, rtol=0
    ):
        raise ValueError(f"{path}: video PTS is not a uniform {FPS} FPS grid")


def encode_episode_videos(
    source, out, episode_index, num_frames, workers, reuse_video_paths=None
):
    jobs = []
    for camera in POLICY_CAMERAS:
        output = (
            out
            / "videos"
            / "chunk-000"
            / f"observation.images.{camera}"
            / f"episode_{episode_index:06d}.mp4"
        )
        reuse = (
            None
            if reuse_video_paths is None
            else Path(reuse_video_paths[camera])
        )
        jobs.append((source / "frames" / camera, output, reuse))

    def run(job):
        frame_dir, output, reuse = job
        if reuse is None:
            encode_video(frame_dir, output)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                output.hardlink_to(reuse)
            except OSError:
                shutil.copy2(reuse, output)
        validate_video(output, num_frames)

    if workers == 1:
        for job in jobs:
            run(job)
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        list(executor.map(run, jobs))


def channel_stats(values):
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [len(values)],
    }


def image_stats(num_frames):
    return {
        "mean": [[[0.5]]] * 3,
        "std": [[[0.25]]] * 3,
        "max": [[[1.0]]] * 3,
        "min": [[[0.0]]] * 3,
        "q01": [[[0.01]]] * 3,
        "q99": [[[0.99]]] * 3,
        "count": [num_frames],
    }


def load_sources(paths, manifest_assignments=None):
    episode_paths = expand_episode_paths(paths)
    if not episode_paths:
        raise ValueError("no complete source episodes found")
    sources = []
    failures = []
    seeds = set()
    for path in episode_paths:
        info, errors = validate_episode(path, manifest_assignments)
        if errors:
            failures.append(f"{path}: {'; '.join(errors)}")
            continue
        if info["seed"] in seeds:
            failures.append(f"{path}: duplicate seed {info['seed']}")
            continue
        seeds.add(info["seed"])
        sources.append(info)
    task_versions = {source["task_version"] for source in sources}
    collection_profiles = {
        source["collection_profile"] for source in sources
    }
    campaigns = {
        source["diversity"]["assignment"]["campaign"]
        for source in sources
        if source["task_version"] == DIVERSE_TASK_VERSION
    }
    if len(task_versions) > 1:
        failures.append(
            f"source task versions cannot be mixed: {sorted(task_versions)}"
        )
    if (
        task_versions == {DIVERSE_TASK_VERSION}
        and manifest_assignments is None
    ):
        failures.append("v10 sources require --manifest")
    if len(collection_profiles) > 1:
        failures.append(
            "source collection profiles cannot be mixed: "
            f"{sorted(collection_profiles)}"
        )
    if task_versions == {DIVERSE_TASK_VERSION} and len(campaigns) != 1:
        failures.append(f"source v10 campaigns cannot be mixed: {sorted(campaigns)}")
    if failures:
        raise ValueError("source validation failed:\n" + "\n".join(failures))
    return sort_sources(sources)


def build_dataset(sources, out, encode_workers):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    (out / "meta").mkdir(parents=True)
    (out / "data" / "chunk-000").mkdir(parents=True)
    episode_lines = []
    stats_lines = []
    source_lines = []
    split_ranges = {}
    global_index = 0
    prompts = sorted({source.get("prompt") or PROMPT for source in sources})
    task_index_by_prompt = {
        prompt: index for index, prompt in enumerate(prompts)
    }
    source_task_versions = sorted({
        source["task_version"] for source in sources
    })
    source_task_version = (
        source_task_versions[0]
        if len(source_task_versions) == 1
        else "mixed"
    )
    collection_profiles = sorted({
        source["collection_profile"] for source in sources
    })
    if len(collection_profiles) != 1:
        raise ValueError("source collection profiles cannot be mixed")
    collection_profile = collection_profiles[0]
    source_campaigns = sorted({
        source.get("campaign")
        or (
            ((source.get("diversity") or {}).get("assignment") or {})
            .get("campaign")
        )
        for source in sources
    } - {None})
    source_campaign = source_campaigns[0] if len(source_campaigns) == 1 else None
    legacy_diversity_assignments = [
        source["diversity"]["assignment"]
        for source in sources
        if source["task_version"] == DIVERSE_TASK_VERSION
        and isinstance(source.get("diversity"), dict)
    ]
    source_diversity_counts = (
        manifest_counts(legacy_diversity_assignments)
        if source_task_versions == [DIVERSE_TASK_VERSION]
        else {}
    )

    try:
        for episode_index, source in enumerate(sources):
            source_path = Path(source["path"])
            with np.load(
                source_path / "episode_data.npz", allow_pickle=False
            ) as payload:
                state, action = policy_state_action(payload)
            num_frames = len(state)
            split = source["split"]
            prompt = source.get("prompt") or PROMPT
            task_index = task_index_by_prompt[prompt]
            split_ranges.setdefault(split, [episode_index, episode_index])
            split_ranges[split][1] = episode_index + 1
            timestamp = (np.arange(num_frames) / FPS).astype(np.float32)
            table = pa.table({
                "observation.state": pa.array(
                    list(state), type=pa.list_(pa.float32(), state.shape[1])
                ),
                "action": pa.array(
                    list(action), type=pa.list_(pa.float32(), action.shape[1])
                ),
                "timestamp": pa.array(timestamp),
                "frame_index": pa.array(
                    np.arange(num_frames, dtype=np.int64)
                ),
                "episode_index": pa.array(
                    np.full(num_frames, episode_index, dtype=np.int64)
                ),
                "index": pa.array(
                    np.arange(
                        global_index,
                        global_index + num_frames,
                        dtype=np.int64,
                    )
                ),
                "task_index": pa.array(
                    np.full(num_frames, task_index, dtype=np.int64)
                ),
            })
            pq.write_table(
                table,
                out
                / "data"
                / "chunk-000"
                / f"episode_{episode_index:06d}.parquet",
            )
            encode_episode_videos(
                source_path,
                out,
                episode_index,
                num_frames,
                encode_workers,
                source.get("reuse_video_paths"),
            )
            episode_lines.append(json.dumps({
                "episode_index": episode_index,
                "tasks": [prompt],
                "length": num_frames,
                "source_seed": source["seed"],
                "source_split": split,
                "source_task_version": source["task_version"],
                "source_collection_profile": source["collection_profile"],
                "source_diversity": source.get("diversity"),
                "source_scenario": source.get("scenario"),
            }))
            episode_stats = {
                "observation.state": channel_stats(state),
                "action": channel_stats(action),
            }
            episode_stats.update({
                f"observation.images.{camera}": image_stats(num_frames)
                for camera in POLICY_CAMERAS
            })
            stats_lines.append(json.dumps({
                "episode_index": episode_index,
                "stats": episode_stats,
            }))
            source_lines.append(json.dumps({
                "source_seed": source["seed"],
                "split": split,
                "path": str(source_path),
                "source_frames": num_frames,
                "dataset_episode_index": episode_index,
                "source_task_version": source["task_version"],
                "source_collection_profile": source["collection_profile"],
                "prompt": prompt,
                "diversity": source.get("diversity"),
                "scenario": source.get("scenario"),
            }))
            global_index += num_frames
            print(
                f"[build] episode={episode_index} seed={source['seed']} "
                f"split={split} frames={num_frames}",
                flush=True,
            )
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise

    splits = {
        split: f"{split_ranges[split][0]}:{split_ranges[split][1]}"
        for split in SPLIT_ORDER
        if split in split_ranges
    }
    image_features = {
        f"observation.images.{camera}": {
            "dtype": "video",
            "shape": list(IMAGE_SHAPE),
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": FPS,
                "video.height": IMAGE_SHAPE[0],
                "video.width": IMAGE_SHAPE[1],
                "video.channels": IMAGE_SHAPE[2],
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
        for camera in POLICY_CAMERAS
    }
    scalar_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(STATE_NAMES)],
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": [len(ACTION_NAMES)],
            "names": list(ACTION_NAMES),
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "cruzr_s2",
        "source_task_version": source_task_version,
        "source_task_versions": source_task_versions,
        "source_campaign": source_campaign,
        "source_campaigns": source_campaigns,
        "collection_profile": collection_profile,
        "source_diversity_counts": source_diversity_counts,
        "policy_image_map": POLICY_IMAGE_MAP,
        "total_episodes": len(sources),
        "total_frames": global_index,
        "total_tasks": len(prompts),
        "total_videos": len(sources) * len(POLICY_CAMERAS),
        "total_source_episodes": len(sources),
        "total_chunks": 1,
        "chunks_size": max(2000, len(sources) + 1),
        "fps": FPS,
        "splits": splits,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {**image_features, **scalar_features},
    }
    (out / "meta" / "info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    (out / "meta" / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": index, "task": prompt}) + "\n"
            for index, prompt in enumerate(prompts)
        ),
        encoding="utf-8",
    )
    for name, lines in (
        ("episodes.jsonl", episode_lines),
        ("episodes_stats.jsonl", stats_lines),
        ("source_episodes.jsonl", source_lines),
    ):
        (out / "meta" / name).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    (out / "meta" / "rejected_sources.jsonl").write_text(
        "", encoding="utf-8"
    )
    print(
        f"[build] complete episodes={len(sources)} frames={global_index} "
        f"out={out}",
        flush=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--encode-workers", type=int, default=1)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if args.encode_workers < 1:
        parser.error("--encode-workers must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    manifest_assignments = None
    if args.manifest:
        manifest = load_manifest(args.manifest)
        manifest_assignments = {
            assignment["seed"]: assignment
            for assignment in manifest["assignments"]
        }
    sources = load_sources(args.sources, manifest_assignments)
    print(f"[build] validated sources={len(sources)}", flush=True)
    build_dataset(sources, args.out.resolve(), args.encode_workers)


if __name__ == "__main__":
    main()
