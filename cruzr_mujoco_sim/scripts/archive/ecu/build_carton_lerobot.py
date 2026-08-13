#!/usr/bin/env python3
"""Aggregate scripted-auto transport_carton recordings -> ONE multi-episode LeRobot v2.1 dataset.

Source: sim/recorder.py products under outputs/teleop/<run>/ :
    meta.json, episode_data.npz (timestamp/state/action/action_real/base/base_velocity/base_action/phase),
    frames/{top_head,hand_left,hand_right,...}/frame_%06d.jpg
Each recording = ONE episode of the AgiBot G2 dual-arm omnipicker doing
approach->grasp->lift(->place) on the carton in the Isaac/GenieSim transport_carton scene.

Output (HF_LEROBOT_HOME/<repo_id>), LeRobot v2.1:
    meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl}
    data/chunk-000/episode_%06d.parquet
    videos/chunk-000/observation.images.<cam>/episode_%06d.mp4  (h264)

Fixed-base recordings use 16-dim joint-POSITION state+action
(7 L arm + 7 R arm + 2 gripper open fractions, 1=open/0=closed).
Mobile-base recordings append base pose/velocity to state and base command to action:
state 21D = 16D arm/gripper + base_x/base_y/base_yaw/base_v_fwd/base_wz,
action 18D = 16D arm/gripper command + base_cmd_v_fwd/base_cmd_wz.
3 cameras mapped to openpi pi05 slots: top_head=base, hand_left=left_wrist, hand_right=right_wrist.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

# v5 (2026-07-16): all five recorded cameras go into the dataset; the training config
# picks its three views via repack. Override via BUILD_CAMS for older recordings.
CAMS = os.environ.get(
    "BUILD_CAMS", "stereo_left,stereo_right,waist_front,hand_left,hand_right").split(",")
FPS = 30
H, W = 480, 640
PROMPT = "pick up the parts box and place it down"
MOBILE_STATE_NAMES = ["base_x", "base_y", "base_yaw", "base_v_fwd", "base_wz"]
MOBILE_ACTION_NAMES = ["base_cmd_v_fwd", "base_cmd_wz"]


def feat_stats(arr):
    a = arr.astype(np.float64).reshape(len(arr), -1)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
            "count": [len(a)]}


def img_stats(ds, n):
    from PIL import Image
    out = {}
    idxs = [0, n // 2, n - 1]
    for c in CAMS:
        imgs = np.stack([
            np.asarray(Image.open(f"{ds}/frames/{c}/frame_{i:06d}.jpg")) for i in idxs
        ]).astype(np.float64) / 255.0
        ch = imgs.reshape(-1, 3)
        out[f"observation.images.{c}"] = {
            "min": [[[v]] for v in ch.min(0)], "max": [[[v]] for v in ch.max(0)],
            "mean": [[[v]] for v in ch.mean(0)], "std": [[[v]] for v in ch.std(0)],
            "count": [len(idxs)]}
    return out


def encode_videos(ds, out_root, ep_idx, n):
    for c in CAMS:
        out_dir = f"{out_root}/videos/chunk-000/observation.images.{c}"
        os.makedirs(out_dir, exist_ok=True)
        out = f"{out_dir}/episode_{ep_idx:06d}.mp4"
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", f"{ds}/frames/{c}/frame_%06d.jpg", "-frames:v", str(n),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out],
            capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
            raise RuntimeError(f"ffmpeg failed {c} ep{ep_idx}: {r.stderr[:300]}")


def episode_arrays(data, meta):
    state = data["state"].astype(np.float32)
    action = data["action"].astype(np.float32)
    state_names = list(meta.get("state_joint_names", []))
    action_names = list(meta.get("action_names", state_names))
    if {"base", "base_velocity", "base_action"}.issubset(set(data.files)):
        base_state = np.concatenate([
            data["base"].astype(np.float32),
            data["base_velocity"].astype(np.float32),
        ], axis=1)
        state = np.concatenate([state, base_state], axis=1)
        action = np.concatenate([action, data["base_action"].astype(np.float32)], axis=1)
        state_names = state_names + list(meta.get("base_state_names", MOBILE_STATE_NAMES))
        action_names = action_names + list(meta.get("base_action_names", MOBILE_ACTION_NAMES))
    return state, action, state_names, action_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="text file: one episode dir per line (# comment ok)")
    ap.add_argument("--out", required=True, help="output dataset root (HF_LEROBOT_HOME/<repo_id>)")
    ap.add_argument("--root", default=".", help="prefix for relative episode dirs")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    dirs = []
    for ln in open(args.list):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        dirs.append(os.path.join(args.root, ln))

    out = args.out
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(f"{out}/meta", exist_ok=True)
    os.makedirs(f"{out}/data/chunk-000", exist_ok=True)

    ep_lines, stat_lines = [], []
    task_table = {}          # prompt string -> task_index (per-episode meta["prompt"], 2026-07-14)
    global_index = 0
    total_frames = 0
    state_names = None
    action_names = None
    state_dim = None
    action_dim = None
    kept = 0

    for ep_idx, ds in enumerate(dirs):
        meta = json.load(open(f"{ds}/meta.json"))
        ep_prompt = meta.get("prompt", PROMPT)
        task_idx = task_table.setdefault(ep_prompt, len(task_table))
        data = np.load(f"{ds}/episode_data.npz", allow_pickle=False)
        state, action, ep_state_names, ep_action_names = episode_arrays(data, meta)
        n = len(state)
        assert state.ndim == 2 and action.ndim == 2 and len(action) == n, (
            f"{ds} bad dims state={state.shape} action={action.shape}")
        assert np.isfinite(state).all() and np.isfinite(action).all(), f"{ds} non-finite"
        assert len(ep_state_names) == state.shape[1], f"{ds} state names/dim mismatch"
        assert len(ep_action_names) == action.shape[1], f"{ds} action names/dim mismatch"
        # verify frames exist for all 3 cams
        for c in CAMS:
            cnt = len([f for f in os.listdir(f"{ds}/frames/{c}") if f.endswith(".jpg")])
            assert cnt >= n, f"{ds} cam {c} has {cnt} < {n} frames"
        if state_names is None:
            state_names = ep_state_names
            action_names = ep_action_names
            state_dim = state.shape[1]
            action_dim = action.shape[1]
        else:
            assert ep_state_names == state_names, f"{ds} state names differ"
            assert ep_action_names == action_names, f"{ds} action names differ"
            assert state.shape[1] == state_dim and action.shape[1] == action_dim, f"{ds} dims differ"

        ts = (np.arange(n) / FPS).astype(np.float32)
        table = pa.table({
            "observation.state": pa.array(list(state), type=pa.list_(pa.float32(), state_dim)),
            "action": pa.array(list(action), type=pa.list_(pa.float32(), action_dim)),
            "timestamp": pa.array(ts),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, ep_idx, dtype=np.int64)),
            "index": pa.array(np.arange(global_index, global_index + n, dtype=np.int64)),
            "task_index": pa.array(np.full(n, task_idx, dtype=np.int64)),
        })
        pq.write_table(table, f"{out}/data/chunk-000/episode_{ep_idx:06d}.parquet")
        encode_videos(ds, out, ep_idx, n)

        ep_lines.append({"episode_index": ep_idx, "tasks": [ep_prompt], "length": n})
        stats = {"observation.state": feat_stats(state), "action": feat_stats(action),
                 "timestamp": feat_stats(ts.reshape(-1, 1))}
        for k in ("frame_index", "episode_index", "index", "task_index"):
            stats[k] = feat_stats(np.asarray(table[k]).reshape(-1, 1))
        stats.update(img_stats(ds, n))
        stat_lines.append({"episode_index": ep_idx, "stats": stats})

        global_index += n
        total_frames += n
        kept += 1
        print(f"[{ep_idx:3d}] {ds}  n={n}  cum={total_frames}", flush=True)

    feats = {}
    for c in CAMS:
        feats[f"observation.images.{c}"] = {
            "dtype": "video", "shape": [H, W, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": FPS, "video.height": H, "video.width": W,
                     "video.channels": 3, "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                     "has_audio": False}}
    feats["observation.state"] = {
        "dtype": "float32", "shape": [state_dim], "names": {"motors": state_names}
    }
    feats["action"] = {
        "dtype": "float32", "shape": [action_dim], "names": {"motors": action_names}
    }
    for k, dt in (("timestamp", "float32"), ("frame_index", "int64"),
                  ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
        feats[k] = {"dtype": dt, "shape": [1], "names": None}

    info = {
        "codebase_version": "v2.1", "robot_type": "agibot_g2_omnipicker",
        "total_episodes": kept, "total_frames": total_frames, "total_tasks": max(1, len(task_table)),
        "total_videos": kept * len(CAMS), "total_chunks": 1, "chunks_size": 5000,
        "fps": FPS, "splits": {"train": f"0:{kept}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feats,
    }
    json.dump(info, open(f"{out}/meta/info.json", "w"), indent=2)
    with open(f"{out}/meta/tasks.jsonl", "w") as f:
        for tp, ti in sorted(task_table.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": ti, "task": tp}) + "\n")
    with open(f"{out}/meta/episodes.jsonl", "w") as f:
        for e in ep_lines:
            f.write(json.dumps(e) + "\n")
    with open(f"{out}/meta/episodes_stats.jsonl", "w") as f:
        for s in stat_lines:
            f.write(json.dumps(s) + "\n")
    print(f"\n[DONE] {kept} episodes, {total_frames} frames -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
