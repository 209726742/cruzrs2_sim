#!/usr/bin/env python3
"""E2E dataset builder v2 — de-ambiguation + navigation oversampling.

Fixes the start-frame ambiguity that killed the v1 end-to-end policy (probe evidence: the model
predicts every navigation frame correctly EXCEPT the episode start, which it flattens 0.25->0.06,
because 'home pose + rack ahead + v=0' appears in the data both as "about to depart" (v=0.25 next)
and as "parked, about to grasp" (v=0 for the next 1000+ frames)):

  1. DE-AMBIGUATION CUT: inside every episode, find runs where BOTH the base command and the arm
     command are static for > CUT_MIN frames; keep only the first/last KEEP frames of each run.
     This removes the 'parked doing nothing' plateaus so a static-home observation no longer
     carries a 'stay still' label thousands of times.
  2. NAV OVERSAMPLING: emit every continuous navigation span as its own derived episode.
  3. SOURCE-LEVEL SPLITS: assign train/val/test from the source seed before deriving clips, so
     all clips from one demonstration remain in the same split.

Output: LeRobot v2.1 using the deployable state18/action18/3-camera contract.
Every output episode is one continuous source interval, so a 50-frame action chunk cannot cross
an internal cut.

Env: EPISODES glob, OUT root. CUT_MIN=40 KEEP=10 NAV_PAD=15 NAV_GAP=30.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
CORE_DIR = os.path.join(SCRIPTS_DIR, "core")
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, CORE_DIR)
sys.path.insert(0, HERE)

from shelf_e2e_contract import (
    ACTION_DIM,
    ACTION_NAMES,
    CHUNK_SIZE,
    FPS,
    IMAGE_SHAPE,
    STATE_NAMES,
    make_state,
)
from shelf_e2e_source import (
    SPLIT_ORDER,
    require_single_collection_profile,
    require_single_task_version,
    require_unique_seeds,
    validate_source_dir,
)
from shelf_e2e_profiles import policy_image_map

import numpy as np

TEL = os.path.join(ROOT, "out", "teleop", "shelf_e2e_dual")
CUT_MIN = int(os.environ.get("CUT_MIN", "40"))
KEEP = int(os.environ.get("KEEP", "10"))
NAV_PAD = int(os.environ.get("NAV_PAD", "15"))
NAV_GAP = int(os.environ.get("NAV_GAP", "30"))
PROMPT = "move the steel pillar to the middle shelf of the cart, then move the rubber strip to the top shelf"
MIN_SPAN_FRAMES = max(60, CHUNK_SIZE + 1)


def spans_keep(z):
    """De-ambiguation cut-list: all frames minus interiors of long double-static runs."""
    act = z["action"]
    bact = z["base_action"]
    N = len(act)
    darm = np.abs(np.diff(act, axis=0)).max(1)
    darm = np.concatenate([[1.0], darm])
    static = (np.abs(bact).max(1) < 0.01) & (darm < 0.004)
    keep = np.ones(N, bool)
    i = 0
    while i < N:
        if static[i]:
            j = i
            while j < N and static[j]:
                j += 1
            if j - i > CUT_MIN:
                keep[i + KEEP: j - KEEP] = False
            i = j
        else:
            i += 1
    return mask_to_spans(keep)


def spans_nav(z):
    """Navigation spans: base moving, padded and merged."""
    bact = z["base_action"]
    N = len(bact)
    mv = np.abs(bact).max(1) >= 0.01
    m = np.zeros(N, bool)
    idx = np.where(mv)[0]
    for i in idx:
        m[max(0, i - NAV_PAD): min(N, i + NAV_PAD + 1)] = True
    spans = mask_to_spans(m)
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] < NAV_GAP:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def mask_to_spans(mask):
    spans = []
    i = 0
    N = len(mask)
    while i < N:
        if mask[i]:
            j = i
            while j < N and mask[j]:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def policy_state_action(z):
    st = np.stack(
        [
            make_state(robot_state, base_velocity)
            for robot_state, base_velocity in zip(
                z["state"], z["base_velocity"], strict=True
            )
        ]
    )
    ac = np.concatenate([z["action"].astype(np.float32),
                         z["base_action"].astype(np.float32)], 1)
    return st, ac


def episode_specs(z):
    """Return one (variant, span_index, start, stop) per continuous output episode."""
    specs = []
    for variant, spans in (("full", spans_keep(z)), ("nav", spans_nav(z))):
        for span_index, (start, stop) in enumerate(spans):
            if stop - start >= MIN_SPAN_FRAMES:
                specs.append((variant, span_index, start, stop))
    return specs


def encode_spans(src_dir, spans, out_path, tmpdir):
    """Encode one continuous source span with a perfectly uniform 30 FPS timeline."""
    seq = os.path.join(tmpdir, "seq")
    os.makedirs(seq, exist_ok=True)
    k = 0
    for a, b in spans:
        for f in range(a, b):
            dst = os.path.join(seq, f"frame_{k:06d}.jpg")
            os.symlink(os.path.abspath(os.path.join(src_dir, f"frame_{f:06d}.jpg")), dst)
            k += 1
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
         "-i", os.path.join(seq, "frame_%06d.jpg"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out_path],
        capture_output=True, text=True)
    shutil.rmtree(seq, ignore_errors=True)
    if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"ffmpeg encode failed: {r.stderr[:200]}")


def validate_encoded_video(path, expected_frames):
    """Verify dimensions, frame count, rate, and the exact uniform frame timeline."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames:frame=best_effort_timestamp_time",
            "-of", "json", path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr[:200]}")
    probe = json.loads(result.stdout)
    stream = (probe.get("streams") or [{}])[0]
    expected_hw = IMAGE_SHAPE[:2]
    actual_hw = (stream.get("height"), stream.get("width"))
    if actual_hw != expected_hw or stream.get("r_frame_rate") != f"{FPS}/1":
        raise ValueError(f"{path}: video contract mismatch {actual_hw}/{stream.get('r_frame_rate')}")
    if int(stream.get("nb_read_frames", -1)) != expected_frames:
        raise ValueError(f"{path}: video frame count != {expected_frames}")
    pts = np.asarray(
        [float(frame["best_effort_timestamp_time"]) for frame in probe.get("frames", [])]
    )
    expected_pts = np.arange(expected_frames) / FPS
    if pts.shape != expected_pts.shape or not np.allclose(pts, expected_pts, atol=1e-5, rtol=0):
        raise ValueError(f"{path}: video PTS is not a uniform {FPS} FPS grid")


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = os.environ.get("OUT", "/data1/hsr/.cache/huggingface/lerobot/safe_vla/cruzr_shelf_e2e_dual_v2")
    dirs = sorted(glob.glob(os.environ.get("EPISODES", os.path.join(TEL, "shelf_e2e_dual_*"))))
    dirs = [d for d in dirs if os.path.isdir(d)]
    sources = []
    rejected = []
    for d in dirs:
        source, errors = validate_source_dir(d)
        if errors:
            rejected.append({"path": os.path.abspath(d), "errors": errors})
        else:
            sources.append(source)
    print(f"source episodes: {len(sources)} accepted / {len(dirs)} found")
    for item in rejected[:10]:
        print(f"  REJECT {os.path.basename(item['path'])}: {'; '.join(item['errors'])}")
    if len(rejected) > 10:
        print(f"  ... and {len(rejected) - 10} more rejected sources")
    if not sources:
        raise RuntimeError("no source episode passed the final quality gate")
    require_unique_seeds(sources)
    task_version = require_single_task_version(sources)
    collection_profile, cameras = require_single_collection_profile(sources)
    image_map = policy_image_map(collection_profile)
    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    sources.sort(key=lambda source: (split_rank[source["split"]], source["seed"], source["path"]))
    if os.path.exists(out):
        raise FileExistsError(f"OUT already exists; choose a new dataset root: {out}")
    os.makedirs(f"{out}/meta", exist_ok=True)
    os.makedirs(f"{out}/data/chunk-000", exist_ok=True)

    st_names = list(STATE_NAMES)
    ac_names = list(ACTION_NAMES)
    ep_lines, stat_lines, source_lines = [], [], []
    gidx = 0
    ep_idx = 0
    total = 0
    kept_frames = 0
    src_frames = 0
    nav_frames = 0
    split_ranges = {}
    diversity_mode_counts = {}
    layout_mode_counts = {}
    tmpd = tempfile.mkdtemp(prefix="e2ev2_")

    for source in sources:
        ds = source["path"]
        split = source["split"]
        diversity_mode = source["diversity_mode"]
        layout_mode = source["layout_mode"]
        diversity_mode_counts[diversity_mode] = (
            diversity_mode_counts.get(diversity_mode, 0) + 1
        )
        layout_mode_counts[layout_mode] = layout_mode_counts.get(layout_mode, 0) + 1
        split_ranges.setdefault(split, [ep_idx, ep_idx])
        with np.load(os.path.join(ds, "episode_data.npz"), allow_pickle=False) as z:
            st, ac = policy_state_action(z)
            specs = episode_specs(z)
        if st.shape != (source["num_frames"], len(STATE_NAMES)):
            raise ValueError(f"{ds}: policy state shape {st.shape} does not match the contract")
        if ac.shape != (source["num_frames"], ACTION_DIM):
            raise ValueError(f"{ds}: policy action shape {ac.shape} does not match the contract")
        if not np.isfinite(st).all() or not np.isfinite(ac).all():
            raise ValueError(f"{ds}: policy state/action contains NaN/Inf")
        if not specs:
            raise ValueError(f"{ds}: no continuous span has at least {MIN_SPAN_FRAMES} frames")

        src_frames += source["num_frames"]
        derived = []
        for tag, span_index, start, stop in specs:
            n = stop - start
            s, a_ = st[start:stop], ac[start:stop]
            if tag == "full":
                kept_frames += n
            else:
                nav_frames += n
            ts = (np.arange(n) / FPS).astype(np.float32)
            table = pa.table({
                "observation.state": pa.array(list(s), type=pa.list_(pa.float32(), s.shape[1])),
                "action": pa.array(list(a_), type=pa.list_(pa.float32(), a_.shape[1])),
                "timestamp": pa.array(ts),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "episode_index": pa.array(np.full(n, ep_idx, dtype=np.int64)),
                "index": pa.array(np.arange(gidx, gidx + n, dtype=np.int64)),
                "task_index": pa.array(np.zeros(n, dtype=np.int64)),
            })
            pq.write_table(table, f"{out}/data/chunk-000/episode_{ep_idx:06d}.parquet")
            for c in cameras:
                vd = f"{out}/videos/chunk-000/observation.images.{c}"
                os.makedirs(vd, exist_ok=True)
                video_path = f"{vd}/episode_{ep_idx:06d}.mp4"
                encode_spans(os.path.join(ds, "frames", c), [(start, stop)],
                             video_path, tmpd)
                validate_encoded_video(video_path, n)
            ep_lines.append(json.dumps({
                "episode_index": ep_idx,
                "tasks": [PROMPT],
                "length": n,
                "source_seed": source["seed"],
                "source_split": split,
                "source_task_version": source["task_version"],
                "source_collection_profile": source["collection_profile"],
                "source_diversity_mode": diversity_mode,
                "source_layout_mode": layout_mode,
                "variant": tag,
                "source_span": [start, stop],
            }))
            stat_lines.append(json.dumps({
                "episode_index": ep_idx,
                "stats": {"observation.state": chan_stats(s), "action": chan_stats(a_),
                          **{f"observation.images.{c}": img_stats(n) for c in cameras}}}))
            derived.append(ep_idx)
            gidx += n
            total += n
            ep_idx += 1
        split_ranges[split][1] = ep_idx
        source_lines.append(json.dumps({
            "source_seed": source["seed"],
            "task_version": source["task_version"],
            "collection_profile": source["collection_profile"],
            "diversity_mode": diversity_mode,
            "layout_mode": layout_mode,
            "diversity": source["diversity"],
            "split": split,
            "path": ds,
            "source_frames": source["num_frames"],
            "derived_episode_indices": derived,
            "derived_episode_count": len(derived),
        }))
        print(
            f"[{ep_idx}] seed={source['seed']} split={split} "
            f"clips={len(derived)} cum={total}",
            flush=True,
        )

    dataset_splits = {
        name: f"{split_ranges[name][0]}:{split_ranges[name][1]}"
        for name in SPLIT_ORDER if name in split_ranges
    }
    info = {
        "codebase_version": "v2.1", "robot_type": "cruzr_s2",
        "source_task_version": task_version,
        "collection_profile": collection_profile,
        "policy_image_map": image_map,
        "total_episodes": ep_idx, "total_frames": total, "total_tasks": 1, "total_videos": ep_idx * len(cameras),
        "total_source_episodes": len(sources),
        "source_diversity_mode_counts": diversity_mode_counts,
        "source_layout_mode_counts": layout_mode_counts,
        "total_chunks": 1, "chunks_size": max(2000, ep_idx + 1), "fps": FPS,
        "splits": dataset_splits,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            **{f"observation.images.{c}": {
                "dtype": "video", "shape": [224, 224, 3], "names": ["height", "width", "channel"],
                "info": {"video.fps": FPS, "video.height": 224, "video.width": 224,
                         "video.channels": 3, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                         "video.is_depth_map": False, "has_audio": False}} for c in cameras},
            "observation.state": {"dtype": "float32", "shape": [len(STATE_NAMES)], "names": st_names},
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": ac_names},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    json.dump(info, open(f"{out}/meta/info.json", "w"), indent=2)
    open(f"{out}/meta/tasks.jsonl", "w").write(json.dumps({"task_index": 0, "task": PROMPT}) + "\n")
    open(f"{out}/meta/episodes.jsonl", "w").write("\n".join(ep_lines) + "\n")
    open(f"{out}/meta/episodes_stats.jsonl", "w").write("\n".join(stat_lines) + "\n")
    open(f"{out}/meta/source_episodes.jsonl", "w").write("\n".join(source_lines) + "\n")
    open(f"{out}/meta/rejected_sources.jsonl", "w").write("\n".join(json.dumps(x) for x in rejected))
    shutil.rmtree(tmpd, ignore_errors=True)
    print(f"[DONE] {ep_idx} episodes, {total} frames -> {out}")
    print(f"  源帧 {src_frames}  剪辑后全程帧 {kept_frames} (剪去 {100 - kept_frames / src_frames * 100:.0f}%)"
          f"  导航过采样帧 {nav_frames} (运动帧占比 ~{(nav_frames * 2) / total * 100:.0f}%)")


def chan_stats(x):
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "max": x.max(0).tolist(), "min": x.min(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(), "count": [len(x)]}


def img_stats(n):
    return {"mean": [[[0.5]]] * 3, "std": [[[0.25]]] * 3, "max": [[[1.0]]] * 3,
            "min": [[[0.0]]] * 3, "q01": [[[0.01]]] * 3, "q99": [[[0.99]]] * 3, "count": [n]}


if __name__ == "__main__":
    main()
