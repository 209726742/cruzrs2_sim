#!/usr/bin/env python3
"""Append DAgger correction episodes onto a copy of the v2 dataset -> cruzr_shelf_e2e_dagger.

Corrections (corr_*.npz + corr_*_<cam>/frames) are the REAL-PHYSICS expert grasps recorded from
the v2 policy's own drift states. They are appended as ordinary episodes so DAgger fine-tuning
sees "policy-drift-state -> recover+grasp". Each correction is duplicated REPEAT times (the whole
v2 corpus is 602 eps; a few dozen corrections would be drowned out otherwise).

Env: SRC (v2 dataset), OUT (dagger dataset), CORR (dir of corr_*.npz), REPEAT=8.
"""
import glob
import json
import os
import shutil
import subprocess

import numpy as np

SRC = os.environ.get("SRC", "/data1/hsr/.cache/huggingface/lerobot/safe_vla/cruzr_shelf_e2e_v2")
OUT = os.environ.get("OUT", "/data1/hsr/.cache/huggingface/lerobot/safe_vla/cruzr_shelf_e2e_dagger")
CORR = os.environ.get("CORR", "/data1/hsr/embod/safe_vla_factory/mujoco_teleop/out/dagger_corr")
REPEAT = int(os.environ.get("REPEAT", "8"))
CAMS = ["head_stereo_l_shelf", "hand_left_shelf", "hand_right_shelf", "chassis_front"]
FPS = 30


def chan_stats(x):
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "max": x.max(0).tolist(), "min": x.min(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(), "count": [len(x)]}


def img_stats(n):
    return {"mean": [[[0.5]]] * 3, "std": [[[0.25]]] * 3, "max": [[[1.0]]] * 3,
            "min": [[[0.0]]] * 3, "q01": [[[0.01]]] * 3, "q99": [[[0.99]]] * 3, "count": [n]}


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    shutil.copytree(SRC, OUT)
    info = json.load(open(f"{OUT}/meta/info.json"))
    info.pop("repo_id", None)
    ep_idx = info["total_episodes"]
    gidx = info["total_frames"]
    ep_lines = open(f"{OUT}/meta/episodes.jsonl").read().splitlines()
    stat_lines = open(f"{OUT}/meta/episodes_stats.jsonl").read().splitlines()
    prompt = json.loads(ep_lines[0])["tasks"][0]

    corrs = sorted(glob.glob(f"{CORR}/corr_*.npz"))
    print(f"corrections: {len(corrs)}, repeat x{REPEAT} -> {len(corrs)*REPEAT} appended episodes")
    added = 0
    for cf in corrs:
        seed = os.path.basename(cf).split("_")[1].split(".")[0]
        z = np.load(cf)
        st, ac = z["state22"].astype(np.float32), z["action18"].astype(np.float32)
        n = len(st)
        if n < 40:
            continue
        framedirs = {c: f"{CORR}/corr_{seed}_{c}" for c in CAMS}
        if not all(os.path.isdir(framedirs[c]) for c in CAMS):
            print(f"  skip {seed}: missing frames")
            continue
        for _ in range(REPEAT):
            ts = (np.arange(n) / FPS).astype(np.float32)
            table = pa.table({
                "observation.state": pa.array(list(st), type=pa.list_(pa.float32(), 22)),
                "action": pa.array(list(ac), type=pa.list_(pa.float32(), 18)),
                "timestamp": pa.array(ts),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "episode_index": pa.array(np.full(n, ep_idx, dtype=np.int64)),
                "index": pa.array(np.arange(gidx, gidx + n, dtype=np.int64)),
                "task_index": pa.array(np.zeros(n, dtype=np.int64)),
            })
            pq.write_table(table, f"{OUT}/data/chunk-000/episode_{ep_idx:06d}.parquet")
            for c in CAMS:
                vd = f"{OUT}/videos/chunk-000/observation.images.{c}"
                r = subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                     "-i", f"{framedirs[c]}/frame_%06d.jpg", "-frames:v", str(n),
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                     f"{vd}/episode_{ep_idx:06d}.mp4"], capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"ffmpeg {c} {seed}: {r.stderr[:200]}")
            ep_lines.append(json.dumps({"episode_index": ep_idx, "tasks": [prompt], "length": n}))
            stat_lines.append(json.dumps({"episode_index": ep_idx, "stats": {
                "observation.state": chan_stats(st), "action": chan_stats(ac),
                **{f"observation.images.{c}": img_stats(n) for c in CAMS}}}))
            gidx += n
            ep_idx += 1
            added += 1
    info["total_episodes"] = ep_idx
    info["total_frames"] = gidx
    info["total_videos"] = ep_idx * len(CAMS)
    info["total_chunks"] = 1
    info["chunks_size"] = max(2000, ep_idx + 1)
    info["splits"] = {"train": f"0:{ep_idx}"}
    json.dump(info, open(f"{OUT}/meta/info.json", "w"), indent=2)
    open(f"{OUT}/meta/episodes.jsonl", "w").write("\n".join(ep_lines) + "\n")
    open(f"{OUT}/meta/episodes_stats.jsonl", "w").write("\n".join(stat_lines) + "\n")
    print(f"[DONE] appended {added} correction episodes -> total {ep_idx} eps / {gidx} frames -> {OUT}")


if __name__ == "__main__":
    main()
