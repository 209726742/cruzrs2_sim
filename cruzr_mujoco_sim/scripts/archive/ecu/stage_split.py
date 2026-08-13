#!/usr/bin/env python3
"""Split full-task episodes into 4 stage sub-episodes with stage-specific prompts
(sub-goal decomposition, 2026-07-14). Also down-samples base-transit frames (1-in-3)
inside each stage, replacing trim_transit.py in the new pipeline.

Stage boundaries from the recorded right-gripper open fraction + base speed:
  S1 pick-from-stand   : [0, first base motion after close1)
  S2 place-on-fixture  : [S1end, first base motion after open1)
  S3 pick-from-fixture : [S2end, first base motion after close2)
  S4 insert-into-rack  : [S3end, end)
Each sub-episode gets a 6-frame overlap head so stage transitions are covered.

Usage: stage_split.py <glob-root...>   (e.g. out/teleop/ecu_f3_s* out/teleop/ecu_fn_s*)
"""
import glob
import json
import os
import sys

import numpy as np

STRIDE = 3
OVERLAP = 6
PROMPTS = [
    "pick up the ECU from the stand",
    "carry the ECU and place it on the fixture",
    "pick up the ECU from the fixture",
    "carry the ECU and insert it into the rack bay",
]


def first_motion_after(speed, i0):
    idx = np.where(speed[i0:] > 0.03)[0]
    return int(i0 + idx[0]) if len(idx) else len(speed)


def split_one(src):
    meta = json.load(open(os.path.join(src, "meta.json")))
    if not meta.get("success"):
        return 0
    d = np.load(os.path.join(src, "episode_data.npz"))
    n = meta["num_frames"]
    g = d["state"][:, 15]
    speed = np.linalg.norm(d["base_velocity"], axis=1)

    close1 = int(np.argmax(g < 0.9))
    if close1 == 0 and g[0] >= 0.9:
        return 0
    open1 = close1 + int(np.argmax(g[close1:] > 0.95))
    close2 = open1 + int(np.argmax(g[open1:] < 0.9))
    open2 = close2 + int(np.argmax(g[close2:] > 0.95))
    if not (0 < close1 < open1 < close2 < open2 < n):
        return 0
    b1 = first_motion_after(speed, close1)
    b2 = first_motion_after(speed, open1)
    b3 = first_motion_after(speed, close2)

    # PURE-MANIPULATION mode (2026-07-14, hybrid architecture): navigation belongs to the
    # system NavTo stack (real CRUZR SDK), so each stage clip starts where the base PARKS
    # before the manipulation anchor - driving frames never reach the policy data.
    def parked_start(anchor, lo):
        t = anchor
        while t > lo and speed[t] <= 0.03:
            t -= 1                     # walk back over the parked span
        return max(lo, t + 1 - 15)     # 15-frame settle buffer after arrival

    bounds = [
        (0, b1),                                   # S1 pick from stand (starts parked)
        (parked_start(open1, b1), b2),             # S2' place on fixture (post-arrival)
        (parked_start(close2, b2), b3),            # S3' pick from fixture
        (parked_start(open2, b3), n),              # S4' insert into rack
    ]

    made = 0
    for si, (a, b) in enumerate(bounds):
        if b - a < 30:
            return made               # malformed split -> keep what we have
        dst = f"{src}_st{si+1}"
        if os.path.exists(os.path.join(dst, "meta.json")):
            made += 1
            continue
        idx = np.arange(a, b)
        # transit down-sample inside the stage
        keep = np.ones(len(idx), dtype=bool)
        run = 0
        for j, t in enumerate(idx):
            if speed[t] > 0.03:
                keep[j] = (run % STRIDE == 0)
                run += 1
            else:
                run = 0
        idx = idx[keep]
        os.makedirs(dst, exist_ok=True)
        np.savez(os.path.join(dst, "episode_data.npz"),
                 **{k: (d[k][idx] if d[k].shape[:1] == (n,) else d[k]) for k in d.files})
        # v5 录制是 5 相机且无 top_head：不再硬编码相机清单，按源目录实际内容拷贝
        for cam in sorted(os.listdir(os.path.join(src, "frames"))):
            os.makedirs(os.path.join(dst, "frames", cam), exist_ok=True)
            for new, old in enumerate(idx):
                t = os.path.join(dst, "frames", cam, f"frame_{new:06d}.jpg")
                if not os.path.exists(t):
                    os.link(os.path.join(src, "frames", cam, f"frame_{old:06d}.jpg"), t)
        m2 = dict(meta)
        m2["num_frames"] = int(len(idx))
        m2["prompt"] = PROMPTS[si]
        m2["stage"] = si + 1
        m2["stage_bounds"] = [int(a), int(b)]
        json.dump(m2, open(os.path.join(dst, "meta.json"), "w"), indent=2)
        made += 1
    return made


def main():
    roots = sys.argv[1:] or ["out/teleop/ecu_f3_s*", "out/teleop/ecu_fn_s*"]
    srcs = []
    for r in roots:
        srcs += [p for p in glob.glob(r) if "_st" not in p and "_trim" not in p]
    total = 0
    for src in sorted(srcs):
        total += split_one(src)
    print(f"stage sub-episodes created/present: {total} from {len(srcs)} full episodes")


if __name__ == "__main__":
    main()
