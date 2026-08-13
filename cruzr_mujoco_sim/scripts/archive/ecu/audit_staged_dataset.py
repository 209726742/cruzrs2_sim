#!/usr/bin/env python3
"""Quality audit of ALL stage sub-episodes in the training set (2026-07-15).

Checks per episode:
  integrity : row/frame counts, finite values, ~30fps timestamps
  semantics : start/end states match the STAGE DEFINITION
      S1 pick-from-stand   : starts gripper OPEN,  ends HELD + arm near carry pose
      S2 place-on-fixture  : starts HELD,          ends RELEASED
      S3 pick-from-fixture : starts gripper OPEN,  ends HELD + arm near carry pose
      S4 insert-into-rack  : starts HELD,          ends RELEASED
  diversity : per-stage pairwise trajectory distance (R-arm signature)
  alignment : per-stage START R-arm pose distribution (mean/std) - the reference the
              FSM handover must match at rollout time.
Prints a per-stage summary + writes review sheets (3 key frames x every episode).
"""
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
Q_CARRY = np.array([-0.358, -0.156, 0.371, -1.721, -1.114, 0.024, -1.223])

groups = {1: [], 2: [], 3: [], 4: []}
g3 = sorted(glob.glob(os.path.join(ROOT, "out/teleop/ecu_g3_s*")))
for e in g3:
    groups[1].append((e, "g3"))
for e in sorted(glob.glob(os.path.join(ROOT, "out/teleop/ecu_f3_s*_st?"))) + \
         sorted(glob.glob(os.path.join(ROOT, "out/teleop/ecu_fn_s*_st?"))):
    groups[int(e[-1])].append((e, "split"))

report = {}
for st, eps in groups.items():
    bad = []
    sigs, starts, ends = [], [], []
    n_ok = 0
    for e, src in eps:
        try:
            meta = json.load(open(os.path.join(e, "meta.json")))
            if not meta.get("success"):
                continue
            d = np.load(os.path.join(e, "episode_data.npz"))
            n = meta["num_frames"]
            stt, act, ts = d["state"], d["action"], d["timestamp"]
            probs = []
            if not (len(stt) == len(act) == len(ts) == n):
                probs.append("count")
            k = len(glob.glob(os.path.join(e, "frames/top_head/*.jpg")))
            if k != n:
                probs.append(f"jpg {k}!={n}")
            if not (np.isfinite(stt).all() and np.isfinite(act).all()):
                probs.append("nonfinite")
            g = stt[:, 15]
            qR0, qR1 = stt[0, 7:14], stt[-1, 7:14]
            if st in (1, 3):
                if g[0] < 0.9:
                    probs.append(f"start not open ({g[0]:.2f})")
                if g[-1] > 0.6:
                    probs.append(f"end not held ({g[-1]:.2f})")
                if src == "split" and np.abs(qR1 - Q_CARRY).max() > 0.45:
                    probs.append(f"end far from carry ({np.abs(qR1-Q_CARRY).max():.2f})")
            else:
                if g[0] > 0.6:
                    probs.append(f"start not held ({g[0]:.2f})")
                if g[-1] < 0.9:
                    probs.append(f"end not released ({g[-1]:.2f})")
            if probs:
                bad.append((os.path.basename(e), probs))
            else:
                n_ok += 1
                picks = np.linspace(0, n - 1, 8).astype(int)
                sigs.append(stt[picks, 7:14].flatten())
                starts.append(qR0)
                ends.append(qR1)
        except Exception as exc:  # noqa: BLE001
            bad.append((os.path.basename(e), [f"exc {exc}"]))
    S = np.array(starts) if starts else np.zeros((0, 7))
    X = np.array(sigs) if sigs else np.zeros((0, 56))
    mind = -1.0
    if len(X) >= 2:
        idx = np.random.default_rng(0).choice(len(X), size=min(len(X), 300), replace=False)
        D = np.linalg.norm(X[idx, None] - X[None, idx], axis=2)
        np.fill_diagonal(D, np.inf)
        mind = float(np.median(D.min(axis=1)))
    report[st] = dict(total=len(eps), ok=n_ok, bad=bad, start_mean=S.mean(0) if len(S) else None,
                      start_std=S.std(0) if len(S) else None, div=mind)

for st in (1, 2, 3, 4):
    r = report[st]
    print(f"\n=== STAGE {st}: {r['ok']}/{r['total']} pass audit ===")
    if r["start_mean"] is not None:
        print(f"  start qR mean: {np.round(r['start_mean'],3).tolist()}")
        print(f"  start qR std : {np.round(r['start_std'],3).tolist()}")
    print(f"  trajectory diversity (median NN dist): {r['div']:.3f}")
    for name, probs in r["bad"][:8]:
        print(f"  BAD {name}: {probs}")
    if len(r["bad"]) > 8:
        print(f"  ... and {len(r['bad'])-8} more bad")

# review sheets: 3 key frames per episode, stages 2-4 (stage1 was eyeballed before)
os.makedirs(os.path.join(ROOT, "out/_staged_review"), exist_ok=True)
for st in (2, 3, 4):
    tiles = []
    for e, src in groups[st]:
        try:
            meta = json.load(open(os.path.join(e, "meta.json")))
            if not meta.get("success"):
                continue
            n = meta["num_frames"]
            row = []
            for t in (0, n // 2, n - 3):
                im = Image.open(os.path.join(e, f"frames/top_head/frame_{t:06d}.jpg")).resize((180, 135))
                dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, 116, 12], fill=(0, 0, 0))
                dr.text((2, 1), f"{os.path.basename(e)[7:]} f{t}", fill=(255, 255, 80))
                row.append(np.asarray(im))
            tiles.append(np.concatenate(row, axis=1))
        except Exception:
            continue
    for si in range(0, len(tiles), 60):
        img = np.concatenate(tiles[si:si + 60], axis=0)
        Image.fromarray(img).save(os.path.join(ROOT, f"out/_staged_review/st{st}_{si//60:02d}.jpg"), quality=80)
    print(f"stage {st}: review sheets x{(len(tiles)+59)//60} written")
