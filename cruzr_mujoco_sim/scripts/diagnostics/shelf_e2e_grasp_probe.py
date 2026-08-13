#!/usr/bin/env python3
"""At which frame of the demo's grasp segment do the pads actually reach the part?

The strip grasp is aligned by shifting the demo-derived target, which needs a prediction of
where the pads end up. Using the last frame of the grasp segment is wrong: that pose is
already lifted, about 136 mm above where the bar rests, so a correction computed there
drives the arm into the rack. This walks the whole segment and reports, per frame, how far
the predicted pad centre sits from the bar's nearest resting cross-section, so the contact
frame can be identified instead of guessed.

The prediction needs no rollout: transform_mount works on the demo's recorded world poses,
and the pad offset relative to the mount is fixed by the arm's geometry.

Usage: python scripts/diagnostics/shelf_e2e_grasp_probe.py [--seeds 21,22]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
EXPERT = os.path.join(SCRIPTS_DIR, "collection", "shelf_e2e_dual_expert.py")


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def probe(seed):
    os.environ["SEED"] = str(seed)
    os.environ["E2E_NOREC"] = "1"
    os.environ["EXPERT_OUT"] = f"out/sweeps/_probe_{seed}"
    src = open(EXPERT).read()
    # Everything up to the first replay helper: model, demos, geometry and transform_mount.
    # These are definitions only, so nothing steps the simulation.
    head = src.split("def approach_first(")[0]
    g = {"__name__": "probe", "__file__": EXPERT}
    exec(compile(head, "head", "exec"), g)

    d, ct, PADG = g["d"], g["ct"], g["PADG"]
    ref = g["REF"]["strip"]
    strip_xy = g["strip_xy"]
    demo_obj = np.asarray(g["VAR"][ref["variant"]]["obj"])
    offset = np.array([strip_xy[0] - demo_obj[0], strip_xy[1] - demo_obj[1],
                       -g["VAR"][ref["variant"]]["dz"]])
    offset[0] -= 0.04
    yaw_rotation = np.pi

    pad_local = {}
    for hand, arm in (("l", ct.L), ("r", ct.R)):
        mount_pos, mount_rot = d.xpos[arm.mount], d.xmat[arm.mount].reshape(3, 3)
        pad_local[hand] = [mount_rot.T @ (d.geom_xpos[gi] - mount_pos) for gi in PADG[hand]]

    def sections():
        return [ct.geom_aabb(gi) for gi in g["OBJECTS"]["strip"]["geoms"]]

    secs = sections()
    paths = ref["mount"]["grasp"]
    gs, ge = ref["gs"], ref["ge"]
    print(f"\n=== seed {seed}: grasp segment frames {gs}..{ge} ({ge - gs} frames), "
          f"grasp_close={ref['grasp_close']} (index {ref['grasp_close'] - gs}) ===")
    print(f"{'idx':>5} {'l_dx_mm':>9}{'l_dz_mm':>9}{'l_dist':>9}   "
          f"{'r_dx_mm':>9}{'r_dz_mm':>9}{'r_dist':>9}")
    rows = []
    n = len(paths["l"])
    for idx in range(n):
        row = {"idx": idx}
        for hand in ("l", "r"):
            pm, rm = paths[hand][idx]
            tp, tr = g["transform_mount"](pm, rm, offset, yaw_rotation, ref["ref_center"])
            pads = np.mean([tp + tr @ loc for loc in pad_local[hand]], axis=0)
            best = min(secs, key=lambda ab: abs(0.5 * (ab[0][1] + ab[1][1]) - pads[1]))
            tgt = 0.5 * (best[0] + best[1])
            row[hand] = (1000 * (tgt[0] - pads[0]), 1000 * (tgt[2] - pads[2]))
        rows.append(row)
    for row in rows:
        ldx, ldz = row["l"]
        rdx, rdz = row["r"]
        ld, rd = np.hypot(ldx, ldz), np.hypot(rdx, rdz)
        mark = ""
        if ld == min(np.hypot(*r["l"]) for r in rows):
            mark += "  <- left closest"
        if rd == min(np.hypot(*r["r"]) for r in rows):
            mark += "  <- right closest"
        if row["idx"] % max(1, n // 25) == 0 or mark:
            print(f"{row['idx']:>5} {ldx:>9.1f}{ldz:>9.1f}{ld:>9.1f}   "
                  f"{rdx:>9.1f}{rdz:>9.1f}{rd:>9.1f}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="21,22")
    args = ap.parse_args()
    for seed in parse_seeds(args.seeds):
        probe(seed)


if __name__ == "__main__":
    sys.exit(main())
