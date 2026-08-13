#!/usr/bin/env python3
"""Is 'fully on the shelf' geometrically achievable, and by how much?

The containment gate demands all four horizontal AABB edges of the part land inside the
shelf AABB. Before tuning any placement code it is worth knowing the size of the window
that leaves, because the strip is long and the shelves are not much longer. This loads
each seed's scene, measures shelf and part spans with no rollout, and reports the
centring freedom in each axis.

A negative window means the gate cannot be satisfied at any pose and the gate, not the
placement code, is what needs revisiting. Note the spans are world-axis-aligned, so a
part carried at a yaw offset presents a larger footprint than these numbers -- the
window is an upper bound that assumes the part stays axis-aligned.

Usage: python scripts/diagnostics/shelf_e2e_reach_probe.py [--seeds 1-13]
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
COLLECTION_DIR = os.path.join(SCRIPTS_DIR, "collection")
ROOT = os.path.dirname(SCRIPTS_DIR)
PYTHON = os.path.join(os.path.dirname(ROOT), "envs", "mjx", "bin", "python")

PROBE = r'''
import os, numpy as np, mujoco
os.environ["E2E_NOREC"] = "0"
import importlib.util
spec = importlib.util.spec_from_file_location("dual", os.path.join(r"{here}", "shelf_e2e_dual_expert.py"))
mod = importlib.util.module_from_spec(spec)
import builtins
_exit = builtins.SystemExit
try:
    spec.loader.exec_module(mod)
except BaseException:
    pass
'''


def parse_seeds(spec):
    seeds = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part.strip():
            seeds.append(int(part))
    return seeds


SNIPPET = r'''
import os, sys, numpy as np, mujoco
sys.argv = ["probe"]
os.environ["SEED"] = "%d"
os.environ["E2E_PROBE_ONLY"] = "1"
src = open(os.path.join(r"%s", "shelf_e2e_dual_expert.py")).read()
head = src.split("def gate(")[0]
g = {"__name__": "probe", "__file__": os.path.join(r"%s", "shelf_e2e_dual_expert.py")}
exec(compile(head, "head", "exec"), g)
m, d, ct = g["m"], g["d"], g["ct"]
mujoco.mj_forward(m, d)
out = {}
for name in ("pillar", "strip"):
    slo, shi = ct.geom_aabb(g["SHELF"][name])
    lo, hi = g["obj_extent"](name)
    out[name] = (shi - slo, hi - lo)
print("RESULT", os.environ["SEED"], " ".join(
    f"{n}:{out[n][0][0]:.4f},{out[n][0][1]:.4f},{out[n][1][0]:.4f},{out[n][1][1]:.4f}"
    for n in ("pillar", "strip")))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-13")
    args = ap.parse_args()

    print(f"{'seed':>4} {'part':<7} {'shelf x':>8} {'shelf y':>8} {'part x':>7} {'part y':>7} "
          f"{'win x':>8} {'win y':>8}  verdict")
    worst = {}
    for seed in parse_seeds(args.seeds):
        env = dict(os.environ, MUJOCO_GL="egl", SEED=str(seed))
        proc = subprocess.run([PYTHON, "-c", SNIPPET % (seed, COLLECTION_DIR, COLLECTION_DIR)],
                              cwd=ROOT, env=env, capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")), None)
        if not line:
            print(f"{seed:>4} probe failed: {proc.stderr.strip().splitlines()[-1:]}")
            continue
        for field in line.split()[2:]:
            name, nums = field.split(":")
            sx, sy, px, py = (float(v) for v in nums.split(","))
            wx, wy = sx - px, sy - py
            verdict = "impossible" if min(wx, wy) < 0 else (
                "tight" if min(wx, wy) < 0.10 else "roomy")
            print(f"{seed:>4} {name:<7} {sx:>8.3f} {sy:>8.3f} {px:>7.3f} {py:>7.3f} "
                  f"{wx:>+8.3f} {wy:>+8.3f}  {verdict}")
            key = (name,)
            if key not in worst or min(wx, wy) < worst[key]:
                worst[key] = min(wx, wy)
    print()
    for (name,), w in worst.items():
        print(f"{name}: tightest window across seeds {w:+.3f} m "
              f"(centring must be within +/-{w/2:+.3f} m of shelf centre)")


if __name__ == "__main__":
    main()
