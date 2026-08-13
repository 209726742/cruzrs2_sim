#!/usr/bin/env python3
"""Run a seed sweep of shelf_e2e_dual_expert.py and summarise result.json files.

Every run gets its own scratch output directory named after the candidate, so a sweep
can never land on an official episode name. Results are read from result.json rather
than from terminal output, which is easy to truncate and easy to misread.

Usage:
  python scripts/diagnostics/shelf_e2e_sweep.py --tag baseline --seeds 1-13
  python scripts/diagnostics/shelf_e2e_sweep.py --tag mycand --seeds 4,8,9,2 --jobs 4 --env E2E_TOUCHDOWN=0
  python scripts/diagnostics/shelf_e2e_sweep.py --tag baseline --report-only
  python scripts/diagnostics/shelf_e2e_sweep.py --diff baseline mycand

A sweep is only comparable to another sweep with the same seed set. Kicks are forced off
unless overridden, because the random base shoves otherwise mask the change under test.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
COLLECTION_DIR = os.path.join(SCRIPTS_DIR, "collection")
ROOT = os.path.dirname(SCRIPTS_DIR)
PYTHON = os.path.join(os.path.dirname(ROOT), "envs", "mjx", "bin", "python")
EXPERT = os.path.join(COLLECTION_DIR, "shelf_e2e_dual_expert.py")
SWEEP_ROOT = os.path.join(ROOT, "out", "sweeps")


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_one(tag: str, seed: int, extra_env: dict[str, str], gpu: int) -> dict:
    out_rel = os.path.join("out", "sweeps", tag, f"seed_{seed:06d}")
    out_abs = os.path.join(ROOT, out_rel)
    os.makedirs(out_abs, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "SEED": str(seed),
        "E2E_NOREC": "1",
        "E2E_KICKS": env.get("E2E_KICKS", "0"),
        "EXPERT_OUT": out_rel,
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
    })
    env.update(extra_env)
    log_path = os.path.join(out_abs, "run.log")
    with open(log_path, "w") as log:
        proc = subprocess.run([PYTHON, EXPERT], cwd=ROOT, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    result_path = os.path.join(out_abs, "result.json")
    if os.path.isfile(result_path):
        with open(result_path) as fh:
            return json.load(fh)
    return {"seed": seed, "passed": False, "first_failed_gate": "<no result.json>",
            "sim_seconds": None, "peak_grip_force_n": None,
            "crashed": True, "returncode": proc.returncode, "log": log_path}


def collect(tag: str) -> list[dict]:
    out = []
    tag_dir = os.path.join(SWEEP_ROOT, tag)
    if not os.path.isdir(tag_dir):
        return out
    for name in sorted(os.listdir(tag_dir)):
        path = os.path.join(tag_dir, name, "result.json")
        if os.path.isfile(path):
            with open(path) as fh:
                out.append(json.load(fh))
        elif os.path.isdir(os.path.join(tag_dir, name)):
            seed = int(name.split("_")[-1]) if name.startswith("seed_") else -1
            out.append({"seed": seed, "passed": False,
                        "first_failed_gate": "<no result.json>", "crashed": True})
    return sorted(out, key=lambda r: r["seed"])


def fmt_geom(res: dict, name: str) -> str:
    g = (res.get("geometry") or {}).get(name)
    if not g:
        return "-"
    flag = "in" if g["fully_on_shelf"] else "OUT"
    return f"{flag} m={g['margin_mm']:+.0f} gap={g['gap_to_surface_mm']:+.0f}"


def report(tag: str, results: list[dict]) -> None:
    if not results:
        print(f"no results under {os.path.join(SWEEP_ROOT, tag)}", file=sys.stderr)
        return
    print(f"\n=== sweep {tag} ({len(results)} seeds) ===")
    print(f"{'seed':>4} {'result':<6} {'first failed gate':<18} {'sim s':>7} "
          f"{'peakN':>7}  {'pillar':<22} {'strip':<22}")
    for r in results:
        sim = f"{r['sim_seconds']:.1f}" if r.get("sim_seconds") else "-"
        pk = f"{r['peak_grip_force_n']:.0f}" if r.get("peak_grip_force_n") else "-"
        print(f"{r['seed']:>4} {'PASS' if r['passed'] else 'FAIL':<6} "
              f"{str(r.get('first_failed_gate') or '-'):<18} {sim:>7} {pk:>7}  "
              f"{fmt_geom(r, 'pillar'):<22} {fmt_geom(r, 'strip'):<22}")

    passed = [r for r in results if r["passed"]]
    print(f"\npass {len(passed)}/{len(results)} = {100.0*len(passed)/len(results):.1f}%"
          f"   passing seeds: {[r['seed'] for r in passed] or '-'}")
    sims = [r["sim_seconds"] for r in results if r.get("sim_seconds")]
    if sims:
        print(f"mean sim {sum(sims)/len(sims):.1f}s over {len(sims)} runs")

    dist: dict[str, int] = {}
    for r in results:
        if not r["passed"]:
            dist[str(r.get("first_failed_gate"))] = dist.get(str(r.get("first_failed_gate")), 0) + 1
    if dist:
        print("failure distribution: " + ", ".join(
            f"{k} x{v}" for k, v in sorted(dist.items(), key=lambda kv: -kv[1])))

    # A candidate that "passes" while spiking hundreds of newtons is a docking collision.
    hot = [(r["seed"], r["peak_grip_force_n"]) for r in results
           if (r.get("peak_grip_force_n") or 0) >= 100.0]
    if hot:
        print("WARNING peak grip force >=100 N: " + ", ".join(f"seed {s}: {f:.0f}N" for s, f in hot))
    bad = [r["seed"] for r in results if r["passed"] and not all(
        (r.get("geometry") or {}).get(o, {}).get("fully_on_shelf", False)
        for o in ("pillar", "strip"))]
    if bad:
        print(f"WARNING passed but not fully on shelf: {bad}")


def diff(tag_a: str, tag_b: str) -> None:
    a = {r["seed"]: r for r in collect(tag_a)}
    b = {r["seed"]: r for r in collect(tag_b)}
    shared = sorted(set(a) & set(b))
    if set(a) != set(b):
        print(f"WARNING seed sets differ: only in {tag_a}: {sorted(set(a)-set(b))}, "
              f"only in {tag_b}: {sorted(set(b)-set(a))}", file=sys.stderr)
    print(f"\n=== {tag_a} -> {tag_b} ({len(shared)} shared seeds) ===")
    print(f"{'seed':>4} {tag_a[:16]:<16} {tag_b[:16]:<16} verdict")
    fixed, broke = [], []
    for s in shared:
        ra, rb = a[s], b[s]
        va = "PASS" if ra["passed"] else str(ra.get("first_failed_gate"))
        vb = "PASS" if rb["passed"] else str(rb.get("first_failed_gate"))
        verdict = ""
        if not ra["passed"] and rb["passed"]:
            verdict, _ = "FIXED", fixed.append(s)
        elif ra["passed"] and not rb["passed"]:
            verdict, _ = "BROKE", broke.append(s)
        print(f"{s:>4} {va:<16} {vb:<16} {verdict}")
    na = sum(r["passed"] for r in a.values() if r["seed"] in shared)
    nb = sum(r["passed"] for r in b.values() if r["seed"] in shared)
    print(f"\npass {na}/{len(shared)} -> {nb}/{len(shared)}   fixed {fixed}   broke {broke}")
    if nb <= na:
        print("verdict: no net gain -- do not keep this candidate on pass rate alone")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="sweep name; also the scratch directory name")
    ap.add_argument("--seeds", default="1-13")
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--diff", nargs=2, metavar=("TAG_A", "TAG_B"))
    args = ap.parse_args()

    if args.diff:
        diff(*args.diff)
        return
    if not args.tag:
        ap.error("--tag is required unless --diff is used")
    if args.report_only:
        report(args.tag, collect(args.tag))
        return

    extra_env = {}
    for item in args.env:
        key, _, value = item.partition("=")
        extra_env[key] = value
    seeds = parse_seeds(args.seeds)
    print(f"sweep {args.tag}: {len(seeds)} seeds, {args.jobs} at a time, "
          f"env {extra_env or '{}'}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, args.tag, s, extra_env, args.gpu): s for s in seeds}
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            print(f"  seed {r['seed']:>3} {'PASS' if r['passed'] else 'FAIL':<5} "
                  f"{r.get('first_failed_gate') or ''}", flush=True)
    report(args.tag, collect(args.tag))


if __name__ == "__main__":
    main()
