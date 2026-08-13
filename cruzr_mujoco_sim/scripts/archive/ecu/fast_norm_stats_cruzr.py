#!/usr/bin/env python3
"""Compute openpi norm stats for the G2 carton dataset directly from parquet state/action.

Equivalent to scripts/compute_norm_stats.py but skips video decode (norm stats only need
state + action, which are raw 16-dim at the norm-stats stage — no model padding/delta applied).
Writes the same NormStats format openpi training expects under
  <openpi>/assets/pi05_cruzr_ecu_lora/safe_vla/cruzr_ecu_transport/norm_stats.json
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "/data1/hsr/openpi-main/src")
import pyarrow.parquet as pq  # noqa: E402

import openpi.shared.normalize as normalize  # noqa: E402
import openpi.training.config as _config  # noqa: E402

CONFIG = "pi05_cruzr_ecu_lora"
DATA_ROOT = "/data1/hsr/.cache/huggingface/lerobot/safe_vla/cruzr_ecu_transport/data/chunk-000"


def main():
    files = sorted(glob.glob(f"{DATA_ROOT}/episode_*.parquet"))
    assert files, "no parquet files"
    st = normalize.RunningStats()
    ac = normalize.RunningStats()
    nrows = 0
    H = 16   # model action horizon: stats must cover the chunk-delta distribution
    for f in files:
        t = pq.read_table(f, columns=["observation.state", "action"])
        s = np.stack(t["observation.state"].to_pylist()).astype(np.float32)   # (n,21)
        a = np.stack(t["action"].to_pylist()).astype(np.float32)              # (n,18)
        # match the policy pipeline (2026-07-13): state 21->18 (drop base x/y/yaw),
        # actions -> DELTA for the 14 arm joints over the whole action chunk.
        s18 = np.concatenate([s[:, :16], s[:, 19:21]], axis=1)
        n = len(a)
        for k in range(H):
            ak = a[k:n].copy()
            ak[:, :14] -= s18[: n - k, :14]
            ac.update(ak)
        st.update(s18)
        nrows += n
    norm_stats = {"state": st.get_statistics(), "actions": ac.get_statistics()}

    cfg = _config.get_config(CONFIG)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    out = cfg.assets_dirs / dc.repo_id
    normalize.save(out, norm_stats)
    print(f"rows={nrows} episodes={len(files)}")
    print(f"state.mean={np.round(norm_stats['state'].mean,3)}")
    print(f"state.std ={np.round(norm_stats['state'].std,3)}")
    print(f"action.mean={np.round(norm_stats['actions'].mean,3)}")
    print(f"action.std ={np.round(norm_stats['actions'].std,3)}")
    print(f"Wrote stats to: {out}")


if __name__ == "__main__":
    main()
