#!/usr/bin/env python3
"""Env worker for shelf_e2e_rl_train.py -- runs in the `mjx` conda env, NOT the openpi venv.

The two halves of the trainer need different Python environments: the actor needs openpi/JAX, the
env needs MuJoCo 3.9 (the scene uses `actuatorfrcrange`, which the openpi venv's older mujoco
rejects with a schema error). Rather than mutate the shared openpi venv, each env is its own OS
process launched with the mjx interpreter, talking to the trainer over a UNIX socket.

Protocol: 4-byte big-endian length prefix + pickle. ("reset", phase) -> (obs, seed);
("step", chunk) -> (obs, reward, done, info); ("close", None) -> exit.
"""
import argparse
import os
import pickle
import socket
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ndarray codec: the two interpreters have different numpy majors (mjx=2.x, openpi venv=1.x) and
# numpy's own pickles are not portable across that boundary ("No module named numpy._core").
# Encode arrays as (tag, dtype-str, shape, raw bytes) instead.
_ND = "__nd__"


def enc(o):
    if isinstance(o, np.ndarray):
        return (_ND, o.dtype.str, o.shape, o.tobytes())
    if isinstance(o, dict):
        return {k: enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(enc(v) for v in o)
    if isinstance(o, np.generic):
        return o.item()
    return o


def dec(o):
    if isinstance(o, tuple) and len(o) == 4 and o[0] == _ND:
        return np.frombuffer(o[3], dtype=np.dtype(o[1])).reshape(o[2])
    if isinstance(o, dict):
        return {k: dec(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(dec(v) for v in o)
    return o


def send(sock, obj):
    b = pickle.dumps(enc(obj), protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(b)) + b)


def recv(sock):
    hdr = b""
    while len(hdr) < 4:
        c = sock.recv(4 - len(hdr))
        if not c:
            return None
        hdr += c
    n = struct.unpack(">I", hdr)[0]
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(min(1 << 20, n - len(buf)))
        if not c:
            return None
        buf += c
    return dec(pickle.loads(bytes(buf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--phase", default="A")
    ap.add_argument("--eps-per-seed", type=int, default=5)
    ap.add_argument("--snap-dir", default="")
    a = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import shelf_e2e_rlenv as E

    pool = [int(x) for x in a.seeds.split(",") if x]
    rng = np.random.default_rng(pool[0])
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(a.sock)

    env, cur, n_ep = None, None, 0
    while True:
        msg = recv(s)
        if msg is None or msg[0] == "close":
            break
        cmd, data = msg
        if cmd == "reset":
            if env is None or n_ep % a.eps_per_seed == 0:
                cur = int(rng.choice(pool))
                env = E.PillarRLEnv(seed=cur, snap_dir=a.snap_dir or None)
            n_ep += 1
            send(s, (env.reset(a.phase), cur))
        elif cmd == "step":
            o, r, d, info = env.step(data)
            send(s, (o, float(r), bool(d),
                     {"term": info["term"], "stage": int(info["stage"]),
                      "ret": float(info["ret"]), "latches": info["latches"]}))
    s.close()


if __name__ == "__main__":
    main()
