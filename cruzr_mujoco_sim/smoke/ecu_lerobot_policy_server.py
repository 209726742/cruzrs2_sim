#!/usr/bin/env python3
"""把 Baseline (LeRobot PyTorch π0.5) 的 checkpoint 包装成 openpi-client websocket 协议服务器。

供 cruzr_mujoco_sim 的 rollout 脚本（ecu_hybrid_rollout.py / ecu_policy_rollout.py）连接。
必须在 ghrc_2026 docker 容器内运行（需要 torch + Baseline 代码），PYTHONPATH 指向 Baseline 根目录。

与 rollout 侧的契约（同 openpi serve_policy.py）:
  输入 obs dict:
    "observation/state":            (21,) float32   16 臂/爪 + 5 底盘
    "observation/image":            (480,640,3) u8  ← stereo_left   (rollout POLICY_CAMS[0])
    "observation/left_wrist_image": 同上            ← stereo_right  (rollout POLICY_CAMS[1])
    "observation/right_wrist_image":同上            ← waist_front   (rollout POLICY_CAMS[2])
    "prompt": str
  返回: {"actions": (chunk,18) float32}   16 臂/爪绝对关节位置 + 2 底盘速度

注意: 训练数据集必须用 BUILD_CAMS=stereo_left,stereo_right,waist_front 构建（3 相机），
使 checkpoint 的 image_features 与推理时可用的相机完全一致。

环境变量:
  POLICY_CKPT    必填，指向 <run>/checkpoints/XXXXXX/pretrained_model
  POLICY_PORT    默认 8735
  POLICY_DEVICE  默认 cuda
"""

import os
import traceback

import numpy as np
import torch

CKPT = os.environ["POLICY_CKPT"]
PORT = int(os.environ.get("POLICY_PORT", "8735"))
DEVICE = os.environ.get("POLICY_DEVICE", "cuda")

from openpi_client import msgpack_numpy  # noqa: E402
import websockets.sync.server  # noqa: E402

from src.lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from src.lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402

# rollout(openpi 三槽命名) -> 训练数据集（3 相机）的特征名
_IMG_MAP = {
    "observation/image": "observation.images.stereo_left",
    "observation/left_wrist_image": "observation.images.stereo_right",
    "observation/right_wrist_image": "observation.images.waist_front",
}


def build_policy():
    policy = PI05Policy.from_pretrained(CKPT)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=CKPT)
    return policy, pre, post


def infer(policy, pre, post, obs: dict) -> dict:
    batch = {}
    for src, dst in _IMG_MAP.items():
        img = np.asarray(obs[src])  # HWC uint8
        batch[dst] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    state = np.asarray(obs["observation/state"], dtype=np.float32)
    batch["observation.state"] = torch.from_numpy(state)
    batch["task"] = str(obs.get("prompt", ""))

    proc = pre(batch)
    with torch.no_grad():
        actions = policy.predict_action_chunk(proc)  # (1, chunk, 18)
    actions = post(actions)  # 反归一化 -> cpu
    return {"actions": np.asarray(actions[0], dtype=np.float32)}


def main() -> None:
    policy, pre, post = build_policy()
    print(f"[server] ckpt={CKPT} device={DEVICE} port={PORT}", flush=True)

    packer = msgpack_numpy.Packer()

    def handler(ws):
        ws.send(packer.pack({"model": "lerobot-pi05", "ckpt": CKPT}))
        print(f"[server] client connected: {ws.remote_address}", flush=True)
        for message in ws:
            try:
                obs = msgpack_numpy.unpackb(message)
                reply = infer(policy, pre, post, obs)
                ws.send(packer.pack(reply))
            except Exception:
                traceback.print_exc()
                ws.send(traceback.format_exc())  # 客户端收到 str 会抛错

    with websockets.sync.server.serve(handler, "0.0.0.0", PORT, compression=None, max_size=None) as srv:
        print(f"[server] listening on 0.0.0.0:{PORT}", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
