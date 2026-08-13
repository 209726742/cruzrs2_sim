#!/usr/bin/env python3
"""立柱线（shelf_e2e）推理 server：把 Baseline (LeRobot PyTorch π0.5) checkpoint
包装成 openpi-client websocket 协议，供 shelf_e2e_rollout.py 连接。

必须在 ghrc_2026 docker 容器内运行（需要 torch + Baseline 代码），PYTHONPATH 指向 Baseline 根目录。

与 rollout 侧的契约（对齐 shelf_e2e_rollout.py 的 obs 构造）:
  输入 obs dict:
    "observation/state":            (18,) float32   16 臂/爪 + 底盘速度2
    "observation/image":            (224,224,3) u8  ← head_stereo_l_shelf (CAMS[0])
    "observation/left_wrist_image": 同上            ← chassis_front (CAMS[1], 驾驶视角)
    "observation/right_wrist_image":同上            ← hand_right_shelf (CAMS[2])
    "prompt": str
  返回: {"actions": (chunk,18) float32}   14 臂关节目标 + 2 夹爪指令(张开度) + 2 底盘速度

注意: 立柱线是端到端单策略（连底盘也由策略驱动），与 ECU 的混合式不同；
状态是 18 维，不包含 MuJoCo 才能直接获得的物体/料车真值。

环境变量:
  POLICY_CKPT    必填，指向 <run>/checkpoints/XXXXXX/pretrained_model
  POLICY_PORT    默认 8731（对齐 shelf_e2e_rollout.py 默认端口）
  POLICY_DEVICE  默认 cuda
"""

import os
import traceback
import sys

import numpy as np
import torch

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)

from shelf_e2e_contract import (  # noqa: E402
    ACTION_DIM,
    CHUNK_SIZE,
    POLICY_IMAGE_MAP,
    STATE_DIM,
    validate_action_chunk,
    validate_policy_observation,
)
CKPT = os.environ["POLICY_CKPT"]
PORT = int(os.environ.get("POLICY_PORT", "8731"))
DEVICE = os.environ.get("POLICY_DEVICE", "cuda")

from openpi_client import msgpack_numpy  # noqa: E402
import websockets.sync.server  # noqa: E402

from src.lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from src.lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402

# rollout(openpi 三槽命名) -> 训练数据集的特征名
# 与 ECU 不同：立柱线的"驾驶视角"在 left_wrist 槽（chassis_front）
_IMG_MAP = POLICY_IMAGE_MAP


def build_policy():
    policy = PI05Policy.from_pretrained(CKPT)
    state_shape = tuple(policy.config.input_features["observation.state"].shape)
    action_shape = tuple(policy.config.output_features["action"].shape)
    image_keys = {
        key for key in policy.config.input_features if key.startswith("observation.images.")
    }
    if state_shape != (STATE_DIM,):
        raise ValueError(f"checkpoint state contract is {state_shape}, expected ({STATE_DIM},)")
    if action_shape != (ACTION_DIM,):
        raise ValueError(f"checkpoint action contract is {action_shape}, expected ({ACTION_DIM},)")
    if image_keys != set(_IMG_MAP.values()):
        raise ValueError(
            f"checkpoint camera keys are {sorted(image_keys)}, expected {sorted(_IMG_MAP.values())}"
        )
    if policy.config.chunk_size != CHUNK_SIZE:
        raise ValueError(f"checkpoint chunk_size is {policy.config.chunk_size}, expected {CHUNK_SIZE}")
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=CKPT)
    return policy, pre, post


def infer(policy, pre, post, obs: dict) -> dict:
    validate_policy_observation(obs)
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
    return {"actions": validate_action_chunk(actions[0])}


def main() -> None:
    policy, pre, post = build_policy()
    print(f"[server] pillar ckpt={CKPT} device={DEVICE} port={PORT}", flush=True)

    packer = msgpack_numpy.Packer()

    def handler(ws):
        ws.send(packer.pack({"model": "lerobot-pi05", "task": "pillar", "ckpt": CKPT}))
        print(f"[server] client connected: {ws.remote_address}", flush=True)
        for message in ws:
            try:
                obs = msgpack_numpy.unpackb(message)
                reply = infer(policy, pre, post, obs)
                ws.send(packer.pack(reply))
            except Exception:
                traceback.print_exc()
                ws.send(traceback.format_exc())

    with websockets.sync.server.serve(handler, "0.0.0.0", PORT, compression=None, max_size=None) as srv:
        print(f"[server] listening on 0.0.0.0:{PORT}", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
