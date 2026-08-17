#!/usr/bin/env python3
"""Serve a local LeRobot policy through the OpenPI websocket client protocol."""

from __future__ import annotations

import argparse
import logging
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from openpi_client import msgpack_numpy
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from src.lerobot.configs.policies import PreTrainedConfig
from src.lerobot.policies.factory import get_policy_class, make_pre_post_processors


IMAGE_MAP = {
    "observation/image": "observation.images.stereo_left",
    "observation/left_wrist_image": "observation.images.waist_front",
    "observation/right_wrist_image": "observation.images.chassis_front",
}
STATE_DIM = 18
ACTION_SHAPE = (50, 18)
IMAGE_SHAPE = (224, 224, 3)


def observation_to_batch(observation: dict) -> dict:
    state = np.asarray(observation.get("observation/state"), dtype=np.float32)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"observation/state must be finite with shape ({STATE_DIM},)")

    batch = {
        "observation.state": torch.from_numpy(state.copy()),
        "task": str(observation.get("prompt", "")),
    }
    for source_key, target_key in IMAGE_MAP.items():
        image = np.asarray(observation.get(source_key))
        if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
            raise ValueError(f"{source_key} must be uint8 with shape {IMAGE_SHAPE}")
        batch[target_key] = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
    return batch


class LeRobotPolicyAdapter:
    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint.resolve()
        self.config = PreTrainedConfig.from_pretrained(self.checkpoint, local_files_only=True)
        if self.config.type != "pi05":
            raise ValueError(f"expected a pi05 checkpoint, got {self.config.type!r}")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config, pretrained_path=str(self.checkpoint)
        )
        self.policy = get_policy_class(self.config.type).from_pretrained(
            self.checkpoint,
            config=self.config,
            local_files_only=True,
            strict=True,
        )
        self.lock = threading.Lock()

    @torch.inference_mode()
    def infer(self, observation: dict) -> dict:
        with self.lock:
            batch = self.preprocessor(observation_to_batch(observation))
            normalized_actions = self.policy.predict_action_chunk(batch)
            actions = self.postprocessor(normalized_actions)
        if tuple(actions.shape) != (1, *ACTION_SHAPE) or not bool(torch.isfinite(actions).all()):
            raise ValueError(f"policy returned invalid actions with shape {tuple(actions.shape)}")
        return {"actions": actions[0].numpy().astype(np.float32, copy=False)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    adapter = LeRobotPolicyAdapter(args.checkpoint)
    packer = msgpack_numpy.Packer()
    metadata = msgpack_numpy.packb(
        {
            "policy_type": adapter.config.type,
            "checkpoint": str(adapter.checkpoint),
            "action_shape": ACTION_SHAPE,
        }
    )

    def handler(websocket) -> None:
        websocket.send(metadata)
        while True:
            try:
                message = websocket.recv()
            except ConnectionClosed:
                logging.info("client disconnected")
                return
            started = time.perf_counter()
            try:
                if isinstance(message, str):
                    raise TypeError("expected a binary msgpack request")
                observation = msgpack_numpy.unpackb(message)
                response = adapter.infer(observation)
                websocket.send(packer.pack(response))
                logging.info("inference completed in %.3f s", time.perf_counter() - started)
            except Exception as exc:
                logging.error("inference failed: %s\n%s", exc, traceback.format_exc())
                websocket.send(f"{type(exc).__name__}: {exc}")

    logging.info("serving %s on ws://%s:%d", args.checkpoint, args.host, args.port)
    with serve(handler, args.host, args.port, compression=None, max_size=None) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
