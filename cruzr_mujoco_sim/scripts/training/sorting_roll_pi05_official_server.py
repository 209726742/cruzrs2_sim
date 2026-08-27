#!/usr/bin/env python3
"""Serve official LeRobot π0.5 inference for Sorting Roll fixed-seed evaluation."""

import argparse
import logging
import random
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for site_packages in (PROJECT_ROOT / "envs/mjx/lib").glob("python*/site-packages"):
    sys.path.append(str(site_packages))
from openpi_client import msgpack_numpy
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from src.lerobot.configs.policies import PreTrainedConfig
from src.lerobot.policies.factory import get_policy_class, make_pre_post_processors


IMAGE_MAP = {
    "observation/image": "observation.images.stereo_left",
    "observation/left_wrist_image": "observation.images.left_wrist_realsense",
    "observation/right_wrist_image": "observation.images.right_wrist_realsense",
}
STATE_DIM = 18
ACTION_SHAPE = (50, 18)
IMAGE_SHAPE = (224, 224, 3)
DEFAULT_POLICY_SEED = 28000


def observation_to_batch(observation):
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
        batch[target_key] = (
            torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        )
    return batch


class OfficialPI05Adapter:
    def __init__(self, checkpoint, device):
        self.checkpoint = checkpoint.resolve()
        try:
            self.checkpoint_step = int(self.checkpoint.parent.name)
        except ValueError as exc:
            raise ValueError(
                f"checkpoint parent must be a numeric step directory: {self.checkpoint}"
            ) from exc
        self.device = device
        self.config = PreTrainedConfig.from_pretrained(
            self.checkpoint, local_files_only=True
        )
        if self.config.type != "pi05":
            raise ValueError(f"expected a pi05 checkpoint, got {self.config.type!r}")
        self.config.device = device
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}},
        )
        policy_class = get_policy_class(self.config.type)
        self.policy = policy_class.from_pretrained(
            self.checkpoint,
            config=self.config,
            local_files_only=True,
            strict=True,
        )
        if self.policy.__class__.__name__ != "PI05Policy":
            raise TypeError(f"unexpected official policy class {self.policy.__class__.__name__}")
        self.lock = threading.Lock()

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    @torch.inference_mode()
    def infer(self, observation):
        with self.lock:
            batch = self.preprocessor(observation_to_batch(observation))
            normalized_actions = self.policy.predict_action_chunk(batch)
            actions = self.postprocessor(normalized_actions)
        if tuple(actions.shape) != (1, *ACTION_SHAPE):
            raise ValueError(f"official PI05 returned shape {tuple(actions.shape)}")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("official PI05 returned NaN/Inf")
        return {
            "actions": actions[0].detach().cpu().numpy().astype(np.float32, copy=False)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8742)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.25)
    parser.add_argument("--default-policy-seed", type=int, default=DEFAULT_POLICY_SEED)
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA inference requested, but CUDA is unavailable")
        if not 0.0 < args.cuda_memory_fraction <= 1.0:
            raise ValueError("--cuda-memory-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device=args.device
        )

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    random.seed(args.default_policy_seed)
    np.random.seed(args.default_policy_seed)
    torch.manual_seed(args.default_policy_seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    adapter = OfficialPI05Adapter(args.checkpoint, args.device)
    packer = msgpack_numpy.Packer()
    metadata = msgpack_numpy.packb({
        "policy_type": adapter.config.type,
        "policy_class": adapter.policy.__class__.__name__,
        "inference_api": "PI05Policy.predict_action_chunk",
        "checkpoint": str(adapter.checkpoint),
        "checkpoint_step": adapter.checkpoint_step,
        "device": adapter.device,
        "action_shape": ACTION_SHAPE,
    })

    def handler(websocket):
        adapter.reset()
        websocket.send(metadata)
        connection_seed = None
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
                requested_seed = int(observation.pop("policy_seed", args.default_policy_seed))
                if connection_seed is None:
                    connection_seed = requested_seed
                    torch.manual_seed(connection_seed)
                    if adapter.device.startswith("cuda"):
                        torch.cuda.manual_seed_all(connection_seed)
                    logging.info("policy sampling seed set to %d", connection_seed)
                elif requested_seed != connection_seed:
                    raise ValueError(
                        f"policy_seed changed within one rollout: "
                        f"{connection_seed} -> {requested_seed}"
                    )
                response = adapter.infer(observation)
                websocket.send(packer.pack(response))
                logging.info(
                    "official PI05 inference completed in %.3f s",
                    time.perf_counter() - started,
                )
            except Exception as exc:
                logging.error("inference failed: %s\n%s", exc, traceback.format_exc())
                websocket.send(f"{type(exc).__name__}: {exc}")

    logging.info(
        "serving official %s from checkpoint step %d on ws://%s:%d",
        adapter.policy.__class__.__name__,
        adapter.checkpoint_step,
        args.host,
        args.port,
    )
    with serve(handler, args.host, args.port, compression=None, max_size=None) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
