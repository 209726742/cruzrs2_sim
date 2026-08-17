#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=$(cd "$SCRIPT_DIR/../.." && pwd)
ROOT=$(cd "$PKG/.." && pwd)

MJX_PY=${RL_MJX_PY:-$ROOT/envs/mjx/bin/python}
ISAAC_PY=${ISAAC_PY:-/isaac-sim/python.sh}
PALIGEMMA_PATH=${PALIGEMMA_PATH:-$ROOT/pretrained/paligemma-3b-pt-224}
PI05_PATH=${PI05_PATH:-$ROOT/pretrained/pi05_base_remapped}
FULL=0

if [[ ${1:-} == "--full" ]]; then
  FULL=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--full]" >&2
  exit 2
fi

failures=0
warnings=0

pass() { printf '[PASS] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; failures=$((failures + 1)); }
warn() { printf '[WARN] %s\n' "$1"; warnings=$((warnings + 1)); }
info() { printf '[INFO] %s\n' "$1"; }

echo "CRUZR S2 environment check"
echo "root=$ROOT"
echo "mjx_python=$MJX_PY"
echo "isaac_python=$ISAAC_PY"
echo "paligemma=$PALIGEMMA_PATH"
echo "pi05=$PI05_PATH"
echo

if [[ -x $MJX_PY ]]; then
  pass "MuJoCo Python exists"
else
  fail "MuJoCo Python is missing: $MJX_PY"
fi

if [[ -x $MJX_PY ]]; then
  if output=$(
    "$MJX_PY" - <<'PY' 2>&1
import imageio_ffmpeg
import mujoco
import numpy
import openpi_client
import pyarrow
import scipy
import websockets
from PIL import Image

assert mujoco.__version__ == "3.9.0", mujoco.__version__
print(
    f"python/mujoco={mujoco.__version__} numpy={numpy.__version__} "
    f"scipy={scipy.__version__} pyarrow={pyarrow.__version__}"
)
PY
  ); then
    pass "MuJoCo runtime dependencies: $output"
  else
    fail "MuJoCo runtime dependency import failed: $output"
  fi
fi

if [[ -x $ISAAC_PY ]]; then
  pass "Isaac Sim Python exists"
else
  fail "Isaac Sim Python is missing: $ISAAC_PY"
fi

if [[ -x $MJX_PY ]]; then
  if output=$(
    "$MJX_PY" - "$PALIGEMMA_PATH" "$PI05_PATH" <<'PY' 2>&1
import json
import os
import struct
import sys

paligemma, pi05 = sys.argv[1:]
required = [
    os.path.join(pi05, "config.json"),
    os.path.join(pi05, "model.safetensors"),
    os.path.join(pi05, "policy_preprocessor.json"),
    os.path.join(pi05, "policy_postprocessor.json"),
    os.path.join(paligemma, "model.safetensors.index.json"),
    os.path.join(paligemma, "tokenizer_config.json"),
    os.path.join(paligemma, "tokenizer.json"),
]
missing = [path for path in required if not os.path.isfile(path) or os.path.getsize(path) == 0]
if missing:
    raise SystemExit("missing files: " + ", ".join(missing))

with open(os.path.join(paligemma, "model.safetensors.index.json"), encoding="utf-8") as stream:
    index = json.load(stream)
shards = sorted(set(index["weight_map"].values()))
model_files = [os.path.join(paligemma, shard) for shard in shards]
model_files.append(os.path.join(pi05, "model.safetensors"))

for path in model_files:
    with open(path, "rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    ends = [value["data_offsets"][1] for key, value in header.items() if key != "__metadata__"]
    expected_size = 8 + header_size + max(ends, default=0)
    if expected_size != os.path.getsize(path):
        raise SystemExit(f"truncated safetensors file: {path}")

print(f"PaliGemma shards={len(shards)}, PI0.5 base=OK")
PY
  ); then
    pass "Local model files: $output"
  else
    fail "Local model validation failed: $output"
  fi
fi

if [[ -x $ISAAC_PY ]]; then
  if output=$(
    cd /tmp
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      PALIGEMMA_PATH="$PALIGEMMA_PATH" \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      "$ISAAC_PY" - "$PI05_PATH" <<'PY' 2>&1
import sys
import torch
from src.lerobot.configs.policies import PreTrainedConfig
from src.lerobot.policies.factory import make_pre_post_processors
from src.lerobot.policies.pi05.modeling_pi05 import PI05Policy

checkpoint = sys.argv[1]
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
config = PreTrainedConfig.from_pretrained(checkpoint)
preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=checkpoint)
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"gpus={torch.cuda.device_count()} policy={PI05Policy.__name__} "
    f"processors={len(preprocessor.steps)}/{len(postprocessor.steps)}"
)
PY
  ); then
    pass "PI0.5 policy runtime: $output"
  else
    fail "PI0.5 policy runtime failed: $output"
  fi
fi

if [[ -w $PKG/out ]]; then
  pass "Output directory is writable: $PKG/out"
else
  fail "Output directory is not writable: $PKG/out"
fi

if command -v docker >/dev/null 2>&1; then
  pass "Docker is available for the host-side smoke wrapper"
elif [[ -x $ISAAC_PY ]]; then
  warn "Already inside the Isaac image; run_pillar_smoke.sh is host-side and cannot launch nested Docker here"
else
  warn "Docker is unavailable, so the host-side smoke wrapper cannot run"
fi

if [[ -n ${CRUZR_POLICY_CKPT:-} ]]; then
  if [[ -d $CRUZR_POLICY_CKPT ]]; then
    pass "Trained CRUZR policy checkpoint exists: $CRUZR_POLICY_CKPT"
  else
    fail "CRUZR_POLICY_CKPT does not exist: $CRUZR_POLICY_CKPT"
  fi
else
  info "No trained checkpoint requested; set CRUZR_POLICY_CKPT to validate a rollout artifact"
fi

if (( FULL )); then
  echo
  echo "Running full project verification..."
  if output=$(
    cd "$PKG"
    PYTHONDONTWRITEBYTECODE=1 "$MJX_PY" -B -m unittest discover -s scripts/tests -p 'test_*.py' 2>&1
  ); then
    summary=$(printf '%s\n' "$output" | grep -E '^Ran [0-9]+ tests|^OK$' | tr '\n' ' ')
    pass "Current CRUZR test suite: $summary"
  else
    fail "Current CRUZR test suite failed: $output"
  fi

  if output=$(
    cd "$PKG"
    PYTHONDONTWRITEBYTECODE=1 "$MJX_PY" -B - <<'PY' 2>&1
import os
import shutil
import tempfile

import mujoco

assets = os.path.join(os.getcwd(), "assets")
descriptor, scene = tempfile.mkstemp(prefix="_envcheck_", suffix=".xml", dir=assets)
os.close(descriptor)
try:
    shutil.copyfile(os.path.join(assets, "e2e", "template_pillar_v1.xml"), scene)
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert (model.nbody, model.nmesh, model.nu) == (49, 49, 19)
    print(f"pillar={model.nbody}/{model.nmesh}/{model.nu} time={data.time:.3f}")
finally:
    if os.path.exists(scene):
        os.remove(scene)
PY
  ); then
    pass "Pillar scene compile/step: $output"
  else
    fail "Pillar scene compile/step failed: $output"
  fi
fi

echo
echo "result: failures=$failures warnings=$warnings"
if (( failures )); then
  exit 1
fi
exit 0
