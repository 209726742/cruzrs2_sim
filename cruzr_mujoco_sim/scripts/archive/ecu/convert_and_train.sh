#!/bin/bash
# After batch45: build the LeRobot dataset from ALL PASS episodes (+ the user's demo),
# compute norm stats, then launch the pi0.5 stage-2 LoRA fine-tune on GPU3.
set -e
cd "$(dirname "$0")/.."
TELEOP=out/teleop
LIST=out/_ecu_dataset_list.txt
OPENPI=${OPENPI_ROOT:-/data1/hsr/openpi-main}   # override: OPENPI_ROOT
OPY=${OPENPI_PY:-/data1/hsr/tools/miniconda3/envs/openpi_env/bin/python}   # override: OPENPI_PY

echo "=== 1. episode list (success=true only) ==="
: > "$LIST"
for d in "$TELEOP"/_mujoco_rec "$TELEOP"/ecu_expert_r*; do
    [ -f "$d/meta.json" ] || continue
    ok=$(python3 -c "import json;print(json.load(open('$d/meta.json'))['success'])")
    [ "$ok" = "True" ] && echo "$(realpath "$d")" >> "$LIST"
done
N=$(wc -l < "$LIST")
echo "episodes: $N"
[ "$N" -ge 40 ] || { echo "too few episodes, aborting"; exit 1; }

echo "=== 2. build LeRobot dataset safe_vla/cruzr_ecu_transport ==="
OUT="$HOME/.cache/huggingface/lerobot/safe_vla/cruzr_ecu_transport"
rm -rf "$OUT"
$OPY "$(dirname "$0")/build_carton_lerobot.py" \
    --list "$LIST" --out "$OUT" --root /

echo "=== 3. norm stats ==="
cd "$OPENPI"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 CUDA_VISIBLE_DEVICES=3 \
  $OPY scripts/compute_norm_stats.py --config-name pi05_cruzr_ecu_lora --max-frames 20000

echo "=== 4. train (GPU3, 10k steps) ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=3 \
  $OPY scripts/train.py pi05_cruzr_ecu_lora --exp-name=cruzr_ecu_s2 \
    --num-train-steps=10000 --save-interval=2000 --overwrite
