#!/bin/bash
# s2b: merge full-task (55) + grasp-densification (45) episodes, rebuild dataset,
# recompute norm stats, retrain 10k steps as cruzr_ecu_s2b on GPU3.
set -e
cd "$(dirname "$0")/.."
TELEOP=out/teleop
LIST=out/_ecu_dataset_list_b.txt
OPENPI=${OPENPI_ROOT:-/data1/hsr/openpi-main}   # override: OPENPI_ROOT
OPY=${OPENPI_PY:-/data1/hsr/tools/miniconda3/envs/openpi_env/bin/python}   # override: OPENPI_PY

: > "$LIST"
for d in "$TELEOP"/_mujoco_rec "$TELEOP"/ecu_expert_r* "$TELEOP"/ecu_grasp_s*; do
    [ -f "$d/meta.json" ] || continue
    ok=$(python3 -c "import json;print(json.load(open('$d/meta.json'))['success'])")
    [ "$ok" = "True" ] && echo "$(realpath "$d")" >> "$LIST"
done
N=$(wc -l < "$LIST"); echo "episodes: $N"
[ "$N" -ge 90 ] || { echo "too few"; exit 1; }

OUT="$HOME/.cache/huggingface/lerobot/safe_vla/cruzr_ecu_transport"
rm -rf "$OUT"
$OPY "$(dirname "$0")/build_carton_lerobot.py" \
    --list "$LIST" --out "$OUT" --root /

# fast_norm_stats writes to <cwd>/assets (cfg.assets_dirs is relative!) - run from a fixed dir
PKG="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd "$PKG"
$OPY scripts/fast_norm_stats_cruzr.py
cp assets/pi05_cruzr_ecu_lora/safe_vla/cruzr_ecu_transport/norm_stats.json \
   "$OPENPI"/assets/pi05_cruzr_ecu_lora/safe_vla/cruzr_ecu_transport/
cd - >/dev/null

cd "$OPENPI"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=3 \
  $OPY scripts/train.py pi05_cruzr_ecu_lora --exp-name=cruzr_ecu_s2b \
    --num-train-steps=10000 --save-interval=2000 --overwrite
