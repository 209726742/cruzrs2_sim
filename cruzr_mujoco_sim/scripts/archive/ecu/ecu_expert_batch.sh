#!/bin/bash
# Batch-record 9 ECU expert episodes, one per rack bay (3 levels x 3 columns).
cd "$(dirname "$0")/.."
PY=${RL_MJX_PY:-/data1/hsr/tools/miniconda3/envs/mjx/bin/python}   # override: RL_MJX_PY
GPU=${EXPERT_GPU:-3}
pass=0; fail=0
for row in 0 1 2; do
  for col in 0 1 2; do
    name="ecu_expert_r${row}c${col}"
    echo "=== [$name] bay row=$row col=$col ==="
    rm -rf "out/teleop/$name"
    if MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU TELEOP_RECORD_GPU=$GPU \
       EXPERT_OUT=$name EXPERT_BAY="$row,$col" \
       $PY scripts/ecu_expert_record.py > "out/_batch_${name}.log" 2>&1; then
      pass=$((pass+1)); echo "  PASS"
    else
      fail=$((fail+1)); echo "  FAIL (see out/_batch_${name}.log)"
    fi
  done
done
echo "BATCH DONE: $pass PASS, $fail FAIL"
