#!/bin/bash
# v3 grasp-densification batch: 220 randomized episodes (wide spawn + path + DART noise,
# FIXED cameras), 4 workers. PASS episodes -> out/teleop/ecu_g3_s<seed>/
cd "$(dirname "$0")/.."
PY=${RL_MJX_PY:-/data1/hsr/tools/miniconda3/envs/mjx/bin/python}   # override: RL_MJX_PY
GPU=${EXPERT_GPU:-3}
JOBS=out/_g3_jobs.txt
: > "$JOBS"
for s in $(seq 10221 10280); do echo "$s" >> "$JOBS"; done
worker() {
  while true; do
    local s
    s=$(flock out/_g3.lock -c "head -n1 $JOBS && sed -i '1d' $JOBS") || break
    [ -z "$s" ] && break
    local name="ecu_g3_s${s}"
    rm -rf "out/teleop/$name"
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU TELEOP_RECORD_GPU=$GPU \
      EXPERT_OUT=$name EXPERT_SEED=$s EXPERT_STAGE=grasp_only \
      $PY scripts/ecu_expert_record.py > "out/_g3_${name}.log" 2>&1
    echo "[$name] $(grep -oE 'EPISODE (PASS|FAIL)' out/_g3_${name}.log | tail -1)"
  done
}
touch out/_g3.lock
for w in 1 2 3 4; do worker & done
wait
echo "G3 BATCH DONE: $(grep -l 'EPISODE PASS' out/_g3_ecu_g3_*.log | wc -l)/60 PASS (top-up)"
