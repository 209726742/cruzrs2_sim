#!/bin/bash
# 45 grasp-only densification episodes, 4 workers (seeds 201-245).
cd "$(dirname "$0")/.."
PY=${RL_MJX_PY:-/data1/hsr/tools/miniconda3/envs/mjx/bin/python}   # override: RL_MJX_PY
GPU=${EXPERT_GPU:-3}
JOBS=out/_graspb_jobs.txt
: > "$JOBS"
for s in $(seq 201 245); do echo "$s" >> "$JOBS"; done
worker() {
  while true; do
    local s
    s=$(flock out/_graspb.lock -c "head -n1 $JOBS && sed -i '1d' $JOBS") || break
    [ -z "$s" ] && break
    local name="ecu_grasp_s${s}"
    rm -rf "out/teleop/$name"
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU TELEOP_RECORD_GPU=$GPU \
      EXPERT_OUT=$name EXPERT_SEED=$s EXPERT_STAGE=grasp_only \
      $PY scripts/ecu_expert_record.py > "out/_graspb_${name}.log" 2>&1
    echo "[$name] $(grep -oE 'EPISODE (PASS|FAIL)' out/_graspb_${name}.log | tail -1)"
  done
}
touch out/_graspb.lock
for w in 1 2 3 4; do worker & done
wait
echo "GRASP BATCH DONE: $(grep -l 'EPISODE PASS' out/_graspb_ecu_grasp_*.log | wc -l)/45 PASS"
