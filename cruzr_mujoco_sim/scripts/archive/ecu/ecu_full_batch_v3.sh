#!/bin/bash
# v3 FULL-TASK batch: 180 attempts (9 bays x 20 seeds), 4 workers, PASS -> ecu_f3_s<seed>
cd "$(dirname "$0")/.."
PY=${RL_MJX_PY:-/data1/hsr/tools/miniconda3/envs/mjx/bin/python}   # override: RL_MJX_PY
GPU=${EXPERT_GPU:-3}
JOBS=out/_f3_jobs.txt
: > "$JOBS"
i=0
for row in 0 1 2; do for col in 0 1 2; do
  for k in $(seq 1 20); do
    echo "$((20000 + i)) $row,$col" >> "$JOBS"; i=$((i+1))
  done
done; done
worker() {
  while true; do
    local line
    line=$(flock out/_f3.lock -c "head -n1 $JOBS && sed -i '1d' $JOBS") || break
    [ -z "$line" ] && break
    local s=${line%% *}; local bay=${line##* }
    local name="ecu_f3_s${s}"
    rm -rf "out/teleop/$name"
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU TELEOP_RECORD_GPU=$GPU \
      EXPERT_OUT=$name EXPERT_SEED=$s EXPERT_BAY=$bay \
      $PY scripts/ecu_expert_record.py > "out/_f3_${name}.log" 2>&1
    local res=$(grep -oE 'EPISODE (PASS|FAIL)' "out/_f3_${name}.log" | tail -1)
    [ "$res" = "EPISODE FAIL" ] && rm -rf "out/teleop/$name"
    echo "[$name bay=$bay] ${res:-CRASH}"
  done
}
touch out/_f3.lock
for w in 1 2 3 4; do worker & done
wait
echo "F3 BATCH DONE: $(ls -d out/teleop/ecu_f3_s* 2>/dev/null | wc -l) PASS kept"
