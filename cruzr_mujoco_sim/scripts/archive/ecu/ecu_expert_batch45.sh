#!/bin/bash
# 45-episode randomized batch: 9 bays x 5 seeds, 4 parallel workers.
# Episodes: out/teleop/ecu_expert_r{row}c{col}_s{seed}/
cd "$(dirname "$0")/.."
PY=${RL_MJX_PY:-/data1/hsr/tools/miniconda3/envs/mjx/bin/python}   # override: RL_MJX_PY
GPU=${EXPERT_GPU:-3}
JOBS=out/_batch45_jobs.txt
: > "$JOBS"
for row in 0 1 2; do
  for col in 0 1 2; do
    for k in 1 2 3 4 5; do
      seed=$((row * 100 + col * 10 + k))
      echo "$row $col $seed" >> "$JOBS"
    done
  done
done

worker() {
  local wid=$1
  while true; do
    local line
    line=$(flock out/_batch45.lock -c "head -n1 $JOBS && sed -i '1d' $JOBS") || break
    [ -z "$line" ] && break
    set -- $line
    local row=$1 col=$2 seed=$3
    local name="ecu_expert_r${row}c${col}_s${seed}"
    echo "=== [w$wid] $name start $(date +%H:%M:%S) ==="
    rm -rf "out/teleop/$name"
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU TELEOP_RECORD_GPU=$GPU \
      EXPERT_OUT=$name EXPERT_BAY="$row,$col" EXPERT_SEED=$seed \
      $PY scripts/ecu_expert_record.py > "out/_batch45_${name}.log" 2>&1
    local st=$?
    local res
    res=$(grep -oE "EPISODE (PASS|FAIL)" "out/_batch45_${name}.log" | tail -1)
    echo "=== [w$wid] $name ${res:-CRASH(exit $st)} $(date +%H:%M:%S) ==="
  done
}

touch out/_batch45.lock
for w in 1 2 3 4; do worker $w & done
wait
P=$(grep -l "EPISODE PASS" out/_batch45_ecu_expert_*.log 2>/dev/null | wc -l)
echo "BATCH45 DONE: $P/45 PASS"
