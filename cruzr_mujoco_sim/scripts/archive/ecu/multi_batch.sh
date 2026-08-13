#!/bin/bash
# Night batch: EXPERT_STAGE=multi runs -> st2/st3/st4 sub-episodes, W parallel workers.
# Usage: bash multi_batch.sh <seed_lo> <seed_hi> [workers]
cd "$(dirname "$0")/.."
PY=${RL_MJX_PY:-/data1/hsr/tools/miniconda3/envs/mjx/bin/python}   # override: RL_MJX_PY
LO=${1:?seed_lo}; HI=${2:?seed_hi}; W=${3:-6}
BAYS=("0,0" "0,1" "0,2" "1,0" "1,2" "2,0" "2,1" "2,2")   # skip (1,1): jig2 occupies it
run_one() {
  sd=$1
  bay=${BAYS[$((sd % 8))]}
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 TELEOP_RECORD_GPU=2 TELEOP_HOME=droop \
  EXPERT_STAGE=multi EXPERT_SEED=$sd EXPERT_BAY=$bay EXPERT_OUT=ecu_m4_s$sd \
    timeout 2400 $PY scripts/ecu_expert_record.py > /dev/null 2>&1
  ok=0
  for st in 2 3 4; do
    [ -f "out/teleop/ecu_m4_s${sd}_st${st}/meta.json" ] && \
      grep -q '"success": true' "out/teleop/ecu_m4_s${sd}_st${st}/meta.json" && ok=$((ok+1))
  done
  echo "seed $sd bay $bay : $ok/3 stages ok"
}
export -f run_one 2>/dev/null || true
ACTIVE=0
for sd in $(seq $LO $HI); do
  run_one $sd &
  ACTIVE=$((ACTIVE+1))
  if [ $ACTIVE -ge $W ]; then wait -n; ACTIVE=$((ACTIVE-1)); fi
done
wait
echo "BATCH DONE $LO..$HI"
