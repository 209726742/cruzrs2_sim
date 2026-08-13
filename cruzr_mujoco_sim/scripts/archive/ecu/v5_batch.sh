#!/bin/bash
# v5 night batch: five-camera recording (REC_CAMS default in cruzr_teleop).
# Usage: bash v5_batch.sh <mode: multi|grasp_only> <seed_lo> <seed_hi> [workers]
cd "$(dirname "$0")/.."
PY=${RL_MJX_PY:-/data1/hsr/tools/miniconda3/envs/mjx/bin/python}   # override: RL_MJX_PY
MODE=${1:?mode}; LO=${2:?lo}; HI=${3:?hi}; W=${4:-6}
BAYS=("0,0" "0,1" "0,2" "1,0" "1,2" "2,0" "2,1" "2,2")
run_one() {
  sd=$1
  bay=${BAYS[$((sd % 8))]}
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 TELEOP_RECORD_GPU=2 TELEOP_HOME=droop \
  EXPERT_STAGE=$MODE EXPERT_SEED=$sd EXPERT_BAY=$bay EXPERT_OUT=ecu_v5_s$sd \
    timeout 2400 $PY scripts/ecu_expert_record.py > /dev/null 2>&1
  if [ "$MODE" = "multi" ]; then
    ok=0
    for st in 2 3 4; do
      grep -q '"success": true' "out/teleop/ecu_v5_s${sd}_st${st}/meta.json" 2>/dev/null && ok=$((ok+1))
    done
    echo "seed $sd bay $bay : $ok/3"
  else
    grep -q '"success": true' "out/teleop/ecu_v5_s${sd}/meta.json" 2>/dev/null && echo "seed $sd : PASS" || echo "seed $sd : fail"
  fi
}
ACTIVE=0
for sd in $(seq $LO $HI); do
  run_one $sd &
  ACTIVE=$((ACTIVE+1))
  if [ $ACTIVE -ge $W ]; then wait -n; ACTIVE=$((ACTIVE-1)); fi
done
wait
echo "V5BATCH_DONE $MODE $LO..$HI"
