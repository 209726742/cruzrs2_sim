#!/bin/bash
# Generate curriculum reset snapshots (+ in-sim expert reward scores) for a range of seeds.
#   usage: rl_snap_batch.sh <seed_lo> <seed_hi> <workers> <gpu_list_csv>
# 产出: out/rl/snap/snap_*.npz ；日志: out/logs/shelf_e2e/rl_snap_batch.log
set -u
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"   # package root
LO=${1:-1}; HI=${2:-200}; W=${3:-6}; GPUS=${4:-0,2}
MJ=${RL_MJX_PY:-/data1/hsr/tools/miniconda3/envs/mjx/bin/python}   # override: RL_MJX_PY
IFS=',' read -ra G <<< "$GPUS"
mkdir -p out/rl/snap out/logs/shelf_e2e out/rl/val
LOG=out/logs/shelf_e2e/rl_snap_batch.log; : > $LOG
echo "=== rl snapshots seeds $LO..$HI workers=$W gpus=$GPUS ===" | tee -a $LOG

one () {  # $1=seed $2=gpu
  local seed=$1 gpu=$2 res
  res=$(SEED=$seed E2E_RLHOOK=1 E2E_NOREC=1 E2E_KICKS=0 \
        MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu TELEOP_RECORD_GPU=$gpu \
        E2E_SNAP_DIR=out/rl/snap \
        EXPERT_OUT=out/rl/val/_snaponly_$seed timeout 900 \
        $MJ scripts/shelf_e2e_expert.py 2>&1 | grep -aE "^\[rlhook\] SCORE" | tail -1)
  rm -rf assets/e2e_scene_${seed}.xml out/rl/val/_snaponly_$seed
  if echo "$res" | grep -q "term=success"; then
    echo "OK   seed=$seed $res" >> $LOG
  else
    rm -f out/rl/snap/snap_$(printf '%06d' $seed).npz
    echo "DROP seed=$seed ${res:-<no score line>}" >> $LOG
  fi
}

i=0
for ((s=LO; s<=HI; s++)); do
  one $s ${G[$((i % ${#G[@]}))]} &
  i=$((i+1))
  while (( $(jobs -rp | wc -l) >= W )); do wait -n; done
done
wait
echo "=== done: $(grep -c '^OK' $LOG) usable snapshots ===" | tee -a $LOG
