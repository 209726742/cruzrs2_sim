#!/bin/bash
# Cross-check: does the RL reward's verdict agree with the expert's OWN placed/grip_firm gates?
# Any seed where the expert says PASS but the reward says failure (or vice versa) is a reward bug.
set -u
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"   # package root
MJ=${RL_MJX_PY:-/data1/hsr/tools/miniconda3/envs/mjx/bin/python}   # override: RL_MJX_PY
GPU=${GPU:-0}
OUT=out/logs/shelf_e2e/rl_agreement.log; : > $OUT
mkdir -p out/logs/shelf_e2e out/rl/val out/rl/snap_agree
for seed in "$@"; do
  (
   res=$(SEED=$seed E2E_RLHOOK=1 E2E_NOREC=1 E2E_KICKS=0 MUJOCO_GL=egl \
         MUJOCO_EGL_DEVICE_ID=$GPU TELEOP_RECORD_GPU=$GPU \
         EXPERT_OUT=out/rl/val/_agree_$seed E2E_SNAP_DIR=out/rl/snap_agree timeout 900 \
         $MJ scripts/shelf_e2e_expert.py 2>&1)
   gate=$(echo "$res" | grep -a '\[gate\] placed' | tail -1)
   grip=$(echo "$res" | grep -a '\[gate\] grip_firm' | tail -1)
   score=$(echo "$res" | grep -a '\[rlhook\] SCORE' | tail -1 | sed -E 's/.*(expert_return=[-+0-9.]+ term=[a-z_/]+).*/\1/')
   rm -rf out/rl/val/_agree_$seed assets/e2e_scene_${seed}.xml
   echo "seed=$seed | ${gate:-<no placed gate>} | ${grip:-<no grip gate>} | ${score:-<no score>}" >> $OUT
  ) &
done
wait
sort -t= -k2 -n $OUT
