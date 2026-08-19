#!/usr/bin/env bash
set -euo pipefail

SORTING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$SORTING_DIR/.." && pwd)
PYTHON_BIN=${RL_MJX_PY:-$WORKSPACE_ROOT/envs/mjx/bin/python}
SCENE_TOOL=$WORKSPACE_ROOT/cruzr_mujoco_sim/scripts/core/sorting_roll_scene.py
GENERATED_SCENE=$WORKSPACE_ROOT/cruzr_mujoco_sim/assets/sorting_roll_scene.xml
MODE=${1:-check}

if [[ ! -x $PYTHON_BIN ]]; then
  echo "MuJoCo Python 不存在或不可执行: $PYTHON_BIN" >&2
  exit 1
fi

case "$MODE" in
  build)
    "$PYTHON_BIN" "$SCENE_TOOL" --build-only
    ;;
  check)
    "$PYTHON_BIN" "$SCENE_TOOL"
    ;;
  preview)
    PREVIEW_PATH=$WORKSPACE_ROOT/cruzr_mujoco_sim/out/sorting_roll/scene_preview.png
    MUJOCO_GL=${MUJOCO_GL:-egl} "$PYTHON_BIN" "$SCENE_TOOL" --render "$PREVIEW_PATH"
    echo "初始预览图: $PREVIEW_PATH"
    echo "目标预览图: ${PREVIEW_PATH%.png}_target.png"
    ;;
  view)
    "$PYTHON_BIN" "$SCENE_TOOL" --build-only
    TELEOP_SCENE_XML=$GENERATED_SCENE \
      MUJOCO_GL=${MUJOCO_GL:-glfw} \
      "$PYTHON_BIN" "$WORKSPACE_ROOT/cruzr_mujoco_sim/scripts/core/cruzr_teleop.py"
    ;;
  *)
    echo "用法: $0 {build|check|preview|view}" >&2
    exit 2
    ;;
esac
