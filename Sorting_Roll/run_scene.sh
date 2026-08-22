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
    echo "目标物理对照图: ${PREVIEW_PATH%.png}_target_physics.png"
    echo "初始预览图: $PREVIEW_PATH"
    echo "目标预览图: ${PREVIEW_PATH%.png}_target.png"
    ;;
  view)
    "$PYTHON_BIN" "$SCENE_TOOL" --build-only
    VIEWER_MODE=${TELEOP_VIEWER:-egl}
    case "$VIEWER_MODE" in
      egl)
        if ! "$PYTHON_BIN" -c 'import cv2' >/dev/null 2>&1; then
          echo "EGL 可视化缺少依赖: $PYTHON_BIN -m pip install 'numpy<2' opencv-python==4.11.0.86" >&2
          exit 1
        fi
        GL_BACKEND=egl
        ;;
      passive|glfw)
        GL_BACKEND=glfw
        ;;
      *)
        echo "不支持的 TELEOP_VIEWER: $VIEWER_MODE (可选: egl, passive, glfw)" >&2
        exit 2
        ;;
    esac
    TELEOP_SCENE_XML=$GENERATED_SCENE \
      TELEOP_VIEWER=$VIEWER_MODE \
      TELEOP_EGL_FAST=${TELEOP_EGL_FAST:-1} \
      EGL_W=${EGL_W:-1280} \
      EGL_H=${EGL_H:-720} \
      TELEOP_FPS=${TELEOP_FPS:-60} \
      MUJOCO_GL=$GL_BACKEND \
      "$PYTHON_BIN" "$WORKSPACE_ROOT/cruzr_mujoco_sim/scripts/core/cruzr_teleop.py"
    ;;
  *)
    echo "用法: $0 {build|check|preview|view}" >&2
    exit 2
    ;;
esac
