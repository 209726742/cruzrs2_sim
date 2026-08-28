#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/real_data}
JOBS=${JOBS:-4}

extract_one() {
  local archive=$1
  local episode_dir visual_dir
  episode_dir=$(dirname "$archive")
  visual_dir=$episode_dir/$(basename "${archive%.tar}")

  if [[ -f $visual_dir/.extract_complete ]]; then
    printf 'skip complete %s\n' "$archive"
    return
  fi

  tar --skip-old-files -xf "$archive" -C "$episode_dir"
  for camera in \
    sensor_camera_stereo_color_raw \
    sensor_camera_wrist_left_color_raw \
    sensor_camera_wrist_right_color_raw; do
    compgen -G "$visual_dir/image_data/$camera/*.jpg" >/dev/null || {
      printf 'missing JPEG stream %s in %s\n' "$camera" "$archive" >&2
      return 1
    }
  done
  touch "$visual_dir/.extract_complete"
  printf 'expanded %s\n' "$archive"
}

if [[ ${1:-} == --one ]]; then
  [[ $# -eq 2 ]] || { printf 'usage: %s --one ARCHIVE\n' "$0" >&2; exit 2; }
  extract_one "$2"
  exit
fi

mapfile -d '' archives < <(
  find "$DATA_ROOT" -type f -name '*_v.tar' \
    ! -path '*/episode3_fail/*' \
    ! -path '*/episode11/*' \
    ! -path '*/episode23_zhedie/*' \
    -print0
)
[[ ${#archives[@]} -eq 23 ]] || {
  printf 'expected 23 selected visual archives, found %s\n' "${#archives[@]}" >&2
  exit 1
}

printf '%s\0' "${archives[@]}" | xargs -0 -n 1 -P "$JOBS" "$0" --one
