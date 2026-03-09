#!/usr/bin/env bash
set -e

num_frames=100
duration=6
fps=$(python - <<EOF
print(${num_frames}/${duration})
EOF
)

ffmpeg -y \
  -framerate "${fps}" \
  -start_number 0 \
  -i "./isaacgymenvs/flow_vis/rgb_flow_%d.png" \
  -frames:v "${num_frames}" \
  -pix_fmt yuv420p \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  rgb_flow_6s.mp4
