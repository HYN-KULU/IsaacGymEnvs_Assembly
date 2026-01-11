#!/bin/bash

INPUT_DIR="iga_eval_visual"
FORWARD_VIDEO="iga_demo.mp4"
REVERSE_VIDEO="iga_demo_reverse.mp4"

# 1. Forward video
ffmpeg -framerate 10 -pattern_type glob -i "${INPUT_DIR}/*.png" \
       -c:v libx264 -pix_fmt yuv420p "${FORWARD_VIDEO}"

echo "Forward video saved to ${FORWARD_VIDEO}"

# 2. Reverse video
ffmpeg -i "${FORWARD_VIDEO}" -vf reverse -af areverse "${REVERSE_VIDEO}"

echo "Reverse video saved to ${REVERSE_VIDEO}"
