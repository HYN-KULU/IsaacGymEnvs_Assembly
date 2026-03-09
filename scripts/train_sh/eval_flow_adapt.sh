#!/bin/bash

# Activate conda environment
source ~/automate/miniconda3/bin/activate
conda activate rlgpu
# train_tasks=(
#   00004 
# )
train_tasks=(
#  00731
# 00015
# 00648
00360
# 00681
)
# train_tasks=(
#   00004 00016 00021 00030 00074 00078 00110 00117 00133 00138
#   00141 00163 00175 00186 00187 00192 00211 00213 00255 00256
#   00271 00293 00301 00318 00319 00320 00329 00346 00388 00410
#   00417 00422 00426 00437 00444 00446 00471 00480 00499 00514
#   00537 00559 00615 00638 00649 00659 00681 00686 00700 00703
#   00768 00783 00860 01029 01041 01092 01102 01132 01136
# )
# Go into isaacgymenvs
export PYTHONPATH=$(pwd):$PYTHONPATH
cd isaacgymenvs

for task_id in "${train_tasks[@]}"; do
    echo "=============================="
    echo "Evaluating task: ${task_id}"
    echo "=============================="

    python eval_depth_relative_flow_cond_adapt.py \
        task=AutoMateTaskDisassemble \
        task.env.overwrite_subassemblies=True \
        task.env.desired_subassemblies="['asset_${task_id}']" \
        headless=True \
        seed=100
done

cd ..