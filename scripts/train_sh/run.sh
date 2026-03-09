#!/bin/bash

# Activate conda environment
source ~/automate/miniconda3/bin/activate
conda activate rlgpu

# Go into isaacgymenvs
cd isaacgymenvs

train_tasks=(
#   00004 00016 00021 00030 00074 00078 00110 00117 00133 00138
#   00141 00163 00175 00186 00187 00192 00211 00213 00255 00256
#   00271 00293 00301 00318 00319 00320 00329 00346 00388 00410
#   00417 00422 00426 00437 00444 00446 00471 00480 
#   00499 00514 00537 00559 00615 00638 00649 00659 00681 00686 00700 00703
#   00768 00783 00860 01029 01041 01092 01102 01132 01136
# 00559 
    00345 00360 00028 00614 00103
    00553 00731 00015 00648 00506
)

for task_id in "${train_tasks[@]}"; do
    echo "========================================="
    echo "Processing task_id: $task_id"
    echo "========================================="

    ###############################################
    # 1. Run Train (IsaacGym)
    ###############################################
    conda activate rlgpu
    cd ~/automate/IsaacGymEnvs_Assembly
    source prepare.sh
    cd isaacgymenvs
    for seed in $(seq 0 10); do
        python train.py \
            task=AutoMateTaskDisassemble \
            task.env.overwrite_subassemblies=True \
            "task.env.desired_subassemblies=['asset_${task_id}']" \
            headless=True \
            seed=$seed
    done
done


# great: 01053 01079 00190 01102 00681  | 00860 00187 00686 00499 00007

# occlusion 00486 

# grasp pose 00417 00470 00652 (too complex) 00768

# unstable grasp 00004 00783 01026

# task confusion 00346

# for seed in $(seq 100 100); do
#     echo "=== Running training with seed=$seed ==="
#     python train.py \
#         task=AutoMateTaskDisassemble \
#         task.env.overwrite_subassemblies=True \
#         task.env.desired_subassemblies=['asset_00553'] \
#         headless=True \
#         seed=$seed
# done

# for seed in $(seq 0 0); do
#     echo "=== Running training with seed=$seed ==="
#     python train.py \
#         task=AutoMateTaskDisassemble \
#         task.env.overwrite_subassemblies=True \
#         task.env.desired_subassemblies=['asset_00681'] \
#         headless=True \
#         seed=$seed
# done



# Go back
# cd ..
