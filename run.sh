#!/bin/bash

# Activate conda environment
source ~/automate/miniconda3/bin/activate
conda activate rlgpu

# Go into isaacgymenvs
cd isaacgymenvs


# MESH_DIR="/home/ubuntu/automate/IsaacGymEnvs_Assembly/assets/automate/mesh"
# source ~/automate/miniconda3/bin/activate
# # iterate through every task_id folder
# for task_id in $(ls "$MESH_DIR"); do
#     echo "========================================="
#     echo "Processing task_id: $task_id"
#     echo "========================================="

#     ###############################################
#     # 1. Run Train (IsaacGym)
#     ###############################################
#     conda activate rlgpu
#     cd ~/automate/IsaacGymEnvs_Assembly
#     source prepare.sh
#     cd isaacgymenvs
#     for seed in $(seq 8 10); do
#         python train.py \
#             task=AutoMateTaskDisassemble \
#             task.env.overwrite_subassemblies=True \
#             "task.env.desired_subassemblies=['asset_${task_id}']" \
#             headless=True \
#             seed=$seed
#     done
# done


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

for seed in $(seq 0 50); do
    echo "=== Running training with seed=$seed ==="
    python train.py \
        task=AutoMateTaskDisassemble \
        task.env.overwrite_subassemblies=True \
        task.env.desired_subassemblies=['asset_00648'] \
        headless=True \
        seed=$seed
done



# Go back
# cd ..
