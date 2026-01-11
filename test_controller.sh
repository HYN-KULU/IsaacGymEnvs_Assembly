#!/bin/bash

# Activate conda environment
source ~/automate/miniconda3/bin/activate
conda activate rlgpu

# Go into isaacgymenvs
export PYTHONPATH=$(pwd):$PYTHONPATH
cd isaacgymenvs


for seed in $(seq 0 49); do
    echo "=== Running training with seed=$seed ==="
    python test_controller.py \
        task=AutoMateTaskDisassemble \
        task.env.overwrite_subassemblies=True \
        task.env.desired_subassemblies=['asset_00042'] \
        headless=True \
        seed=$seed
done

# for seed in $(seq 0 49); do
#     echo "=== Running training with seed=$seed ==="
#     python test_controller.py \
#         task=AutoMateTaskDisassemble \
#         task.env.overwrite_subassemblies=True \
#         task.env.desired_subassemblies=['asset_00028'] \
#         headless=True \
#         seed=$seed
# done

# for seed in $(seq 0 49); do
#     echo "=== Running training with seed=$seed ==="
#     python test_controller.py \
#         task=AutoMateTaskDisassemble \
#         task.env.overwrite_subassemblies=True \
#         task.env.desired_subassemblies=['asset_00030'] \
#         headless=True \
#         seed=$seed
# done

# for seed in $(seq 0 49); do
#     echo "=== Running training with seed=$seed ==="
#     python test_controller.py \
#         task=AutoMateTaskDisassemble \
#         task.env.overwrite_subassemblies=True \
#         task.env.desired_subassemblies=['asset_00042'] \
#         headless=True \
#         seed=$seed
# done

# for seed in $(seq 0 49); do
#     echo "=== Running training with seed=$seed ==="
#     python test_controller.py \
#         task=AutoMateTaskDisassemble \
#         task.env.overwrite_subassemblies=True \
#         task.env.desired_subassemblies=['asset_00110'] \
#         headless=True \
#         seed=$seed
# done

# for seed in $(seq 0 49); do
#     echo "=== Running training with seed=$seed ==="
#     python test_controller.py \
#         task=AutoMateTaskDisassemble \
#         task.env.overwrite_subassemblies=True \
#         task.env.desired_subassemblies=['asset_00681'] \
#         headless=True \
#         seed=$seed
# done

# cd ..