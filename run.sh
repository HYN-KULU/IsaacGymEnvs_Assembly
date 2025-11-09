#!/bin/bash
export PYTHONPATH=$HOME/isaacgym_local/python
export PYTHONPATH=$(pwd):$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/isaacgym_local/python/isaacgym/_bindings/linux-x86_64
# Activate conda environment
source prepare.sh
source ~/automate/miniconda3/bin/activate
conda activate rlgpu

# Go into isaacgymenvs
cd isaacgymenvs
# great: 01053 01079 00062 00190 01102 00681  | 00860 00187 00686 00499 00007

# occlusion 00486 

# grasp pose 00417 00470 00652 (too complex) 00768

# unstable grasp 00004 00783 01026

# task confusion 00346

# for seed in $(seq 0 100); do
#     echo "=== Running training with seed=$seed ==="
#     python train.py \
#         task=AutoMateTaskDisassemble \
#         task.env.overwrite_subassemblies=True \
#         task.env.desired_subassemblies=['asset_01053'] \
#         headless=True \
#         seed=$seed
# done

# for seed in $(seq 0 0); do
#     echo "=== Running training with seed=$seed ==="
#     python train.py \
#         task=AutoMateTaskDisassemble \
#         task.env.overwrite_subassemblies=True \
#         task.env.desired_subassemblies=['asset_01079'] \
#         headless=True \
#         seed=$seed
# done

for seed in $(seq 0 100); do
    echo "=== Running training with seed=$seed ==="
    python train.py \
        task=AutoMateTaskDisassemble \
        task.env.overwrite_subassemblies=True \
        task.env.desired_subassemblies=['asset_00062'] \
        headless=True \
        seed=$seed
done

# for seed in $(seq 400 449); do
#     echo "=== Running training with seed=$seed ==="
#     python train.py \
#         task=AutoMateTaskDisassemble \
#         task.env.overwrite_subassemblies=True \
#         task.env.desired_subassemblies=['asset_00190'] \
#         headless=True \
#         seed=$seed
# done

# for seed in $(seq 450 474); do
#     echo "=== Running training with seed=$seed ==="
#     python train.py \
#         task=AutoMateTaskDisassemble \
#         task.env.overwrite_subassemblies=True \
#         task.env.desired_subassemblies=['asset_01102'] \
#         headless=True \
#         seed=$seed
# done

# for seed in $(seq 475 499); do
#     echo "=== Running training with seed=$seed ==="
#     python train.py \
#         task=AutoMateTaskDisassemble \
#         task.env.overwrite_subassemblies=True \
#         task.env.desired_subassemblies=['asset_00681'] \
#         headless=True \
#         seed=$seed
# done

cd ..
