#!/bin/bash

# Activate conda environment
source ~/automate/miniconda3/bin/activate
conda activate rlgpu

# Go into isaacgymenvs
export PYTHONPATH=$(pwd):$PYTHONPATH
cd isaacgymenvs

for seed in $(seq 0 100); do
    echo "=== Running training with seed=$seed ==="
    python forward_data_collector.py \
        task=AutoMateTaskDisassemble \
        task.env.overwrite_subassemblies=True \
        task.env.desired_subassemblies=['asset_00703'] \
        headless=True \
        seed=$seed
done



cd ..