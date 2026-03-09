#!/bin/bash

# Activate conda environment
source ~/automate/miniconda3/bin/activate
conda activate rlgpu

# Go into isaacgymenvs
export PYTHONPATH=$(pwd):$PYTHONPATH
cd isaacgymenvs
for seed in $(seq 101 160); do
    source ~/automate/IsaacGymEnvs_Assembly/prepare.sh
    echo "=== Running training with seed=$seed ==="
    python roll_out_recovery.py \
        task=AutoMateTaskDisassemble \
        task.env.overwrite_subassemblies=True \
        task.env.desired_subassemblies=['asset_00681'] \
        headless=True \
        seed=$seed
done

cd ..
