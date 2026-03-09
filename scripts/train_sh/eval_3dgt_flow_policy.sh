#!/bin/bash

# Activate conda environment
source ~/automate/miniconda3/bin/activate
conda activate rlgpu

# Go into isaacgymenvs
export PYTHONPATH=$(pwd):$PYTHONPATH
cd isaacgymenvs

python eval_3dgt_flow_policy.py \
        task=AutoMateTaskDisassemble \
        task.env.overwrite_subassemblies=True \
        task.env.desired_subassemblies=['asset_00028'] \
        headless=True \
        seed=0
cd ..