#!/bin/bash

# Activate conda environment
source ~/automate/miniconda3/bin/activate
conda activate rlgpu

# Go into isaacgymenvs
cd isaacgymenvs


python eval.py \
        task=AutoMateTaskDisassemble \
        task.env.overwrite_subassemblies=True \
        task.env.desired_subassemblies=['asset_00681'] \
        headless=True \
        seed=0
