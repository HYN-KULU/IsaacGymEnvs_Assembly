#!/bin/bash

export LD_PRELOAD=/usr/local/lib/libittnotify.so
source ~/automate/miniconda3/bin/activate
conda activate rlgpu

# ===== Choose GPU =====
export CUDA_VISIBLE_DEVICES=0

# ===== Torchrun =====
torchrun --master_addr 127.0.0.1 --master_port 14580 \
  --nproc_per_node 1 --nnodes 1 --node_rank 0 \
  train_insertion_net.py \
  --action_h5 /home/ubuntu/automate/IsaacGymEnvs_Assembly/processed_dataset_depth_relative_insertion_net_action.h5 \
  --init_depth_h5 /home/ubuntu/automate/IsaacGymEnvs_Assembly/processed_dataset_depth_relative_insertion_net_init_depth.h5 \
  --epochs 200 \
  --batch_size 256 \
  --lr 1e-4 \
  --warmup_steps 500 \
  --save_dir logs/automate/insertion_net_ckpt \
  --seed 0
