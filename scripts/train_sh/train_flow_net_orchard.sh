#!/bin/bash
#SBATCH --job-name=train_diffusion_policy
#SBATCH --time=32:00:00
#SBATCH --partition=flame
#SBATCH --qos=flame-16gpu-b_qos
#SBATCH --account=dheld
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=automate_diffusion_policy_insertion_net.log

export TMPDIR=/tmp

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlgpu
cd /tmp
mkdir flow_net
cd ~/automate
python preprocess_data_flow_mask.py
torchrun --master_addr 127.0.0.1 --master_port 14580 \
  --nproc_per_node 8 --nnodes 1 --node_rank 0 \
  train_flow_net.py \
  --epochs 400 \
  --batch_size 32 \
  --lr 1e-4 \
  --warmup_steps 500 \
  --save_dir logs/automate/flow_net_ckpt \
  --seed 0