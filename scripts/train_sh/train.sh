#!/bin/bash
#SBATCH --job-name=train_diffusion_policy
#SBATCH --time=48:00:00
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu_qos
#SBATCH --account=dheld
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --output=automate_debug_flow_diffusion_policy.log

export TMPDIR=/tmp

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlgpu
python download_data.py --task 00681
cd /tmp
# mkdir flow_diffusion_policy
mkdir automate_00681
cd ~/automate
ls /tmp/data/flow_1206/asset_00681
python preprocess_data_flow_diffusion_policy_orchard.py

python -u preprocess_data_depth.py
torchrun --master_addr=127.0.0.1 --master_port=14580 \
  --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  train_diffusion_policy.py \
  --data_path /project/flame/yinongh/flow_diffusion_policy_00681_0108.h5 \
  --num_action 10 \
  --obs_feature_dim 512 \
  --hidden_dim 512 \
  --batch_size 256 \
  --num_epochs 2000 \
  --save_epochs 50 \
  --ckpt_dir logs/automate/debug_flow_diffusion_policy_00681_0108 \
  --num_workers 8