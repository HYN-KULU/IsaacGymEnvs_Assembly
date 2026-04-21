#!/bin/bash
#SBATCH --job-name=train_diffusion_policy_forward
#SBATCH --time=32:00:00
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=dheld
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=64
#SBATCH --mem=80G
#SBATCH --output=automate_diffusion_policy_multitask_0314.log

export TMPDIR=/tmp

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlgpu
# download data
python download_data_3dgt_forward.py
cd /tmp
mkdir data
cd data
mkdir diffusion_policy
cd ~/automate

# Preprocess the flow and depth
python launcher_diffusion_policy.py

# Preprocess the actions 

# Indexing
python index_processing_diffusion_policy.py

echo "Finish Indexing the flow diffusion policy dataset"

torchrun --master_addr=127.0.0.1 --master_port=14580 \
  --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  train_diffusion_policy.py \
  --data_path /project/flame/yinongh/processed_dataset_forward_0314.h5 \
  --num_action 5 \
  --obs_feature_dim 512 \
  --hidden_dim 512 \
  --batch_size 256 \
  --num_epochs 600 \
  --save_epochs 5 \
  --ckpt_dir logs/automate/diffusion_policy_forward_0314 \
  --num_workers 24\
