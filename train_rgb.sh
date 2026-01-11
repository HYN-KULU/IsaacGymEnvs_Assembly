export LD_PRELOAD=/usr/local/lib/libittnotify.so
source ~/automate/miniconda3/bin/activate
conda activate rlgpu
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
torchrun --master_addr 127.0.0.1 --master_port 14548 \
  --nproc_per_node 1 --nnodes 1 --node_rank 0 \
  train_diffusion_policy_rgb.py \
  --data_path processed_dataset_rgb_relative_actions_1004.h5 \
  --num_action 10 \
  --obs_feature_dim 512 \
  --hidden_dim 512 \
  --batch_size 256 \
  --num_epochs 1000 \
  --save_epochs 50 \
  --ckpt_dir logs/automate/diffusion_policy_ckpt_rgb_relative_actions_1004.h5\
  --obs_dir data/rgb_relative_1004
