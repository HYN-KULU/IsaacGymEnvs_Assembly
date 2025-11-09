export LD_PRELOAD=/usr/local/lib/libittnotify.so
# export NCCL_SOCKET_IFNAME=eno1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
source ~/automate/miniconda3/bin/activate
conda activate rlgpu
export CUDA_VISIBLE_DEVICES=0
torchrun --master_addr 127.0.0.1 --master_port 14580 \
  --nproc_per_node 1 --nnodes 1 --node_rank 0 \
  train_diffusion_policy.py \
  --data_path processed_dataset_depth_relative_scale_1005.h5 \
  --num_action 10 \
  --obs_feature_dim 512 \
  --hidden_dim 512 \
  --batch_size 256 \
  --num_epochs 1000 \
  --save_epochs 50 \
  --ckpt_dir logs/automate/diffusion_policy_depth_relative_scale_dagger_1005_ckpt\
  --num_workers 8
