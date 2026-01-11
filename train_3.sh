export LD_PRELOAD=/usr/local/lib/libittnotify.so
# export NCCL_SOCKET_IFNAME=eno1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
source ~/automate/miniconda3/bin/activate
conda activate rlgpu
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --master_addr 10.19.80.237 --master_port 14548 \
  --nproc_per_node 8 --nnodes 3 --node_rank 2 \
  train_diffusion_policy.py \
  --data_path processed_dataset_depth_relative_scale_1005.h5 \
  --num_action 10 \
  --obs_feature_dim 512 \
  --hidden_dim 512 \
  --batch_size 2048 \
  --num_epochs 1500 \
  --save_epochs 50 \
  --ckpt_dir logs/automate/diffusion_policy_depth_relative_scale_dagger_1005_ckpt\
  --num_workers 8
