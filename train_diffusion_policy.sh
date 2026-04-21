# export NCCL_SOCKET_IFNAME=eno1
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1  # Disable Peer-to-Peer (common cause of hangs on older/mixed setups)
torchrun --master_addr=10.19.86.14 --master_port=14580 \
  --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  train_diffusion_policy.py \
  --data_path /home/ubuntu/automate/IsaacGymEnvs_Assembly/utils/preprocess/processed_dataset_forward_00103_0320.h5 \
  --num_action 10 \
  --obs_feature_dim 512 \
  --hidden_dim 512 \
  --batch_size 48 \
  --num_epochs 400 \
  --save_epochs 1 \
  --ckpt_dir logs/automate/diffusion_policy_forward_0320_00103 \
  --num_workers 30\
  --resume_ckpt /home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/diffusion_policy_forward_0320/policy_epoch_200.ckpt\
  --resume_epoch 200