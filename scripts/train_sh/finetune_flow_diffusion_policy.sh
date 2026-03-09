torchrun --master_addr=127.0.0.1 --master_port=14580 \
  --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  train_diffusion_policy_3dflow.py \
  --data_path /home/ubuntu/automate/flow_diffusion_policy_multitask_00028_0301.h5 \
  --num_action 10 \
  --obs_feature_dim 512 \
  --hidden_dim 512 \
  --batch_size 32 \
  --num_epochs 1000 \
  --save_epochs 5 \
  --ckpt_dir logs/automate/finetune_flow_diffusion_policy_multitask_00028_0301 \
  --num_workers 24 \
  --resume_ckpt /home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/3dgt_flow_policy/policy_epoch_360.ckpt \
  --index_path /home/ubuntu/automate/IsaacGymEnvs_Assembly/data/flow_diffusion_policy/flow_diffusion_policy_adapt_index_00028.npy