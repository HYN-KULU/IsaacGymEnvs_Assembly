cd /home/ubuntu/automate/IsaacGymEnvs_Assembly
source ~/automate/miniconda3/bin/activate
conda activate rlgpu
source prepare.sh
torchrun --master_addr 10.19.86.14 --master_port 14580 \
  --nproc_per_node 1 --nnodes 1 --node_rank 0 \
  train_3dgt_mask_flow.py \
  --epochs 200 \
  --batch_size 2 \
  --lr 1e-5 \
  --warmup_steps 500 \
  --save_dir logs/automate/masked_3dgt_flow_net_mask_adapt_00731_fewshot \
  --seed 0 