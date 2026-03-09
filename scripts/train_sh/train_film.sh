#!/bin/bash
set -e

############################################
# Environment
############################################
export LD_PRELOAD=/usr/local/lib/libittnotify.so

# 如果你们 NCCL 网络接口稳定，可以解开
# export NCCL_SOCKET_IFNAME=eno1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

source ~/automate/miniconda3/bin/activate
conda activate rlgpu

############################################
# GPU visibility (单机 1 卡)
############################################
export CUDA_VISIBLE_DEVICES=0

############################################
# Distributed config（按你现在的写法）
############################################
MASTER_ADDR=10.19.80.237
MASTER_PORT=14548

NNODES=1
NODE_RANK=0            # ⚠️ 每个节点要不一样
NPROC_PER_NODE=1

############################################
# Train
############################################
torchrun \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT} \
  --nnodes ${NNODES} \
  --node_rank ${NODE_RANK} \
  --nproc_per_node ${NPROC_PER_NODE} \
  train_film_flow_net.py \
  --pretrained_ckpt logs/automate/flow_net_multitask_ckpt/multi_2_epoch_100.pt \
  --batch_size 8 \
  --epochs 20 \
  --lr 1e-4 \
  --warmup_steps 200 \
  --save_dir logs/automate/film_flow_bottleneck_ckpt
