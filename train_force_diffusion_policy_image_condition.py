import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataset.force_diffusion_policy_image_condition import DepthActionDataset
from policy.force_diffusion_policy_image_condition import Diffusion_Policy   # the class we wrote
from diffusion_utils.training import set_seed, plot_history, sync_loss   # keep your helpers


# -------------------
# Default args
# -------------------
default_args = edict({
    "data_path": "processed_dataset.h5",
    "num_action": 10,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "proprio_dim": 9,
    "action_dim": 9,          # (3 delta + 6 rot6d)
    "ckpt_dir": "logs/policy_ckpt",
    "resume_ckpt": None,
    "resume_epoch": -1,
    "lr": 5e-3, # 5e-4
    "batch_size": 128,
    "num_epochs": 100,
    "save_epochs": 10,
    "num_workers": 24,
    "seed": 233,
})



# -------------------
# Training loop
# -------------------
def train(args_override):
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # distributed setup
    torch.multiprocessing.set_sharing_strategy('file_system')
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    os.environ['NCCL_P2P_DISABLE'] = '1'
    dist.init_process_group(
        backend="nccl", init_method="env://",
        world_size=WORLD_SIZE, rank=RANK
    )

    # device
    set_seed(args.seed)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device("cuda", LOCAL_RANK)

    # dataset
    if RANK == 0:
        print("Loading dataset ...")
    dataset = DepthActionDataset(args.data_path)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size // WORLD_SIZE,
        num_workers=args.num_workers,
        sampler=sampler,
        drop_last=True
    )

    # model
    if RANK == 0:
        print("Loading policy ...")
    policy = Diffusion_Policy(
        num_action=args.num_action,
        obs_feature_dim=args.obs_feature_dim,
        action_dim=args.action_dim,
        proprio_dim=args.proprio_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    print("Finish Loading Policy")
    if args.resume_ckpt is not None:
        policy.load_state_dict(torch.load(args.resume_ckpt, map_location=device), strict=False)
        if RANK == 0:
            print(f"Checkpoint {args.resume_ckpt} loaded.")
    print("Starting DDP ...")
    policy = nn.parallel.DistributedDataParallel(
        policy,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        find_unused_parameters=True
    )
    print("Finish DDP")
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        betas=[0.95, 0.999],
        weight_decay=1e-6
    )

    # # optimizer + scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=2000,
        num_training_steps=len(dataloader) * args.num_epochs
    )
    lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1

    # training
    train_history = []
    policy.train()
    print("Training started ...")
    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        sampler.set_epoch(epoch)
        # if RANK == 0:
        print(f"Epoch {epoch}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: current LR = {current_lr:.6e}")
        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader) if RANK == 0 else dataloader
        avg_loss = 0

        for batch in pbar:
            depth = batch["depth"].to(device)               # (B, 1, H, W)
            proprio = batch["proprioception"].to(device)    # (B, 9)
            actions = batch["actions"].to(device)           # (B, K, 9)
            forces = batch["force"].to(device)             # (B, K, 3)
            socket_depth = batch["socket_depth"].to(device) # (B, H, W)
            init_plug_photo_depth = batch["init_plug_photo_depth"].to(device) #
            loss = policy(depth=depth, proprioception=proprio, actions=actions, force=forces, socket_depth=socket_depth, init_plug_photo_depth=init_plug_photo_depth)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            avg_loss += loss.item()
            if RANK == 0:
                pbar.set_description(f"Loss {loss.item():.4f}")

        avg_loss /= num_steps
        #sync_loss(avg_loss, device)
        train_history.append(avg_loss)

        if RANK == 0:
            print(f"Train loss: {avg_loss:.6f}")
            if (epoch + 1) % args.save_epochs == 0:
                state_dict = policy.module.state_dict()
                ckpt_path = os.path.join(args.ckpt_dir, f"policy_epoch_{epoch+1}.ckpt")
                os.makedirs(args.ckpt_dir, exist_ok=True)
                torch.save(state_dict, ckpt_path)
                plot_history(train_history, epoch, args.ckpt_dir, args.seed)

    if RANK == 0:
        torch.save(policy.module.state_dict(), os.path.join(args.ckpt_dir, "policy_last.ckpt"))


# -------------------
# Main
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="processed_dataset.h5", help="HDF5 dataset path")
    parser.add_argument('--num_action', type=int, default=10)
    parser.add_argument('--obs_feature_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--proprio_dim', type=int, default=7)
    parser.add_argument('--action_dim', type=int, default=9)
    parser.add_argument('--ckpt_dir', type=str, default="logs/policy_ckpt")
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--resume_epoch', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=233)
    train(vars(parser.parse_args()))