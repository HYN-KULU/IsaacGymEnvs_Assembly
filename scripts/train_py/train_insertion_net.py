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

from dataset.insertion_net_dataset import DepthActionDataset
from policy.insertion_net import InsertionNet
from diffusion_utils.training import set_seed, sync_loss


def build_dataloader(args):
    dataset = DepthActionDataset(
        hdf5_file_actions=args.action_h5,
        hdf5_file_depth_init=args.init_depth_h5,
        normalize=True,
        scale_to_unit=True
    )

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True
    )

    return loader, sampler


def train_one_epoch(model, loader, optimizer, scheduler, args):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, disable=(args.local_rank != 0))

    for batch in pbar:
        depth_curr = batch["depth"].cuda(non_blocking=True)
        depth_goal = batch["init_depth"].cuda(non_blocking=True)
        actions_gt = batch["actions"].cuda(non_blocking=True)

        loss = model(depth_curr, depth_goal, actions_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # sync loss across GPUs
        loss_val = sync_loss(loss, args) if args.distributed else loss.item()
        total_loss += loss_val

        pbar.set_description(f"Loss: {loss_val:.4f}")

    return total_loss / len(loader)


def setup_ddp(args):
    args.distributed = ("WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1)

    if args.distributed:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(args.local_rank)
        if args.local_rank == 0:
            print(f"Running DDP on {dist.get_world_size()} GPUs.")
    else:
        args.local_rank = 0


def save_checkpoint(model, optimizer, scheduler, epoch, args):
    if args.local_rank != 0:
        return

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(ckpt, f"{args.save_dir}/epoch_{epoch}.pt")
    print(f"[Saved] checkpoint @ epoch {epoch}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--action_h5", type=str, default="processed_dataset_actions.h5")
    parser.add_argument("--init_depth_h5", type=str, default="processed_dataset_init_depth.h5")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_dir", type=str, default="ckpts_insertion")

    return parser.parse_args()


def main():
    args = parse_args()
    setup_ddp(args)
    set_seed(args.seed)

    # ---- Build dataset ----
    loader, sampler = build_dataloader(args)

    # ---- Build model ----
    model = InsertionNet(
        feature_dim=512,
        hidden_dim=512,
        action_dim=9,
        shared_encoder=True
    ).cuda()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = args.epochs * len(loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # ---- Training Loop ----
    for epoch in range(args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)

        if args.local_rank == 0:
            print(f"\n==== Epoch {epoch}/{args.epochs} ====")

        avg_loss = train_one_epoch(model, loader, optimizer, scheduler, args)

        if args.local_rank == 0:
            print(f"[Epoch {epoch}] Avg Loss = {avg_loss:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch, args)

    if args.distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
