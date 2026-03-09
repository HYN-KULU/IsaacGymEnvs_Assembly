import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataset.flow_dataset import DepthActionDataset   # <-- 你自己的 dataset
from policy.flow_policy import FlowPolicy             # <-- 你自己的 FlowPolicy
from policy.film_flow_policy2 import inject_film_at_bottleneck
from diffusion_utils.training import set_seed, sync_loss
from collections import OrderedDict
def rewrite_monai_keys(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        nk = nk.replace(".sub0", ".submodule.0")
        nk = nk.replace(".sub1", ".submodule.1")
        nk = nk.replace(".sub2", ".submodule.2")
        nk = nk.replace(".subconv", ".submodule.conv")
        nk = nk.replace(".subadn", ".submodule.adn")
        new_sd[nk] = v
    return new_sd


def load_pretrained_flow(policy, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"]
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    sd = rewrite_monai_keys(sd)
    policy.load_state_dict(sd, strict=True)
    policy.eval()
# ------------------------------------------------------------
# Build dataloader
# ------------------------------------------------------------
def build_dataloader(args):
    dataset = DepthActionDataset()

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


# ------------------------------------------------------------
# One epoch training
# ------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, args):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, disable=(args.local_rank != 0))

    for batch in pbar:

        mask  = batch["mask"].cuda(non_blocking=True)      # B,1,H,W
        depth = batch["depth"].cuda(non_blocking=True)     # B,1,H,W
        flow_gt = batch["flow"].cuda(non_blocking=True)    # B,2,H,W

        # forward
        pred_flow = model(mask, depth)

        # L2 loss (MSE)
        loss = nn.functional.mse_loss(pred_flow, flow_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        device = torch.device("cuda", args.local_rank)
        # multi-gpu sync
        loss_val = sync_loss(loss, device)
        total_loss += loss_val

        pbar.set_description(f"Loss: {loss_val:.4f}")

    return total_loss / len(loader)


# ------------------------------------------------------------
# DDP setup
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Save checkpoint
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="ckpts_flow")

    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()
    setup_ddp(args)
    set_seed(args.seed)

    # ---- Build dataset ----
    loader, sampler = build_dataloader(args)

    device = torch.device("cuda", args.local_rank)

    # ---- Build base policy ----
    model = FlowPolicy().to(device)

    load_pretrained_flow(
        model,
        args.pretrained_ckpt,
        device
    )

    # ---- Freeze backbone（强烈推荐先这样）----
    for p in model.net.parameters():
        p.requires_grad = False

    bn_name, film = inject_film_at_bottleneck(
        model.net,
        device=device,
        in_shape=(1, 2, 240, 320),
        target_c=512
    )

    if args.local_rank == 0:
        print(f"Injected FiLM at bottleneck: {bn_name}")

    # ---- Ensure FiLM trainable ----
    for p in film.parameters():
        p.requires_grad = True

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    total_steps = args.epochs * len(loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    device = torch.device("cuda", args.local_rank)
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