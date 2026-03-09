import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
#from dataset.flow_multi_dataset import DepthActionDataset   # <-- 你自己的 dataset
from dataset.flow_3dgt_mask_dataset import DepthActionDataset
from policy.flow_3dgt_mask_policy import FlowPolicy             # <-- 你自己的 FlowPolicy
from diffusion_utils.training import set_seed, sync_loss


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
        num_workers=12,
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
        flow_gt = batch["flow"].cuda(non_blocking=True)    # B,3,H,W

        # forward
        pred_flow = model(mask, depth)

        # L2 loss (MSE)
        loss = nn.functional.mse_loss(pred_flow, flow_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # multi-gpu sync
        device = torch.device("cuda", args.local_rank)
        # loss_val = sync_loss(loss, device)
        loss_val = loss.item() 
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
def save_checkpoint(model, optimizer, scheduler, step, args):
    if args.local_rank != 0:
        return

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(ckpt, f"{args.save_dir}/step_{step}.pt")
    print(f"[Saved] checkpoint @ step {step}")

def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_decoder_and_output(model):
    for name, param in model.named_parameters():

        # Final output layer
        if name.startswith("net.model.2"):
            param.requires_grad = True

        # Last decoder block
        elif name.startswith("net.model.1.submodule.2"):
            param.requires_grad = True



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

    parser.add_argument("--save_dir", type=str, default="ckpts_flow")

    return parser.parse_args()

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

def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            mask  = batch["mask"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            flow_gt = batch["flow"].to(device, non_blocking=True)

            pred_flow = model(mask, depth)

            loss = F.mse_loss(pred_flow, flow_gt, reduction="mean")

            total_loss += loss.item()
            total_samples += mask.size(0)

    # DDP 下需要把各卡结果汇总
    if dist.is_initialized():
        total_loss = torch.tensor(total_loss, device=device)
        total_samples = torch.tensor(total_samples, device=device)

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        total_loss = total_loss.item()
        total_samples = total_samples.item()

    model.train()

    return total_loss / total_samples

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()
    setup_ddp(args)
    set_seed(args.seed)

    # ---- Build dataset ----
    loader, sampler = build_dataloader(args)

    # ---- Build model ----
    model = FlowPolicy().cuda()
    # ckpt = torch.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/metaflow_nofreeze/meta_flow_checkpoint_15000.pt", map_location="cuda")
    ckpt = torch.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/3dgt_flow_policy/epoch_195.pt", map_location="cuda")
    # import pdb;pdb.set_trace()
    sd = ckpt["model"]
    # sd = ckpt
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    sd = rewrite_monai_keys(sd)
    model.load_state_dict(sd, strict=True)
    model.eval()
    # freeze_all(model)
    # unfreeze_last_decoder_and_output(model)
    
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
    device = torch.device("cuda", args.local_rank)
    # ---- Training Loop ----
    global_step = 0
    pbar = tqdm(disable=(args.local_rank != 0))

    if args.local_rank == 0:
        eval_loss = evaluate(model, loader, "cuda")
        print(f"Before finetuning | Eval Loss {eval_loss:.6f}")

    while global_step < 5000:

        model.train()

        for batch in loader:

            mask  = batch["mask"].cuda(non_blocking=True)
            depth = batch["depth"].cuda(non_blocking=True)
            flow_gt = batch["flow"].cuda(non_blocking=True)

            optimizer.zero_grad()

            pred_flow = model(mask, depth)
            loss = F.mse_loss(pred_flow, flow_gt)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # ======================
            # 每 50 step eval + save
            # ======================
            if args.local_rank == 0 and global_step % 50 == 0:
                eval_loss = evaluate(model, loader, mask.device)
                print(f"Step {global_step} | Eval Loss {eval_loss:.6f}")

                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    args
                )

            global_step += 1

            if global_step >= 5000:
                break

            if args.local_rank == 0:
                pbar.set_description(f"Step {global_step} | Loss {loss.item():.6f}")
                pbar.update(1)
    # for epoch in range(args.epochs):
    #     if args.distributed:
    #         sampler.set_epoch(epoch)

    #     if args.local_rank == 0:
    #         print(f"\n==== Epoch {epoch}/{args.epochs} ====")

    #     avg_loss = train_one_epoch(model, loader, optimizer, scheduler, args)

    #     if args.local_rank == 0:
    #         print(f"[Epoch {epoch}] Avg Loss = {avg_loss:.4f}")

    #     if epoch % 1 == 0:
    #         save_checkpoint(model, optimizer, scheduler, epoch, args)

    # if args.distributed:
    #     dist.barrier()


if __name__ == "__main__":
    main()