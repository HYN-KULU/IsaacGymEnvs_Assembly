from policy.flow_3dgt_mask_policy import FlowPolicy
import torch.optim as optim
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from dataset.flow_3dgt_mask_dataset import DepthActionDataset
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import random
from collections import defaultdict
import higher
import torch.distributed as dist
import os
import wandb
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
class MetaFlowDataset:
    def __init__(self, num_workers=8):
        self.base_dataset = DepthActionDataset()
        self.num_workers = num_workers

        # task → traj → indices
        self.task_traj_indices = defaultdict(lambda: defaultdict(list))

        for idx, (task_id, flow_mask_id) in enumerate(self.base_dataset.index):
            task_id = int(task_id)
            traj_id = int(flow_mask_id) // 69  # adjust if needed
            self.task_traj_indices[task_id][traj_id].append(idx)

        self.tasks = list(self.task_traj_indices.keys())
        print("Total tasks:", len(self.tasks))

    def sample_tasks(self, meta_batch_size):
        return random.sample(self.tasks, meta_batch_size)

    def get_trajectory_batch(self, task_id, num_trajs):
        """
        Sample full trajectories and return all timesteps
        """
        traj_dict = self.task_traj_indices[task_id]
        traj_ids = list(traj_dict.keys())

        chosen_trajs = random.sample(traj_ids, num_trajs)

        all_indices = []
        for traj_id in chosen_trajs:
            all_indices.extend(traj_dict[traj_id])

        subset = Subset(self.base_dataset, all_indices)

        loader = DataLoader(
            subset,
            batch_size=12,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return next(iter(loader))

def print_parameter_names(model):
    for name, param in model.named_parameters():
        print(f"{name:60} | requires_grad={param.requires_grad} | shape={tuple(param.shape)}")
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


def print_trainable(model):
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def compute_loss(model, batch, device):
    depth = batch["depth"].to(device)
    flow  = batch["flow"].to(device)
    mask  = batch["mask"].to(device)

    pred = model(mask, depth)
    return torch.nn.functional.mse_loss(pred, flow)

def meta_train(
    model,
    meta_dataset,
    device,
    meta_iters=40000,
    meta_batch_size=2,
    support_trajs=2,
    query_trajs=2,
    inner_steps=5,
    inner_lr=5e-3,
    meta_lr=1e-4,
):

    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    for iteration in range(meta_iters):

        meta_optimizer.zero_grad()
        meta_loss = torch.tensor(0.0, device=device)

        sampled_tasks = meta_dataset.sample_tasks(meta_batch_size)

        for task_id in sampled_tasks:

            support_batch = meta_dataset.get_trajectory_batch(task_id, support_trajs)
            query_batch   = meta_dataset.get_trajectory_batch(task_id, query_trajs)
            inner_optimizer = optim.Adam(
                # filter(lambda p: p.requires_grad, model.parameters()),
                model.parameters(),
                lr=inner_lr
            )
            support_loss_before = compute_loss(model.module, support_batch, device)
            if rank == 0:
                print("Support before:", support_loss_before.item())
            with higher.innerloop_ctx(
                model.module,
                inner_optimizer,
                copy_initial_weights=True,
                track_higher_grads=False
            ) as (fmodel, diffopt):

                for i in range(inner_steps):
                    loss = compute_loss(fmodel, support_batch, device)
                    if rank == 0:
                        print("  inner step", i, loss.item())
                    diffopt.step(loss)
                query_loss = compute_loss(fmodel, query_batch, device)
                meta_loss += query_loss
                if rank == 0:
                    query_before = compute_loss(model.module, query_batch, device)
                    print("Query before:", query_before.item())
                    print("Query after :", query_loss.item())

        meta_loss /= meta_batch_size

        # 🔥 Synchronize loss across GPUs
        dist.all_reduce(meta_loss, op=dist.ReduceOp.SUM)
        meta_loss /= world_size

        meta_loss.backward()
        meta_optimizer.step()

        if rank == 0:
            print(f"Iter {iteration} | Meta Loss {meta_loss.item():.6f}")
            if iteration % 100 ==0:
                torch.save(model.module.state_dict(), f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/metalearn/meta_flow_checkpoint_{iteration}.pt")
                if rank == 0:
                    loss_value = meta_loss.item()
                    print(f"Iter {iteration} | Meta Loss {loss_value:.6f}")
                    with open("/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/meta_loss.log", "a") as f:
                        f.write(f"Iter {iteration} | Meta Loss {loss_value:.6f}\n")
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

def main():

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    if rank ==0:
        log_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/meta_loss.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write("\n===== New Run =====\n")
    model = FlowPolicy().to(device)
    ckpt = torch.load('/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/3dgt_flow_policy/epoch_0.pt', map_location=device)

    sd = ckpt["model"]
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    sd = rewrite_monai_keys(sd)
    model.load_state_dict(sd, strict=True)
    # freeze_all(model)
    # unfreeze_last_decoder_and_output(model)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    if dist.get_rank() == 0:
        print("\nTrainable parameters:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)

    meta_dataset = MetaFlowDataset(num_workers=12)

    meta_train(
        model=model,
        meta_dataset=meta_dataset,
        device=device,
        meta_iters=40000,
        meta_batch_size=2,   # per GPU
        support_trajs=2,
        query_trajs=2,
        inner_steps=5,
        inner_lr=1e-5,
        meta_lr=1e-5,
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
