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
def meta_train(
    model,
    meta_dataset,
    device="cuda",
    meta_iters=2000,
    meta_batch_size=4,
    support_trajs=2,
    query_trajs=2,
    inner_steps=5,
    inner_lr=5e-3,
    meta_lr=1e-4,
):

    model.to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)

    for iteration in range(meta_iters):

        meta_optimizer.zero_grad()
        meta_loss = 0

        sampled_tasks = meta_dataset.sample_tasks(meta_batch_size)

        for task_id in sampled_tasks:

            # Full trajectory support
            support_batch = meta_dataset.get_trajectory_batch(task_id, support_trajs)

            # Different trajectories for query
            query_batch = meta_dataset.get_trajectory_batch(task_id, query_trajs)

            inner_optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=inner_lr
            )

            with higher.innerloop_ctx(
                model,
                inner_optimizer,
                copy_initial_weights=True,
                track_higher_grads=False
            ) as (fmodel, diffopt):

                for _ in range(inner_steps):
                    loss = compute_loss(fmodel, support_batch, device)
                    diffopt.step(loss)

                query_loss = compute_loss(fmodel, query_batch, device)
                meta_loss += query_loss

        meta_loss /= meta_batch_size
        meta_loss.backward()
        meta_optimizer.step()

        if iteration % 1 == 0:
            print(f"Iter {iteration} | Meta Loss {meta_loss.item():.6f}")

    torch.save(model.state_dict(), "meta_flow_checkpoint.pt")

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FlowPolicy()
    ckpt = torch.load('/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/3dgt_flow_policy/epoch_0.pt', map_location=device)

    sd = ckpt["model"]
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    sd = rewrite_monai_keys(sd)
    model.load_state_dict(sd, strict=True)
    freeze_all(model)
    unfreeze_last_decoder_and_output(model)

    print("\nTrainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)

    meta_dataset = MetaFlowDataset(num_workers=8)

    meta_train(
        model=model,
        meta_dataset=meta_dataset,
        device=device,
        meta_iters=2000,
        meta_batch_size=2,
        support_trajs=2,   # 2 trajectories = 240 samples
        query_trajs=2,
        inner_steps=5,
        inner_lr=5e-3,     # increase for trajectory version
        meta_lr=1e-4,
    )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()