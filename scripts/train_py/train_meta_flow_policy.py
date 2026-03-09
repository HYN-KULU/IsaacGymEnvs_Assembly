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
    def __init__(self, num_workers=8, batch_size=16):
        self.base_dataset = DepthActionDataset()

        self.task_to_indices = defaultdict(list)
        for idx, (task_id, flow_mask_id) in enumerate(self.base_dataset.index):
            self.task_to_indices[int(task_id)].append(idx)
        self.tasks = list(self.task_to_indices.keys())

        self.num_workers = num_workers
        self.batch_size = batch_size

        print("Total tasks:", len(self.tasks))

    def sample_tasks(self, meta_batch_size):
        return random.sample(self.tasks, meta_batch_size)

    def get_task_loader(self, task_id, K):
        """
        Return a DataLoader that samples K elements from this task
        """
        indices = self.task_to_indices[task_id]
        chosen = random.sample(indices, K)

        subset = Subset(self.base_dataset, chosen)

        loader = DataLoader(
            subset,
            batch_size=K,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return loader

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
    device="cuda",
    meta_iters=5000,
    meta_batch_size=4,
    K_support=240,
    K_query=240,
    inner_steps=5,
    inner_lr=1e-3,
    meta_lr=1e-4,
):

    model.to(device)
    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)

    for iteration in range(meta_iters):

        meta_optimizer.zero_grad()
        meta_loss = 0

        sampled_tasks = meta_dataset.sample_tasks(meta_batch_size)
        for task_id in sampled_tasks:

            support_loader = meta_dataset.get_task_loader(task_id, K_support)
            support_batch = next(iter(support_loader))
            query_loader = meta_dataset.get_task_loader(task_id, K_query)
            query_batch = next(iter(query_loader))

            inner_optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=inner_lr
            )

            with higher.innerloop_ctx(
                model,
                inner_optimizer,
                copy_initial_weights=True,
                track_higher_grads=False  # FOMAML
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

    # Initialize model
    model = FlowPolicy()

    # Freeze strategy
    freeze_all(model)
    unfreeze_last_decoder_and_output(model)

    print("\nTrainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)

    # Initialize meta dataset
    meta_dataset = MetaFlowDataset(num_workers=8)
    # Start meta training
    meta_train(
        model=model,
        meta_dataset=meta_dataset,
        device=device,
        meta_iters=5000,
        meta_batch_size=4,
        K_support=20,
        K_query=20,
        inner_steps=5,
        inner_lr=1e-3,
        meta_lr=1e-4,
    )


if __name__ == "__main__":
    main()