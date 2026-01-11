import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torch.utils.data import DataLoader
class DepthActionDataset(Dataset):
    def __init__(self, hdf5_file_actions,hdf5_file_depth_init, transform=None, normalize=True, scale_to_unit=True):
        """
        Args:
            hdf5_file (str): Path to processed_dataset.h5
            transform: Optional transform for depth images
            normalize: Whether to normalize delta positions
            scale_to_unit: If True, scale to [-1, 1], otherwise [0, 1]
        """
        self.hdf5_file = hdf5_file_actions
        self.transform = transform
        self.normalize = normalize
        self.scale_to_unit = scale_to_unit

        # open in read-only mode, but don't load into RAM
        self.h5 = h5py.File(self.hdf5_file, "r")
        self.actions = self.h5["actions"]

        self.length = self.actions.shape[0]
        self.h5_init_depth=h5py.File(hdf5_file_depth_init, "r")
        self.init_depth=self.h5_init_depth["init_depth"]
        self.task_id_mapping={"00021":0, "00028":1, "00030":2, "00042":3, "00110":4, "00681":5}
        if self.normalize:
            # compute min/max for delta_pos only (first 3 dims)
            all_actions = self.actions[()]  # (N, K, 9)
            delta_pos = all_actions[..., 0:3]  # (N, K, 3)
            delta_rot = all_actions[...,3:]
            self.pos_min = delta_pos.min(axis=(0, 1))
            self.pos_max = delta_pos.max(axis=(0, 1))
            self.rot_min=delta_rot.min(axis=(0,1))
            self.rot_max=delta_rot.max(axis=(0,1))
        else:
            self.pos_min = None
            self.pos_max = None
        import pdb;pdb.set_trace()
        print(self.pos_min)
        

    def __len__(self):
        # return self.length
        return 100

    def __getitem__(self, idx):
        x = np.load(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/depth_insertion_net/depth_{idx}.npz", allow_pickle=True)
        depth = x["depth"].astype(np.float32)
        task_id = x["task_id"].item()
        depth_init=self.init_depth[self.task_id_mapping[task_id]]

        depth = np.expand_dims(depth, axis=0)
        depth = torch.from_numpy(depth)
        depth = torch.clamp(depth, min=-0.5, max=0.0)
        depth = (depth - (-0.1890)) / 0.0795

        depth_init = np.expand_dims(depth_init, axis=0)
        depth_init = torch.from_numpy(depth_init)
        depth_init = torch.clamp(depth_init, min=-0.5, max=0.0)
        depth_init = (depth_init - (-0.1890)) / 0.0795
        actions = self.actions[idx].astype(np.float32)

        if self.normalize:
            delta_pos = actions[..., 0:3]
            delta_rot6d = actions[..., 3:]

            # min-max normalization
            denom = (self.pos_max - self.pos_min) + 1e-8
            norm_delta = (delta_pos - self.pos_min) / denom
            denom_rot=(self.rot_max-self.rot_min)+1e-8
            norm_delta_rot=(delta_rot6d-self.rot_min)/denom_rot
            if self.scale_to_unit:
                # map to [-1, 1]
                norm_delta = norm_delta * 2.0 - 1.0
                norm_delta_rot=norm_delta_rot*2.0-1.0
            actions = np.concatenate([norm_delta, norm_delta_rot], axis=-1)

        actions = torch.from_numpy(actions)

        return {
            "init_depth": depth_init,
            "depth": depth,
            "actions": actions
        }

def main():
    # Paths
    h5_action_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/processed_dataset_depth_relative_insertion_net.h5"
    h5_depth_init_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/processed_dataset_depth_relative_insertion_net_init_depth.h5"

    # Create dataset
    dataset = DepthActionDataset(
        hdf5_file_actions=h5_action_path,
        hdf5_file_depth_init=h5_depth_init_path,
        transform=None,
        normalize=True,
        scale_to_unit=True
    )

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    print("Dataset length:", len(dataset))

    # Iterate through few batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n=== Batch {batch_idx} ===")

        print("init_depth shape:", batch["init_depth"].shape)
        print("depth shape:", batch["depth"].shape)
        print("actions shape:", batch["actions"].shape)

        if batch_idx == 2:   # just show first 3 batches
            break


if __name__ == "__main__":
    main()