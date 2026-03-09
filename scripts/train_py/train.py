import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class DepthActionDataset(Dataset):
    def __init__(self, hdf5_file, transform=None, normalize=True, scale_to_unit=True):
        """
        Args:
            hdf5_file (str): Path to processed_dataset.h5
            transform: Optional transform for depth images
            normalize: Whether to normalize delta positions
            scale_to_unit: If True, scale to [-1, 1], otherwise [0, 1]
        """
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.normalize = normalize
        self.scale_to_unit = scale_to_unit

        # open in read-only mode, but don't load into RAM
        self.h5 = h5py.File(self.hdf5_file, "r")
        self.proprio = self.h5["proprioception"]
        self.actions = self.h5["actions"]

        self.length = self.actions.shape[0]

        if self.normalize:
            # compute min/max for delta_pos only (first 3 dims)
            all_actions = self.actions[()]  # (N, K, 9)
            delta_pos = all_actions[..., 0:3]  # (N, K, 3)
            delta_rot = all_actions[..., 3:]
            self.pos_min = delta_pos.min(axis=(0, 1))
            self.pos_max = delta_pos.max(axis=(0, 1))
            self.rot_min=delta_rot.min(axis=(0,1))
            self.rot_max=delta_rot.max(axis=(0,1))
        else:
            self.pos_min = None
            self.pos_max = None
        

    def __len__(self):
        return self.length 
        

    def __getitem__(self, idx):
        # depth (H, W) -> (1, H, W) 
        data=np.load(f"/tmp/automate_00681/depth_flow_{idx}.npz",allow_pickle=False)
        # depth = np.load(f"/tmp/automate_00681/depth_{idx}.npy").astype(np.float32)
        depth = data["depth"].copy()
        flow = data["flow"].copy()
        depth = np.expand_dims(depth, axis=0)
        depth = torch.from_numpy(depth)
        # min: -0.5
        # max: -0.0538
        # mean: -0.1890
        # std: 0.0795
        # if self.transform:
        #     depth = self.transform(depth)
        depth = torch.clamp(depth, min=-0.5, max=0.0)
        depth = (depth - (-0.1890)) / 0.0795
        # proprio (9,)
        proprio = self.proprio[idx].astype(np.float32)
        proprio = torch.from_numpy(proprio)[...,2:]

        # actions (K, 9)
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
            "depth": depth,
            "proprioception": proprio,
            "actions": actions,
            "flow": torch.from_numpy(flow)
        }