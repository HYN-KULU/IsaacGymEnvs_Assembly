import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torch.utils.data import DataLoader
import json
class DepthActionDataset(Dataset):
    def __init__(self, task_id=""):
        """
        Args:
            hdf5_file (str): Path to processed_dataset.h5
            transform: Optional transform for depth images
            normalize: Whether to normalize delta positions
            scale_to_unit: If True, scale to [-1, 1], otherwise [0, 1]
        """
        self.task_id = task_id
        self.index = np.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/flow_diffusion_policy_adapt_index_00015_flow.npy", mmap_mode = "r").copy()
        print("Dataset Length: ", len(self.index))
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        task_id , flow_mask_id = self.index[idx]
        path = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/flow_net/asset_{task_id:05d}/flow_mask_{flow_mask_id}.npz"
        try:
            data = np.load(path)
        except Exception as e:
            print(f"Fail to load the path: Taskid: {task_id}, flow_mask_id: {flow_mask_id}; path: {path}.")
            raise
        flow = data["flow"].copy()
        depth = data["depth"].copy()
        depth = np.expand_dims(depth, axis=0)
        depth = torch.from_numpy(depth).float().contiguous().clone()
        depth = torch.clamp(depth, min=-0.5, max=0.0)
        depth = (depth - (-0.1890)) / 0.0795
        flow  = torch.from_numpy(flow).float().contiguous().clone()             # (2,480,640)
        # rate=2
        # depth_down = depth[:, ::rate, ::rate]
        mask = torch.from_numpy(data["mask"].copy()).float().contiguous().clone()
        # print(flow.shape, depth.shape, mask.shape)
        return {
            "depth": depth,
            "flow": flow,
            "mask": mask
        }