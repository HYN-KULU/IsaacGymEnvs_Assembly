import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torch.utils.data import DataLoader
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
        

    def __len__(self):
        # return self.length
        return 10

    def __getitem__(self, idx):
        data = np.load(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/flow_net/asset_{self.task_id}/flow_mask_{idx}.npz", allow_pickle=False)
        flow = data["flow"].copy()
        mask = data["mask"].copy()
        mask  = np.expand_dims(mask, axis=0)
        depth = data["depth"].copy()
        depth = np.expand_dims(depth, axis=0)
        depth = torch.from_numpy(depth)
        depth = torch.clamp(depth, min=-0.5, max=0.0)
        depth = (depth - (-0.1890)) / 0.0795
        flow  = torch.from_numpy(flow)             # (2,480,640)
        mask  = torch.from_numpy(mask) 
        rate=2
        flow_down = flow[:, ::rate, ::rate]    # (2, 240, 320)
        mask_down = mask[:, ::rate, ::rate]
        depth_down = depth[:, ::rate, ::rate]

        return {
            "mask": mask_down,
            "depth": depth_down,
            "flow": flow_down
        }
def main():
    dataset = DepthActionDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    print("Dataset size:", len(dataset))

    for batch in loader:
        print("Batch mask shape: ", batch["mask"].shape)   # (B,1,H,W)
        print("Batch depth shape:", batch["depth"].shape)  # (B,1,H,W)
        print("Batch flow shape: ", batch["flow"].shape)   # (B,2,H,W)
        break  # only test first batch


if __name__ == "__main__":
    main()