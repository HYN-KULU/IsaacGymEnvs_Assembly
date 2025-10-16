from preprocess_data import load_processed_dataset
import torch
data=load_processed_dataset("processed_dataset_depth_relative_1014_DAgger.h5")
all_actions = data['actions'][()]  # (N, K, 9)
delta_pos = all_actions[..., 0:3]  # (N, K, 3)
delta_rot = all_actions[...,3:]
# delta_pos = all_actions[..., 0:3]  # (N, K, 3)
# delta_rot = all_actions[...,3:]
pos_min = delta_pos.min(axis=(0, 1))
pos_max = delta_pos.max(axis=(0, 1))
rot_min=delta_rot.min(axis=(0,1))
rot_max=delta_rot.max(axis=(0,1))
torch.set_printoptions(precision=5, sci_mode=False)
actions=torch.from_numpy(all_actions)
print("pos_min: ", pos_min)
print("pos_max: ", pos_max)
print("rot_min: ", rot_min)
print("rot_max: ", rot_max)

import pdb;pdb.set_trace()