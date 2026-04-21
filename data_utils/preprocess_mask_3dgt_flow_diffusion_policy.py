import h5py
import numpy as np
from policy.flow_3dgt_mask_policy import FlowPolicy
import torch
import cv2
from collections import OrderedDict

def shift_np_integer(x, dx, dy):
    """
    x: (C,H,W)
    dx, dy: integers
    """
    dx = int(round(dx))
    dy = int(round(dy))

    y = np.roll(x, shift=(dy, dx), axis=(1, 2))

    # zero out wrapped region (important!)
    if dy > 0:
        y[:, :dy, :] = 0
    elif dy < 0:
        y[:, dy:, :] = 0

    if dx > 0:
        y[:, :, :dx] = 0
    elif dx < 0:
        y[:, :, dx:] = 0

    return y
def load_processed_dataset(filename):
    import h5py
    data = {}
    with h5py.File(filename, "r") as f:
        # data["depth"] = f["depth"][()]               # (N, H, W)
        data["proprioception"] = f["proprioception"][()]  # (N, 9)
        data["actions"] = f["actions"][()]           # (N, K, 9)
    return data

def read_from_hdf5(filename):
    """
    Load saved trajectories and sensor data from an HDF5 file.
    Returns a dictionary with all available datasets.
    """
    data = {}
    with h5py.File(filename, "r") as f:
        for key in ["camera3_depth", "mask"]:
            try:
                data[key] = f[key][()]   # load as numpy array
            except Exception as e:
                print(f"Could not read {key}: {e}")
    return data

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

def depth_to_vis(depth):
    """
    depth: (H, W), numpy
    return: (H, W, 3) uint8
    """
    d = depth.copy()
    d = np.nan_to_num(d)

    # normalize per-frame for visualization
    d_min, d_max = d.min(), d.max()
    if d_max > d_min:
        d = (d - d_min) / (d_max - d_min)
    else:
        d = np.zeros_like(d)

    d = (d * 255).astype(np.uint8)
    return cv2.cvtColor(d, cv2.COLOR_GRAY2RGB)
def draw_arrows(img, flow, stride=6, scale=5):
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()

    out = img.copy()
    u = flow[0]
    v = flow[1]
    H, W = u.shape

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            dx = u[y, x]
            dy = v[y, x]

            x2 = int(x + dx * scale)
            y2 = int(y + dy * scale)

            cv2.arrowedLine(out, (x, y), (x2, y2),
                            color=(0, 255, 0),
                            thickness=1,
                            tipLength=0.3)
    return out

def flow_to_direction(flow, eps=1e-6):
    """
    flow: (2, H, W)
    return: (2, H, W), unit direction field
    """
    mag = np.linalg.norm(flow, axis=0, keepdims=True)  # (1,H,W)
    return flow / (mag + eps)
def flow_to_hsv(flow):
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()

    u = flow[0]
    v = flow[1]

    mag = np.sqrt(u*u + v*v)
    ang = np.arctan2(v, u)

    H, W = u.shape
    hsv = np.zeros((H, W, 3), dtype=np.uint8)

    hsv[..., 0] = ((ang + np.pi) / (2*np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, required=True)
    return parser.parse_args()
if __name__=="__main__":
    args = parse_args()
    task_id = args.task_id
    proprioception_list=[]
    action_list=[]
    depth_id=0
    policy = FlowPolicy().to("cuda")
    # ckpt_path = os.path.expanduser("~/automate/logs/automate/masked_3dgt_flow_net_mask_input_ckpt/epoch_195.pt")
    ckpt_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/masked_3dgt_flow_net_mask_adapt_00028_fewshot/step_800.pt"
    ckpt = torch.load(ckpt_path, map_location=torch.device("cuda"))
    sd = ckpt["model"]
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    sd = rewrite_monai_keys(sd)
    policy.load_state_dict(sd, strict=True)
    policy.eval()
    depth_root = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_0125_gt/asset_{task_id}"
    hdf_ids = sorted([
        int(f.split("_")[-1].replace(".h5", ""))
        for f in os.listdir(depth_root)
        if f.startswith(f"disassembly_traj_") and f.endswith(".h5")
    ])
    print(hdf_ids)
    for hdf_id in hdf_ids:
        if hdf_id > 2:
            break
        data=read_from_hdf5(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_0125_gt/asset_{task_id}/disassembly_traj_{hdf_id}.h5")      
        
        for i in range(data["camera3_depth"].shape[1]):
            T=data["camera3_depth"].shape[0]
            T = 60
            for step in range(T):  # include all timesteps
                depth = data["camera3_depth"][T-1-step][i]  # (480, 640)
                depth_tensor = torch.from_numpy(depth).float().cuda()
                depth_tensor = torch.clamp(depth_tensor, min=-0.5, max=0.0)
                depth_tensor = (depth_tensor - (-0.1890)) / 0.0795
                rate = 2
                mask = data["mask"][T-1-step][i][::rate,::rate].copy()
                # rot_depth_tensor is rotated and processed, normalized
                # mask is rotated
                pred_flow = policy(torch.from_numpy(mask).unsqueeze(0).cuda(),depth_tensor[::rate,::rate].unsqueeze(0).unsqueeze(0).cuda())[0].detach().cpu().numpy()
                # print("Pred Flow Shape: ", pred_flow.shape)
                # print("mask shape: ", mask.shape)
                pred_flow = pred_flow * mask[None, :, :]
                
                out_dir = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/flow_diffusion_policy/asset_{task_id}"
                os.makedirs(out_dir, exist_ok=True)
                np.savez(
                    f"{out_dir}/depth_flow_{depth_id}.npz",
                    depth=depth[::rate,::rate],
                    flow=pred_flow
                )
                depth_id+=1