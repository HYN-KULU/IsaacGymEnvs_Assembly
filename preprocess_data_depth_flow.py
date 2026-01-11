import h5py
import numpy as np
from policy.flow_policy import FlowPolicy
import torch
import cv2
from collections import OrderedDict
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
if __name__=="__main__":
    K=10
    proprioception_list=[]
    action_list=[]
    depth_id=0
    policy = FlowPolicy().to("cuda")
    ckpt_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/flow_net_multitask_ckpt/multi_2_epoch_100.pt"
    ckpt = torch.load(ckpt_path, map_location=torch.device("cuda"))
    sd = ckpt["model"]
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    sd = rewrite_monai_keys(sd)
    policy.load_state_dict(sd, strict=True)
    policy.eval()

    for hdf_id in range(18,19):
    # for hdf_id in range(100,101):
        print(hdf_id)
        data=read_from_hdf5(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_1206/asset_00681/disassembly_traj_{hdf_id}.h5")      
        
        for i in range(data["camera3_depth"].shape[1]):
            i=2
            T=data["camera3_depth"].shape[0]
            print(T)
            for step in range(T):  # include all timesteps
                depth = data["camera3_depth"][T-1-step][i]  # (480, 640)
                rot_depth = np.rot90(depth, 2).copy()
                rot_depth_tensor = torch.from_numpy(rot_depth).float().cuda()
                rot_depth_tensor = torch.clamp(rot_depth_tensor, min=-0.5, max=0.0)
                rot_depth_tensor = (rot_depth_tensor - (-0.1890)) / 0.0795
                mask = np.rot90(data["mask"][i],2)
                # rot_depth_tensor is rotated and processed, normalized
                # mask is rotated
                rate = 2
                pred_flow = policy(torch.from_numpy(mask[::rate,::rate].copy()).unsqueeze(0).unsqueeze(0).cuda(),rot_depth_tensor[::rate,::rate].unsqueeze(0).unsqueeze(0).cuda())[0].detach().cpu().numpy()
                pred_flow = pred_flow * mask[::rate,::rate][None, :, :]
                # np.savez(
                #     f"/tmp/automate_00681/depth_flow_{depth_id}.npz",
                #     depth=rot_depth[::rate,::rate],
                #     flow=pred_flow
                # )
                depth_id+=1
                
                # import os
                
                flow = pred_flow                      # (2, H, W)
                depth_small = rot_depth[::rate, ::rate]

                # ---- depth ----
                depth_vis = depth_to_vis(depth_small)

                # ---- flow ----
                flow_dir = flow_to_direction(flow)

                stride = 1
                arrow_scale = 4.0
                sparse_flow = np.zeros_like(flow_dir)
                sparse_flow[:, ::stride, ::stride] = flow_dir[:, ::stride, ::stride]
                sparse_flow *= arrow_scale

                flow_hsv = flow_to_hsv(flow)
                flow_vis = draw_arrows(flow_hsv, sparse_flow)

                # ---- resize depth if needed ----
                if depth_vis.shape[:2] != flow_vis.shape[:2]:
                    depth_vis = cv2.resize(
                        depth_vis,
                        (flow_vis.shape[1], flow_vis.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                # ---- concat & save ----
                vis = np.concatenate([depth_vis, flow_vis], axis=1)

                os.makedirs("./flow_vis", exist_ok=True)
                out_path = f"./flow_vis/depth_flow_{depth_id}.png"
                cv2.imwrite(out_path, vis)
            
                # import pdb;pdb.set_trace()
            os._exit(0)