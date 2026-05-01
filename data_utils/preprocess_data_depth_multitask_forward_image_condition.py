import h5py
import numpy as np
import torch
import cv2
from collections import OrderedDict
from data_utils.socket_image import rotate_socket_image_opposite_gripper, relative_yaw_from_quats,quat_xyzw_to_yaw, rotate_socket_depth_opposite_gripper
from data_utils.pointcloud_rgbd import render_top_down_custom
import matplotlib.pyplot as plt

def visualize_depth(
    depth,
    save_path=None,
    title="depth",
    invalid_val=None,
    percentile_clip=(1, 99),
    cmap="viridis",
    show=False,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    depth = np.asarray(depth).astype(np.float32)

    valid = np.isfinite(depth)

    if invalid_val is not None:
        valid = valid & (depth != invalid_val)

    if valid.sum() == 0:
        print("[visualize_depth] No valid depth pixels.")
        vis = np.zeros_like(depth, dtype=np.float32)
        vmin, vmax = 0.0, 1.0
    else:
        d_valid = depth[valid]

        if percentile_clip is not None:
            low, high = percentile_clip
            vmin = np.percentile(d_valid, low)
            vmax = np.percentile(d_valid, high)
        else:
            vmin = d_valid.min()
            vmax = d_valid.max()

        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1e-8

        vis = depth.copy()
        vis[~valid] = vmin

    plt.figure(figsize=(7, 5))
    im = plt.imshow(vis, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        save_dir = os.path.dirname(save_path)

        # Only create folder if save_path includes a folder.
        # Example:
        #   "vis/depth.png" -> create "vis"
        #   "depth.png"     -> do not call os.makedirs("")
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"[visualize_depth] Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

import cv2
import numpy as np


def zoom_depth_center(depth, scale=1.5, invalid_fill=0.0, interpolation=cv2.INTER_NEAREST):
    """
    Zoom into the center of a depth image while keeping the same H, W.

    scale > 1.0: zoom in
    scale = 1.0: unchanged
    scale < 1.0: zoom out
    """
    depth = np.asarray(depth)
    H, W = depth.shape[:2]

    if scale == 1.0:
        return depth.copy()

    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    if scale > 1.0:
        # Crop the center region, then resize back to original size.
        crop_h = int(round(H / scale))
        crop_w = int(round(W / scale))

        crop_h = max(1, min(H, crop_h))
        crop_w = max(1, min(W, crop_w))

        y0 = (H - crop_h) // 2
        x0 = (W - crop_w) // 2

        cropped = depth[y0:y0 + crop_h, x0:x0 + crop_w]

        zoomed = cv2.resize(
            cropped,
            (W, H),
            interpolation=interpolation,
        )

    else:
        # Zoom out: resize smaller, then pad back to original size.
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))

        new_h = max(1, min(H, new_h))
        new_w = max(1, min(W, new_w))

        small = cv2.resize(
            depth,
            (new_w, new_h),
            interpolation=interpolation,
        )

        zoomed = np.full_like(depth, invalid_fill)

        y0 = (H - new_h) // 2
        x0 = (W - new_w) // 2

        zoomed[y0:y0 + new_h, x0:x0 + new_w] = small

    return zoomed.astype(depth.dtype)

def read_from_hdf5(filename):
    """
    Load saved trajectories and sensor data from an HDF5 file.
    Returns a dictionary with all available datasets.
    """
    data = {}
    with h5py.File(filename, "r") as f:
        for key in ["camera3_depth", "fingertip_centered_pos","fingertip_centered_quat", "plug_pos", "plug_quat", "init_plug_pos", "init_plug_quat","actions","init_plug_photo_depth"]:
            try:
                data[key] = f[key][()]   # load as numpy array
            except Exception as e:
                print(f"Could not read {key}: {e}")
        try:
            grp = f["init_point_list"]
            point_list = []
            for idx in range(len(grp.keys())):
                point_list.append(grp[f"{idx}"][()])  # each is (N, D)

            data["init_point_list"] = point_list
        except Exception as e:
            print(f"Could not read init_point_list: {e}")
    return data
def load_processed_dataset(filename):
    import h5py
    data = {}
    with h5py.File(filename, "r") as f:
        # data["depth"] = f["depth"][()]               # (N, H, W)
        data["timestep_index"] = f["timestep_index"][()]  # (N, 9)
    return data
import pickle
import os
import argparse
def normalize_depth_for_shape(depth, invalid_fill=0.0):
    """
    Normalize one depth image independently to [0, 1],
    keeping only relative shape information.

    Works for:
    - positive depth maps
    - negative IsaacGym-style depth maps
    - maps containing inf / -inf / nan
    """
    depth = depth.astype(np.float32)

    # valid pixels: finite only
    valid = np.isfinite(depth)

    # if nothing valid, return zeros
    if valid.sum() == 0:
        return np.full_like(depth, invalid_fill, dtype=np.float32)

    d = depth.copy()

    # normalize using only valid region
    d_valid = d[valid]
    d_min = d_valid.min()
    d_max = d_valid.max()

    # avoid divide-by-zero if all valid depths are identical
    if d_max - d_min < 1e-8:
        out = np.full_like(d, invalid_fill, dtype=np.float32)
        out[valid] = 1.0
        return out

    out = np.full_like(d, invalid_fill, dtype=np.float32)
    out[valid] = (d[valid] - d_min) / (d_max - d_min)

    return out.astype(np.float32)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--hdf_id", type=str, required=True)
    return parser.parse_args()
if __name__=="__main__":
    args = parse_args()
    task_id = args.task_id
    hdf_id = int(args.hdf_id)
    proprioception_list=[]
    action_list=[]
    depth_id=0
    K=10
    with open('/home/ubuntu/automate/IsaacGymEnvs_Assembly/data_utils/preprocess/timestep_index_0429_force_feedback.pkl', 'rb') as f:
        timestep_index = pickle.load(f)
    for hdf_id in [hdf_id]:
        try:
            data=read_from_hdf5(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_0416_forward_force_photo_force_feedback/asset_{task_id}/disassembly_traj_{hdf_id}.h5")      
        except Exception as e:
            continue
        if data["fingertip_centered_pos"].shape[0]==0:
            continue
        if data["fingertip_centered_pos"].shape[0]!=data["actions"].shape[1]:
            continue
        deviation_quat = np.linalg.norm(data["init_plug_quat"] - data["plug_quat"][:,-1,:],axis=1)
        deviation_pos = np.linalg.norm((data["init_plug_pos"] - data["plug_pos"][:,-1,:])[:,:2],axis=1)
        indices = np.where((deviation_quat < 0.2) & (deviation_pos < 3e-3))[0]
        depth_data = data["camera3_depth"][:,indices,:,:]
        for i in range(len(indices)):
            T = timestep_index.get((task_id, hdf_id, i), None)
            if T is None:
                print(f"Missing timestep index for task {task_id}, hdf_id {hdf_id}, index {i}")
                continue
            q_init = data["fingertip_centered_quat"][i,0]
            points= torch.from_numpy(data["init_point_list"][i]).float().to("cuda")
            _, depth_init = render_top_down_custom(points[:,:3], points[:,3:], H=depth_data.shape[2], W=depth_data.shape[3], camera_height_offset=0.02, fov_deg=73.73979365244269, point_radius=5)
            init_plug_photo_depth = np.flipud(data["init_plug_photo_depth"][i])
            distance_camera_to_plug = abs(init_plug_photo_depth.max())
            scale = distance_camera_to_plug / 0.02
            init_plug_photo_depth = normalize_depth_for_shape(zoom_depth_center(init_plug_photo_depth, scale=scale), invalid_fill=0.0)
            # visualize_depth(init_plug_photo_depth, f"debug/asset_{task_id}_hdf_{hdf_id}_index_{i}_init_plug_photo_depth.png", title="init_plug_photo_depth")
            # visualize_depth(normalize_depth_for_shape(depth_init.clone().detach().cpu().numpy(), invalid_fill=0.0), f"debug/asset_{task_id}_hdf_{hdf_id}_index_{i}_depth_init.png", title="depth_init")
            for step in range(T-K):  # include all timesteps
                if step == 0:
                    socket_depth = depth_init.clone().detach().cpu().numpy()
                else:
                    q_final = data["fingertip_centered_quat"][i,step]
                    socket_depth,_ = rotate_socket_depth_opposite_gripper(depth_init, q_init, q_final, invalid_val=0.0)
                depth = depth_data[step][i].copy()  # (480, 640)
                socket_depth = normalize_depth_for_shape(socket_depth, invalid_fill=0.0)
                out_dir = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/diffusion_policy_force_image_condition/asset_{task_id}/hdf_{hdf_id}"
                os.makedirs(out_dir, exist_ok=True)
                np.savez(
                    f"{out_dir}/depth_{depth_id}.npz",
                    depth=depth,  # downsample and crop borders
                    socket_depth=socket_depth,
                    init_plug_photo_depth=init_plug_photo_depth
                )
                depth_id+=1
