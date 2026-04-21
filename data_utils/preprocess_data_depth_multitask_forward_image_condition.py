import h5py
import numpy as np
from policy.flow_policy import FlowPolicy
import torch
import cv2
from collections import OrderedDict
from data_utils.socket_image import rotate_socket_image_opposite_gripper, relative_yaw_from_quats,quat_xyzw_to_yaw, rotate_socket_depth_opposite_gripper
from data_utils.pointcloud_rgbd import render_top_down_custom

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
    return parser.parse_args()
if __name__=="__main__":
    args = parse_args()
    task_id = args.task_id
    proprioception_list=[]
    action_list=[]
    depth_id=0
    depth_root = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_0416_forward_force_photo/asset_{task_id}"
    K=10
    with open('/home/ubuntu/automate/IsaacGymEnvs_Assembly/utils/preprocess/timestep_index_0421_force.pkl', 'rb') as f:
        timestep_index = pickle.load(f)
    for hdf_id in range(6):
        try:
            data=read_from_hdf5(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_0416_forward_force_photo/asset_{task_id}/disassembly_traj_{hdf_id}.h5")      
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
            # T=data["camera3_depth"].shape[0]
            T = timestep_index.get((task_id, hdf_id, i), None)
            if T is None:
                print(f"Missing timestep index for task {task_id}, hdf_id {hdf_id}, index {i}")
                continue
            q_init = data["fingertip_centered_quat"][i,0]
            points= torch.from_numpy(data["init_point_list"][i]).float().to("cuda")
            _, depth_init = render_top_down_custom(points[:,:3], points[:,3:], H=depth_data.shape[2], W=depth_data.shape[3], camera_height_offset=0.08, fov_deg=30, point_radius=3)
            init_plug_photo_depth = normalize_depth_for_shape(np.flipud(data["init_plug_photo_depth"][i]), invalid_fill=0.0)
            for step in range(T-K):  # include all timesteps
                if step == 0:
                    socket_depth = depth_init.clone().detach().cpu().numpy()
                else:
                    q_final = data["fingertip_centered_quat"][i,step]
                    socket_depth,_ = rotate_socket_depth_opposite_gripper(depth_init, q_init, q_final, invalid_val=0.0)
                depth = depth_data[step][i].copy()  # (480, 640)
                socket_depth = normalize_depth_for_shape(socket_depth, invalid_fill=0.0)
                out_dir = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/diffusion_policy_image_condition/asset_{task_id}"
                os.makedirs(out_dir, exist_ok=True)
                np.savez(
                    f"{out_dir}/depth_{depth_id}.npz",
                    depth=depth,  # downsample and crop borders
                    socket_depth=socket_depth,
                    init_plug_photo_depth=init_plug_photo_depth
                )
                depth_id+=1
