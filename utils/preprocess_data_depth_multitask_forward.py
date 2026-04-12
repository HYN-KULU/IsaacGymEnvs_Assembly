import h5py
import numpy as np
from policy.flow_policy import FlowPolicy
import torch
import cv2
from collections import OrderedDict


def read_from_hdf5(filename):
    """
    Load saved trajectories and sensor data from an HDF5 file.
    Returns a dictionary with all available datasets.
    """
    data = {}
    with h5py.File(filename, "r") as f:
        for key in ["camera3_depth", "fingertip_centered_pos", "plug_pos", "plug_quat", "init_plug_pos", "init_plug_quat","actions"]:
            try:
                data[key] = f[key][()]   # load as numpy array
            except Exception as e:
                print(f"Could not read {key}: {e}")
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
    depth_root = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_0307_forward/asset_{task_id}"
    K=10
    with open('/home/ubuntu/automate/IsaacGymEnvs_Assembly/utils/preprocess/timestep_index_00103.pkl', 'rb') as f:
        timestep_index = pickle.load(f)
    for hdf_id in range(6):
        try:
            data=read_from_hdf5(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_0307_forward/asset_{task_id}/disassembly_traj_{hdf_id}.h5")      
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
            for step in range(T-K):  # include all timesteps
                depth = depth_data[step][i].copy()  # (480, 640)
                rate = 2
                out_dir = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/diffusion_policy_multitask_0322/asset_{task_id}"
                os.makedirs(out_dir, exist_ok=True)
                np.savez(
                    f"{out_dir}/depth_{depth_id}.npz",
                    depth=depth[::rate,::rate][30:-30,40:-40],  # downsample and crop borders
                )
                # print(depth_id)
                depth_id+=1
