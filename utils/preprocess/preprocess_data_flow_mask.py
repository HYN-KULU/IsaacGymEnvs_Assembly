import h5py
import numpy as np


def read_from_hdf5(filename):
    """
    Load saved trajectories and sensor data from an HDF5 file.
    Returns a dictionary with all available datasets.
    """
    data = {}
    with h5py.File(filename, "r") as f:
        for key in ["flow", "mask"]:
            try:
                data[key] = f[key][()]   # load as numpy array
            except Exception as e:
                print(f"Could not read {key}: {e}")
        if "id_list" in f.keys():
                id_list=f["id_list"][()].tolist()
        else:
                id_list=[0,1,2,3,4,5,6,7,8,9,10,11]
    data["id_list"] = id_list
    return data

def read_from_hdf5_depth(filename):
    """
    Load saved trajectories and sensor data from an HDF5 file.
    Returns a dictionary with all available datasets.
    """
    data = {}
    with h5py.File(filename, "r") as f:
        for key in ["camera3_depth"]:
            try:
                data[key] = f[key][()]   # load as numpy array
            except Exception as e:
                print(f"Could not read {key}: {e}")
    return data
import matplotlib.pyplot as plt
def save_depth_frames(depth_array, out_path="depth_preview.png", num_frames=10):
    """
    depth_array: (T, N, H, W)
    Saves a 2×5 grid of the first `num_frames` depth frames.
    """
    plt.figure(figsize=(15, 6))

    for i in range(num_frames):
        depth = depth_array[i*5, 0]  # visualize camera index 0

        # normalize depth for viewing
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)

        plt.subplot(2, 5, i + 1)
        plt.imshow(depth_norm, cmap='inferno')
        plt.title(f"Frame {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[Saved] Depth visualization → {out_path}")
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, required=True,
                        help="Task ID, e.g., 00175")
    return parser.parse_args()

if __name__=="__main__":
    proprioception_list=[]
    action_list=[]
    import os
    args = parse_args()
    for task_id in [args.task_id]:
        flow_mask_id=0
        for hdf_id in range(100,101):
            print(hdf_id)
            # data flow shape: 12, 160, 2, 480, 640
            # data mask shape: 12 480 640
            data=read_from_hdf5(f"/home/ubuntu/automate/alltracker/flow_mask_data/{task_id}/flow_asset_{task_id}_{hdf_id}.h5")     
            depth_data= read_from_hdf5_depth(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow/asset_{task_id}/disassembly_traj_{hdf_id}.h5")
            depth = depth_data["camera3_depth"] 
            depth_rot = np.rot90(depth, 2, axes=(2, 3)) 
            depth_rot_rev = depth_rot[::-1].copy()
            id_list=data["id_list"]
            # for i in range(data["flow"].shape[0]):
            for i in range(len(id_list)):
                depth_id = id_list[i]
                T=data["flow"].shape[1]
                for j in range(T):
                    flow=data["flow"][i][j]
                    mask=data["mask"][i]
                    curr_depth=depth_rot_rev[j][depth_id]
                    log_path=f"data/flow_net/asset_{task_id}/flow_mask_{flow_mask_id}.npz"
                    log_dir = os.path.dirname(log_path)
                    os.makedirs(log_dir, exist_ok=True)
                    np.savez(
                                f"data/flow_net/asset_{task_id}/flow_mask_{flow_mask_id}.npz",
                                flow=flow,
                                mask=mask,
                                depth = curr_depth
                            )
                    flow_mask_id+=1
                    print("Processed ", flow_mask_id)
