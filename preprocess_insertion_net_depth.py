import h5py
import numpy as np
from diffusion_utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform

def load_processed_dataset(filename):
    import h5py
    data = {}
    with h5py.File(filename, "r") as f:
        data["depth"] = f["depth"][()]               # (N, H, W)
        # data["proprioception"] = f["proprioception"][()]  # (N, 9)
        # data["actions"] = f["actions"][()]           # (N, K, 9)
    return data

def read_from_hdf5(filename):
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

if __name__=="__main__":
    ### Read the dataset, preprocess the dataset by saving the action chunks paired with depth image observation, and proprioception of current pos and rotation_6d
    ### Action Padding
    ### actions: predict delta positions and absolute 6D rotations
    # Convert quat to rotation 6d
    # Check correspondence, since I need to reverse
    # o0=a0 o1=a1 ...... ot=at
    # So each step, given oi and ai, predict ai-1 to ai-k
    action_list=[]
    file_list=[]
    init_depth_list=[]
    for task_id in ["00021", "00028", "00030", "00042", "00110", "00681"]:
        for i in range(49):
            file_list.append((f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/insertion_net/asset_{task_id}/disassembly_traj_{i}.h5",task_id))
    depth_id=0
    for file, task_id in file_list:
        print(file,task_id)
        data=read_from_hdf5(file)
        for i in range(data["camera3_depth"].shape[1]):
            T=data["camera3_depth"].shape[0]
            for step in range(T):  # include all timesteps
                depth = data["camera3_depth"][step][i]  # (480, 640)
                np.savez(
                        f"data/depth_insertion_net/depth_{depth_id}.npz",
                        depth=depth,
                        task_id=task_id
                    )
                depth_id+=1
