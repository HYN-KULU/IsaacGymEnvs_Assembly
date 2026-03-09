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
        for i in range(1):
            file_list.append(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/insertion_net/asset_{task_id}/disassembly_traj_{i}.h5")
    for file in file_list:
        print(file)
        data=read_from_hdf5(file)
        # data plug_pos shape 12, 82, 3
        # data init plug pos shape 12, 3
        init_depth=data["camera3_depth"][0][0]
        init_depth_list.append(init_depth)
    depth_array=np.stack(init_depth_list)
    # Save to HDF5
    out_filename = "processed_dataset_depth_relative_insertion_net_init_depth.h5"
    with h5py.File(out_filename, "w") as f:
        f.create_dataset("init_depth", data=depth_array, compression="gzip", compression_opts=4)

    # print(f"Saved processed dataset to {out_filename}")
    import pdb;pdb.set_trace()