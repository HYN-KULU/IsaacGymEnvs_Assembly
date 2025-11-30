import h5py
import numpy as np
from diffusion_utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform

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
        for key in ["plug_pos", "plug_quat", "init_plug_pos", "init_plug_quat"]:
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
    for task_id in ["00021", "00028", "00030", "00042", "00110", "00681"]:
        for i in range(49):
            file_list.append(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/insertion_net/asset_{task_id}/disassembly_traj_{i}.h5")
    for file in file_list:
        print(file)
        data=read_from_hdf5(file)
        # data plug_pos shape 12, 82, 3
        # data init plug pos shape 12, 3
        tcp=np.concatenate([data["plug_pos"],data["plug_quat"]],axis=2)
        tcp_rotation_6d=xyz_rot_transform(tcp,from_rep="quaternion", to_rep="rotation_6d")
        tcp_init=np.concatenate([data["init_plug_pos"][:, None, :],data["init_plug_quat"][:, None, :]],axis=2)
        tcp_rotation_6d_init=xyz_rot_transform(tcp_init,from_rep="quaternion", to_rep="rotation_6d")
        action=tcp_rotation_6d_init - tcp_rotation_6d
        action[:,:,2] = 0
        action_list.append(action.reshape(-1,9))
    action_array = np.concatenate(action_list, dtype=np.float32)  
    # Save to HDF5
    out_filename = "processed_dataset_depth_relative_insertion_net.h5"

    with h5py.File(out_filename, "w") as f:
        f.create_dataset("actions", data=action_array, compression="gzip", compression_opts=4)

    print(f"Saved processed dataset to {out_filename}")
    import pdb;pdb.set_trace()