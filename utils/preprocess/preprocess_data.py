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
        for key in ["fingertip_centered_pos", "fingertip_centered_quat", "camera3_depth"]:
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
    K=10
    # data=load_processed_dataset("/home/ubuntu/automate/IsaacGymEnvs_Assembly/processed_dataset_depth_relative_scale_1003.h5")
    # proprioception_list=data["proprioception"].tolist()
    # action_list=data["actions"].tolist()
    proprioception_list=[]
    action_list=[]
    for hdf_id in range(50):
        print(hdf_id)
        data=read_from_hdf5(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/depth_relative_1010/asset_00681_disassembly_traj_{hdf_id}.h5")
        print(data["fingertip_centered_pos"].shape)
        quat = data["fingertip_centered_quat"][:, ::-1, :]   # reverse time (T=180)
        pos  = data["fingertip_centered_pos"][:, ::-1, :]
        tcp=np.concatenate([pos,quat],axis=2)
        tcp_rotation_6d=xyz_rot_transform(tcp,from_rep="quaternion", to_rep="rotation_6d")
            
        for i in range(tcp_rotation_6d.shape[0]):
            pos_env_i=pos[i]
            tcp_rotation_6d_env_i=tcp_rotation_6d[i]
            T = pos_env_i.shape[0]
            for step in range(T):  # include all timesteps
                # proprioception: current absolute pose (3+6)
                proprioception = tcp_rotation_6d_env_i[step] # (9,)

                # depth at current step
                depth = data["camera3_depth"][T-1-step][i]  # (480, 640)

                future_actions = []
                for j in range(1, K+1):
                    if step + j < T:
                        delta_pos = pos_env_i[step+j] - pos_env_i[step+j-1]   # (3,)
                        rot6d = tcp_rotation_6d_env_i[step+j][3:]-tcp_rotation_6d_env_i[step+j-1][3:]                 # (6,)
                    else:
                        delta_pos = np.zeros(3, dtype=np.float32)
                        rot6d = np.zeros(6, dtype=np.float32)  # repeat last rotation
                    action_j = np.concatenate([delta_pos, rot6d])  # (9,)
                    future_actions.append(action_j)

                # final target shape: (K, 9)
                action_target = np.stack(future_actions, axis=0)

                action_list.append(action_target)
                proprioception_list.append(proprioception)
                # depth_list.append(depth)
                np.save(f"data/depth_relative_scale_1010/depth_{len(action_list)-1}.npy",depth)
                # print(len(action_list))
    
    # depth_array = np.array(depth_list, dtype=np.float32)               # shape (N, 480, 640)
    proprio_array = np.array(proprioception_list, dtype=np.float32)    # shape (N, 9)
    action_array = np.array(action_list, dtype=np.float32)             # shape (N, K, 9)

    print("Final dataset shapes:")
    # print("Depth:", depth_array.shape)
    print("Proprioception:", proprio_array.shape)
    print("Actions:", action_array.shape)

    # Save to HDF5
    out_filename = "processed_dataset_depth_relative_scale_1010.h5"
    with h5py.File(out_filename, "w") as f:
        # f.create_dataset("depth", data=depth_array, chunks=(1, 480, 640), compression="gzip", compression_opts=4)
        f.create_dataset("proprioception", data=proprio_array, compression="gzip", compression_opts=4)
        f.create_dataset("actions", data=action_array, compression="gzip", compression_opts=4)

    print(f"Saved processed dataset to {out_filename}")
    import pdb;pdb.set_trace()