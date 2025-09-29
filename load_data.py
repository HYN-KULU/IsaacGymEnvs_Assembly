import h5py
import numpy as np
# from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform
def read_from_hdf5(filename):
    """
    Load saved trajectories and sensor data from an HDF5 file.
    Returns a dictionary with all available datasets.
    """
    data = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            try:
                data[key] = f[key][()]   # load as numpy array
            except Exception as e:
                print(f"Could not read {key}: {e}")
    return data

if __name__=="__main__":
    data=read_from_hdf5("/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/asset_00681_disassembly_traj_0_rgb_0924.h5")
    # Convert quat to rotation 6d
    quat=data["fingertip_centered_quat"]
    pos=data["fingertip_centered_pos"]
    tcp=np.concatenate([pos,quat],axis=2)
    # tcp_rotation_6d=xyz_rot_transform(tcp,from_rep="quaternion", to_rep="rotation_6d")
    import pdb;pdb.set_trace()