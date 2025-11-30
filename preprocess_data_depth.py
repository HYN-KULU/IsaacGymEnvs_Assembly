import h5py
import numpy as np

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
        for key in ["camera3_depth"]:
            try:
                data[key] = f[key][()]   # load as numpy array
            except Exception as e:
                print(f"Could not read {key}: {e}")
    return data

if __name__=="__main__":
    K=10
    proprioception_list=[]
    action_list=[]
    depth_id=0
    for hdf_id in range(1):
        print(hdf_id)
        data=read_from_hdf5(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/automate_1106_multitask/01053/01053disassembly_traj_{hdf_id}.h5")      
        for i in range(data["camera3_depth"].shape[1]):
            T=data["camera3_depth"].shape[0]
            for step in range(T):  # include all timesteps
                depth = data["camera3_depth"][T-1-step][i]  # (480, 640)
                np.save(f"data/depth_relative_automate_1106/depth_{depth_id}.npy",depth)
                depth_id+=1
            