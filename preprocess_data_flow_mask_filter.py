import h5py
import numpy as np


def read_from_hdf5(filename):
    """
    Load saved trajectories and sensor data from an HDF5 file.
    Returns a dictionary with all available datasets.
    """
    data = {}
    with h5py.File(filename, "r") as f:
        # for key in ["flow", "mask"]:
        #     try:
        #         data[key] = f[key][()]   # load as numpy array
        #     except Exception as e:
        #         print(f"Could not read {key}: {e}")
        if "id_list" in f.keys():
            # print(id_list)
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

# 195520

if __name__=="__main__":
    proprioception_list=[]
    action_list=[]
    flow_mask_id=0
    total_length=0
    # for task_id in ["00030"]:
    for task_id in ["00021", "00028", "00030", "00042", "00110","00681"]:
        for hdf_id in range(50):
            print(hdf_id, task_id)
            # data flow shape: 12, 160, 2, 480, 640
            # data mask shape: 12 480 640
            
            data=read_from_hdf5(f"/home/ubuntu/automate/alltracker/flow_mask_data/{task_id}/flow_asset_{task_id}_{hdf_id}.h5") 
            # with h5py.File(f"/home/ubuntu/automate/alltracker/flow_mask_data/{task_id}/flow_asset_{task_id}_{hdf_id}.h5", "r") as f:
            #     print(f.keys())
            #     if "id_list" in f.keys():
            #         print(f["id_list"][()])
            #     print(f["flow"][()].shape)
            # print(len(data["id_list"]))    
            total_length +=len(data["id_list"])
            # if total_length * 160 > 195520:
                # import pdb;pdb.set_trace()
            # depth_data= read_from_hdf5_depth(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow/asset_{task_id}/disassembly_traj_{hdf_id}.h5")
            # depth = depth_data["camera3_depth"]
            # print(depth.shape) 
            # id_list=data["id_list"]
            # depth_rot = np.rot90(depth, 2, axes=(2, 3)) 
            # depth_rot_rev = depth_rot[::-1].copy()
            # for i in range(data["flow"].shape[0]):
            # for i in id_list:
            #     T=data["flow"].shape[1]
            #     for j in range(T):
            #         flow=data["flow"][i][j]
            #         mask=data["mask"][i]
            #         curr_depth=depth_rot_rev[j][i]
            #         np.savez(
            #                     f"/tmp/flow_net/flow_mask_{flow_mask_id}.npz",
            #                     flow=flow,
            #                     mask=mask,
            #                     depth = curr_depth
            #                 )
            #         flow_mask_id+=1
            #         print("Processed ", flow_mask_id)
    print(total_length)