import h5py
import numpy as np
from eval_flow_net_vid import *
def load_processed_dataset(filename):
    import h5py
    data = {}
    with h5py.File(filename, "r") as f:
        # data["depth"] = f["depth"][()]               # (N, H, W)
        data["proprioception"] = f["proprioception"][()]  # (N, 9)
        data["actions"] = f["actions"][()]           # (N, K, 9)
    return data


def read_from_hdf5_mask(filename):
    """
    Load saved trajectories and sensor data from an HDF5 file.
    Returns a dictionary with all available datasets.
    """
    data = {}
    with h5py.File(filename, "r") as f:
        for key in ["mask"]:
        # for key in ["flow", "mask"]:
            try:
                data[key] = f[key][()]   # load as numpy array
            except Exception as e:
                print(f"Could not read {key}: {e}")
        # if "id_list" in f.keys():
            # print(id_list)
            # id_list=f["id_list"][()].tolist()
        # else:
            # id_list=[0,1,2,3,4,5,6,7,8,9,10,11]
        # data["id_list"] = id_list
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
    device="cuda"
    policy = FlowPolicy().to(device)
    ckpt = torch.load("logs/automate/flow_net_ckpt/epoch_low_resolution.pt", map_location=device)

    sd = ckpt["model"]
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    sd = rewrite_monai_keys(sd)
    policy.load_state_dict(sd, strict=True)
    policy.eval()

    K=10
    proprioception_list=[]
    action_list=[]
    depth_flow_action_id=0
    task_id="00681"
    for hdf_id in range(1):
        print(hdf_id)
        mask_data=read_from_hdf5_mask(f"/home/ubuntu/automate/alltracker/flow_mask_data/{task_id}/flow_asset_{task_id}_{hdf_id}.h5")["mask"]
        data=read_from_hdf5(f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow/asset_00681/disassembly_traj_{hdf_id}.h5")      
        depth_data = data["camera3_depth"]
        rate=2
        depth_rot = np.rot90(depth_data, 2, axes=(2, 3)) # rotate it, matching flow_prediction model.
        for i in range(depth_rot.shape[1]):
            T=depth_rot.shape[0]
            for step in range(T):  # include all timesteps
                depth = depth_rot[T-1-step][i] # (480, 640)
                mask = mask_data[i]
                pred_flow = policy(torch.from_numpy(mask[::rate, ::rate].copy()).unsqueeze(0).unsqueeze(0).cuda(), torch.from_numpy(depth[::rate, ::rate].copy()).unsqueeze(0).unsqueeze(0).cuda())[0].detach().cpu().numpy()
                np.savez(
                                f"data/flow_diffusion_policy/flow_depth_action_{depth_flow_action_id}",
                                depth = depth,
                                flow=pred_flow
                            )
                depth_flow_action_id+=1
                print(depth_flow_action_id)            