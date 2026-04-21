import h5py
import numpy as np
import os
import argparse

# -------------------------
# HDF5 readers (不变)
# -------------------------
def read_from_hdf5(filename):
    data = {}
    with h5py.File(filename, "r") as f:
        for key in ["flow", "zs"]:
            try:
                data[key] = f[key][()]
            except Exception as e:
                print(f"Could not read {key}: {e}")
    return data


def read_from_hdf5_depth(filename):
    data = {}
    with h5py.File(filename, "r") as f:
        for key in ["mask","camera3_depth"]:
            try:
                data[key] = f[key][()]
            except Exception as e:
                print(f"Could not read {key}: {e}")
    return data


# -------------------------
# Argparse
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, required=True)
    return parser.parse_args()

def apply_mask_to_flow(flow, socket_mask):
    """
    flow: (3, H, W)
    socket_mask: (H, W), bool or {0,1}

    return: (3, H, W)
    """
    mask = socket_mask.astype(bool)          # (H, W)
    mask = mask[None, :, :]                  # (1, H, W)

    flow_masked = flow * mask                # broadcast over channel dim
    return flow_masked


# =========================
# Main
# =========================
if __name__ == "__main__":

    args = parse_args()
    task_id = args.task_id

    flow_mask_root = f"/home/ubuntu/automate/ego_flow_data_0130_acc_flow_keyframe/{task_id}"
    depth_root = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_0125_gt/asset_{task_id}"

    assert os.path.isdir(flow_mask_root), f"Missing {flow_mask_root}"
    assert os.path.isdir(depth_root), f"Missing {depth_root}"

    # --------------------------------------------------
    # 🔑 1️⃣ 找出实际存在的 hdf_id
    # --------------------------------------------------
    
    hdf_ids = sorted([
        int(f.split("_")[-1].replace(".h5", ""))
        for f in os.listdir(flow_mask_root)
        if f.startswith(f"ego_flow_") and f.endswith(".h5")
    ])

    print(f"[INFO] Found {len(hdf_ids)} hdf files for task {task_id}")
    print("hdf_ids:", hdf_ids)

    flow_mask_id = 0
    num=0
    for hdf_id in hdf_ids:
        num = num + 1
        if hdf_id > 9:
           break
        flow_mask_path = f"{flow_mask_root}/ego_flow_{hdf_id}.h5"
        depth_path = f"{depth_root}/disassembly_traj_{hdf_id}.h5"

        # depth 不存在就跳过（安全）
        if not os.path.exists(depth_path):
            print(f"[WARN] Missing depth file for hdf_id={hdf_id}, skip")
            continue

        print(f"[TASK {task_id}] Processing hdf_id={hdf_id}")
        try:
            data = read_from_hdf5(flow_mask_path)
            depth_data = read_from_hdf5_depth(depth_path)
            depth = depth_data["camera3_depth"]          # (T, N, H, W)
            depth_rot_rev = depth[::-1].copy()
            mask = depth_data["mask"][::-1,...].copy()
        except:
            continue
        T = data["flow"].shape[0]
        # flow (69, 12, 2, 240, 320)
        if_crop = False
        for i in range(data["flow"].shape[1]):
            T = 39
            for j in range(T):
                if if_crop:
                    flow_3d = np.zeros([3,180,240])
                else:
                    flow_3d = np.zeros([3,240,320])
                #print("data[flow], data[zs] shape: ",data["flow"].shape,data["zs"].shape)
                flow = data["flow"][j][i]
                zs = data["zs"][i][j]
                if if_crop:
                    socket_mask = mask[j,i][::2,::2][30:-30,40:-40].copy()
                else:
                    socket_mask = mask[j,i][::2,::2].copy()
                
                zs = (zs + 0.01413) / 0.01727
                #zs = (zs+0.003496)/0.003017
                flow_3d[2,...] = zs
                if if_crop:
                    flow_3d[:2,...]=flow[:,30:-30,40:-40].copy()
                else:
                    flow_3d[:2,...]=flow.copy()
                flow_3d = apply_mask_to_flow(flow_3d, socket_mask)
                # flow shape: 3 * 240 * 320; socket_mask shape: 240 * 320
                
                #print("Debug, print flow shape: ", flow.shape)
                #print("Debug, print zs: ", zs)
                curr_depth = depth_rot_rev[j][i]
                # --------------------------------------------------
                # 🔑 2️⃣ 确保保存目录存在
                # --------------------------------------------------
                out_dir = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/flow_net/asset_{task_id}"
                os.makedirs(out_dir, exist_ok=True)
                out_path = f"{out_dir}/flow_mask_{flow_mask_id}.npz"
                print(flow.shape, socket_mask.shape, curr_depth.shape)
                np.savez(
                    out_path,
                    flow=flow_3d.astype(np.float32),
                    depth=curr_depth[::2,::2][30:-30,40:-40] if if_crop else curr_depth[::2,::2],
                    mask = socket_mask
                )
                try:
                   # print("output_path: ", out_path)
                   test_data=np.load(out_path,allow_pickle=False)
                   #print("Test Loading the data")
                except:
                   print(out_path)
                flow_mask_id += 1
                
        print(f"[INFO] Accumulated samples: {flow_mask_id}")

    print("[DONE]")