import os
import numpy as np
import h5py
import zipfile
import io

def read_from_hdf5(filename):
    with h5py.File(filename, "r") as f:
        depth = f["camera3_depth"][()]  # (180, 12, 480, 640)
    return depth

def save_depth_to_zip(depth, out_dir, hdf_id):
    """
    depth: np.array (12, 180, 480, 640)
    将其展平成 (2160, 480, 640)，再拆成前后两半分别压缩成zip。
    """
    os.makedirs(out_dir, exist_ok=True)

    # flatten to (2160, 480, 640)
    depth = depth.reshape(-1, 480, 640)
    half = depth.shape[0] // 2

    halves = [
        (depth[:half], f"{out_dir}/depth_{hdf_id*2}.zip"),
        (depth[half:], f"{out_dir}/depth_{hdf_id*2+1}.zip"),
    ]

    for part_data, zip_path in halves:
        # 写入单个npz到zip
        buf = io.BytesIO()
        np.savez_compressed(buf, depth=part_data)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"depth_part.npz", buf.getvalue())
        print(f"✅ Saved {zip_path} with shape {part_data.shape}")

if __name__ == "__main__":
    for hdf_id in range(54):
        data_path = (
            "/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/depth_relative_1024_multitask/"+
            f"disassembly_traj_{hdf_id}.h5"
        )
        depth = read_from_hdf5(data_path)
        print("Original shape:", depth.shape)

        # ✅ 倒序时间，再转置为 trajectory-first
        depth = depth[::-1, :, :, :]          # reverse time
        depth = np.transpose(depth, (1, 0, 2, 3))  # (12, 180, 480, 640)
        print("After reverse + transpose:", depth.shape)

        # ✅ 保存为两个zip（每个包含一个npz）
        out_dir = "data/depth_relative_multitask_zip"
        save_depth_to_zip(depth, out_dir, hdf_id)