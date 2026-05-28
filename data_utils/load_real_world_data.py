import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def downsample_depth_with_top_padding(depth_img, target_h=360, target_w=640):
    """Downsample, pad top/width to target size, normalize, rotate 180 degrees, and negate."""
    # Downsample by nearest-neighbor index selection.
    row_idx = np.linspace(0, depth_img.shape[0] - 1, target_h).astype(np.int64)
    col_idx = np.linspace(0, depth_img.shape[1] - 1, target_w).astype(np.int64)
    resized = depth_img[row_idx][:, col_idx]
    # If a later change makes height smaller than target, pad above with the top row.
    if resized.shape[0] < 480:
        pad_top = 480 - resized.shape[0]
        top_row = resized[0:1, :]
        top_pad = np.repeat(top_row, pad_top, axis=0)
        resized = np.concatenate([top_pad, resized], axis=0)

    if resized.shape[1] < target_w:
        pad_w = target_w - resized.shape[1]
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        left_col = resized[:, 0:1]
        right_col = resized[:, -1:]
        left_pad = np.repeat(left_col, pad_left, axis=1)
        right_pad = np.repeat(right_col, pad_right, axis=1)
        resized = np.concatenate([left_pad, resized, right_pad], axis=1)
    resized = - resized
    resized = resized.astype(np.float32, copy=False)
    resized = np.clip(resized, a_min=-0.5, a_max=0.0)
    resized = (resized - (-0.1890)) / 0.0795
    return np.rot90(resized, 2)


def visualize_one_depth(depth_img, title="Depth"):
    """Visualize one depth image with a colorbar."""
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(depth_img)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("visualization.png")

# =========================================================
# Load parquet episode
# =========================================================
out_dir = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/real_world_1"
os.makedirs(out_dir, exist_ok=True)
save_idx = 0

for id in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]:
    path = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/insertion_adaptation_1/data/chunk-000/episode_0000{id}.parquet"

    df = pd.read_parquet(path)

    print("num frames:", len(df))

    # =========================================================
    # Wrist camera depth
    # =========================================================

    wrist_depth = np.stack([
        np.stack(frame)
        for frame in df["observation.images.cam_wrist.depth"]
    ])

    processed_wrist_depth = []
    for i in range(wrist_depth.shape[0]):
        processed_wrist_depth.append(
            downsample_depth_with_top_padding(wrist_depth[i], target_h=360, target_w=640)
        )
    wrist_depth = np.stack(processed_wrist_depth)

    print("wrist_depth shape:", wrist_depth.shape)
    visualize_one_depth(wrist_depth[0], title="Processed Wrist Depth - Frame 0")
    # =========================================================
    # Socket depth image
    # =========================================================

    socket_depth = np.stack([
        np.stack(frame)
        for frame in df["observation.aligned_socket_depth_img"]
    ])

    print("socket_depth shape:", socket_depth.shape)

    # =========================================================
    # Plug depth image
    # =========================================================

    plug_depth = np.stack([
        np.stack(frame)
        for frame in df["observation.init_plug_depth_img"]
    ])

    print("plug_depth shape:", plug_depth.shape)

    # =========================================================
    # Action
    # =========================================================

    actions = np.stack(df["action"].values)

    print("actions shape:", actions.shape)

    # =========================================================
    # Force
    # =========================================================

    forces = np.stack(
        df["observation.eef_internal_forces"].values
    )

    print("forces shape:", forces.shape)

    # Save each timestep in diffusion-policy npz format.
    action_horizon = 10
    for i in range(wrist_depth.shape[0] - action_horizon):
        np.savez(
            f"{out_dir}/data_{save_idx}.npz",
            depth=wrist_depth[i],
            socket_depth=socket_depth[i],
            init_plug_photo_depth=plug_depth[i],
            force=forces[i],
            action9d=actions[i:i + action_horizon],
        )
        save_idx += 1

    # =========================================================
    # Example access
    # =========================================================

    print("\nExample action:")
    print(actions[0])

    print("\nExample force:")
    print(forces[0])
