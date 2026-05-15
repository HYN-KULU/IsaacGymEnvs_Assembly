import os
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_depth(
    depth,
    save_path=None,
    title="depth",
    invalid_val=0.0,
    percentile_clip=(1, 99),
    cmap="viridis",
    show=False,
):
    depth = np.asarray(depth).astype(np.float32)

    valid = np.isfinite(depth)

    if invalid_val is not None:
        valid = valid & (depth != invalid_val)

    if valid.sum() == 0:
        print("[visualize_depth] No valid depth pixels.")
        vis = np.zeros_like(depth, dtype=np.float32)
        vmin, vmax = 0.0, 1.0
    else:
        d_valid = depth[valid]

        if percentile_clip is not None:
            low, high = percentile_clip
            vmin = np.percentile(d_valid, low)
            vmax = np.percentile(d_valid, high)
        else:
            vmin = d_valid.min()
            vmax = d_valid.max()

        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1e-8

        vis = depth.copy()
        vis[~valid] = vmin

    plt.figure(figsize=(7, 5))
    im = plt.imshow(vis, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"[visualize_depth] Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_depth_comparison(
    input_depth,
    output_depth,
    save_path=None,
    title="Depth before / after disturbance",
    invalid_val=0.0,
    percentile_clip=(1, 99),
    cmap="viridis",
    show=False,
):
    """
    Visualize before and after with the same color scale.
    This makes the perturbation easier to compare.
    """
    input_depth = np.asarray(input_depth).astype(np.float32)
    output_depth = np.asarray(output_depth).astype(np.float32)

    valid = np.isfinite(input_depth) & np.isfinite(output_depth)

    if invalid_val is not None:
        valid = valid & (input_depth != invalid_val) & (output_depth != invalid_val)

    all_valid_values = np.concatenate([
        input_depth[valid].reshape(-1),
        output_depth[valid].reshape(-1),
    ])

    if all_valid_values.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        if percentile_clip is not None:
            low, high = percentile_clip
            vmin = np.percentile(all_valid_values, low)
            vmax = np.percentile(all_valid_values, high)
        else:
            vmin = all_valid_values.min()
            vmax = all_valid_values.max()

        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1e-8

    input_vis = input_depth.copy()
    output_vis = output_depth.copy()

    input_valid = np.isfinite(input_vis)
    output_valid = np.isfinite(output_vis)

    if invalid_val is not None:
        input_valid = input_valid & (input_vis != invalid_val)
        output_valid = output_valid & (output_vis != invalid_val)

    input_vis[~input_valid] = vmin
    output_vis[~output_valid] = vmin

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(input_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Original depth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    im = plt.imshow(output_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Disturbed depth")
    plt.axis("off")

    plt.suptitle(title)
    plt.colorbar(im, ax=plt.gcf().axes, fraction=0.025, pad=0.02)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"[visualize_depth_comparison] Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def read_from_hdf5(filename):
    """
    Load saved trajectories and sensor data from an HDF5 file.
    Returns a dictionary with all available datasets.
    """
    data = {}

    with h5py.File(filename, "r") as f:
        for key in [
            "camera3_depth",
            "fingertip_centered_pos",
            "fingertip_centered_quat",
            "plug_pos",
            "plug_quat",
            "init_plug_pos",
            "init_plug_quat",
            "actions",
            "init_plug_photo_depth",
        ]:
            try:
                data[key] = f[key][()]
                print(f"[read_from_hdf5] Loaded {key}, shape = {data[key].shape}")
            except Exception as e:
                print(f"[read_from_hdf5] Could not read {key}: {e}")

        try:
            grp = f["init_point_list"]
            point_list = []
            for idx in range(len(grp.keys())):
                point_list.append(grp[f"{idx}"][()])
            data["init_point_list"] = point_list
            print(f"[read_from_hdf5] Loaded init_point_list, length = {len(point_list)}")
        except Exception as e:
            print(f"[read_from_hdf5] Could not read init_point_list: {e}")

    return data


def get_depth_disturb_config(strength):
    """
    Parameters controlling disturbance strength.

    light:
        Safe for training. Almost preserves geometry.
    medium:
        Good default for sim-to-real depth augmentation.
    strong:
        Mainly for visualization/debug. May be too destructive for training.
    """
    if strength == "light":
        return {
            "warp_prob": 0.5,
            "max_shift_px": 0.5,
            "grid_size": 48,

            "hole_prob": 0.4,
            "hole_seed_prob": 0.0003,
            "blur_kernel_size": 25,
            "blur_sigma": 6.0,
            "hole_threshold": 0.06,

            "noise_prob": 0.5,
            "noise_std": 0.0005,

            "near_depth_quantile": 99.9,
            "far_depth_quantile": 0.1,
        }

    elif strength == "medium":
        return {
            "warp_prob": 0.7,
            "max_shift_px": 1.5,
            "grid_size": 32,

            "hole_prob": 0.7,
            "hole_seed_prob": 0.0008,
            "blur_kernel_size": 31,
            "blur_sigma": 8.0,
            "hole_threshold": 0.08,

            "noise_prob": 0.7,
            "noise_std": 0.0015,

            "near_depth_quantile": 99.9,
            "far_depth_quantile": 0.1,
        }

    elif strength == "strong":
        return {
            "warp_prob": 0.9,
            "max_shift_px": 3.0,
            "grid_size": 24,

            "hole_prob": 0.9,
            "hole_seed_prob": 0.002,
            "blur_kernel_size": 45,
            "blur_sigma": 12.0,
            "hole_threshold": 0.06,

            "noise_prob": 0.9,
            "noise_std": 0.003,

            "near_depth_quantile": 99.9,
            "far_depth_quantile": 0.1,
        }

    else:
        raise ValueError(
            f"Unknown strength={strength}. "
            "Use one of: 'light', 'medium', 'strong'."
        )


def depth_disturb(input_depth, strength="medium"):
    """
    Disturb IsaacGym-style negative depth image.

    IsaacGym convention:
        valid depth: negative value, usually depth < 0
        invalid depth / missing depth: 0

    This function applies:
        1. local correlated warp
           Simulates edge artifacts / stereo matching jitter.
        2. irregular holes
           Simulates missing depth regions.
        3. small Gaussian depth value noise
           Simulates metric sensor noise.

    Args:
        input_depth:
            np.ndarray, shape (H, W)
        strength:
            "light", "medium", or "strong"

    Returns:
        output_depth:
            np.ndarray, shape (H, W)
    """
    cfg = get_depth_disturb_config(strength)

    depth = np.asarray(input_depth).astype(np.float32).copy()
    depth = np.squeeze(depth)

    assert depth.ndim == 2, f"Expected depth shape (H, W), got {depth.shape}"

    H, W = depth.shape
    eps = 1e-6

    # IsaacGym valid depth should be negative.
    depth[~np.isfinite(depth)] = 0.0
    input_valid = depth < -eps
    depth[~input_valid] = 0.0

    if input_valid.sum() == 0:
        print("[depth_disturb] No valid negative depth pixels.")
        return depth.astype(np.float32)

    # Record a reasonable original valid range.
    # This prevents interpolation artifacts from becoming invalid near-zero depth.
    valid_values = depth[input_valid]
    near_bound = np.percentile(valid_values, cfg["near_depth_quantile"])  # closer to 0
    far_bound = np.percentile(valid_values, cfg["far_depth_quantile"])    # more negative

    # ------------------------------------------------------------
    # 1. Local correlated warp
    # ------------------------------------------------------------
    if np.random.rand() < cfg["warp_prob"]:
        max_shift_px = cfg["max_shift_px"]
        grid_size = cfg["grid_size"]

        low_h = max(2, H // grid_size)
        low_w = max(2, W // grid_size)

        dx_low = np.random.uniform(
            -max_shift_px,
            max_shift_px,
            size=(low_h, low_w),
        ).astype(np.float32)

        dy_low = np.random.uniform(
            -max_shift_px,
            max_shift_px,
            size=(low_h, low_w),
        ).astype(np.float32)

        dx = cv2.resize(dx_low, (W, H), interpolation=cv2.INTER_CUBIC)
        dy = cv2.resize(dy_low, (W, H), interpolation=cv2.INTER_CUBIC)

        xs, ys = np.meshgrid(np.arange(W), np.arange(H))

        map_x = (xs + dx).astype(np.float32)
        map_y = (ys + dy).astype(np.float32)

        depth = cv2.remap(
            depth,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # Remove near-zero interpolation artifacts.
        depth[depth >= -eps] = 0.0

    # ------------------------------------------------------------
    # 2. Irregular depth holes
    # ------------------------------------------------------------
    if np.random.rand() < cfg["hole_prob"]:
        hole_seed_prob = cfg["hole_seed_prob"]
        blur_kernel_size = cfg["blur_kernel_size"]
        blur_sigma = cfg["blur_sigma"]
        hole_threshold = cfg["hole_threshold"]

        seed_mask = (np.random.rand(H, W) < hole_seed_prob).astype(np.float32)

        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1

        smooth_mask = cv2.GaussianBlur(
            seed_mask,
            (blur_kernel_size, blur_kernel_size),
            blur_sigma,
        )

        hole_mask = smooth_mask > hole_threshold

        # Only remove currently valid depth.
        valid_now = depth < -eps
        depth[hole_mask & valid_now] = 0.0

    # ------------------------------------------------------------
    # 3. Gaussian depth value noise
    # ------------------------------------------------------------
    if np.random.rand() < cfg["noise_prob"]:
        noise_std = cfg["noise_std"]

        valid_now = depth < -eps
        noise = np.random.randn(H, W).astype(np.float32) * noise_std

        depth[valid_now] += noise[valid_now]

        # Keep invalid pixels exactly 0.
        depth[~valid_now] = 0.0

    # ------------------------------------------------------------
    # 4. Final cleanup
    # ------------------------------------------------------------
    # Remove invalid / positive / near-zero artifacts.
    depth[~np.isfinite(depth)] = 0.0
    depth[depth >= -eps] = 0.0

    # Remove depth values that became unrealistically close to the camera.
    # For your data, original valid max is around -0.05.
    # If a value becomes -0.001, it is usually an interpolation artifact.
    valid_now = depth < -eps
    too_near = valid_now & (depth > near_bound)
    depth[too_near] = 0.0

    # Clamp overly far values to the original far range.
    valid_now = depth < -eps
    too_far = valid_now & (depth < far_bound)
    depth[too_far] = far_bound

    return depth.astype(np.float32)


if __name__ == "__main__":
    # Change this seed to get different random disturbances.
    np.random.seed(0)

    task_id = "20031"
    hdf_id = 0

    hdf5_path = (
        "/home/ubuntu/automate/IsaacGymEnvs_Assembly/"
        "isaacgymenvs/tasks/automate/data/"
        "recovery_0501_20031/"
        f"asset_{task_id}/disassembly_traj_{hdf_id}.h5"
    )

    data = read_from_hdf5(hdf5_path)

    input_depth = data["camera3_depth"][1][1].astype(np.float32)

    # Choose: "light", "medium", or "strong"
    strength = "medium"
    output_depth = depth_disturb(input_depth, strength=strength)

    os.makedirs("depth_aug_debug", exist_ok=True)

    visualize_depth(
        input_depth,
        save_path=f"depth_aug_debug/depth_before_task_{task_id}_traj_{hdf_id}.png",
        title="Original camera3_depth[1][1]",
        invalid_val=0.0,
        percentile_clip=(1, 99),
        show=False,
    )

    visualize_depth(
        output_depth,
        save_path=f"depth_aug_debug/depth_after_task_{task_id}_traj_{hdf_id}_{strength}.png",
        title=f"Disturbed camera3_depth[1][1]",
        invalid_val=0.0,
        percentile_clip=(1, 99),
        show=False,
    )

    visualize_depth_comparison(
        input_depth,
        output_depth,
        save_path=f"depth_aug_debug/depth_compare_task_{task_id}_traj_{hdf_id}_{strength}.png",
        title=f"Depth disturbance comparison",
        invalid_val=0.0,
        percentile_clip=(1, 99),
        show=False,
    )

    input_valid = np.isfinite(input_depth) & (input_depth < -1e-6)
    output_valid = np.isfinite(output_depth) & (output_depth < -1e-6)

    print("\n========== Depth statistics ==========")
    print("strength:", strength)
    print("input_depth shape:", input_depth.shape)
    print("output_depth shape:", output_depth.shape)

    print("input valid pixels:", input_valid.sum())
    print("output valid pixels:", output_valid.sum())

    if input_valid.sum() > 0:
        print("input valid min:", input_depth[input_valid].min())
        print("input valid max:", input_depth[input_valid].max())
        print("input valid mean:", input_depth[input_valid].mean())

    if output_valid.sum() > 0:
        print("output valid min:", output_depth[output_valid].min())
        print("output valid max:", output_depth[output_valid].max())
        print("output valid mean:", output_depth[output_valid].mean())

    dropped = input_valid.sum() - output_valid.sum()
    drop_ratio = dropped / max(input_valid.sum(), 1)

    print("dropped valid pixels:", dropped)
    print("dropped ratio:", drop_ratio)

    print("\nSaved visualizations to:")
    print(f"  depth_aug_debug/depth_before_task_{task_id}_traj_{hdf_id}.png")
    print(f"  depth_aug_debug/depth_after_task_{task_id}_traj_{hdf_id}_{strength}.png")
    print(f"  depth_aug_debug/depth_compare_task_{task_id}_traj_{hdf_id}_{strength}.png")