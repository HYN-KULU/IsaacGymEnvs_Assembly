import os
import zipfile
import io
import numpy as np
import cv2
import imageio

def load_depth_from_zip(zip_path):
    """Read compressed depth npz file from a .zip"""
    with zipfile.ZipFile(zip_path, "r") as zf:
        # find the .npz file (usually 'depth_part.npz')
        npz_name = [name for name in zf.namelist() if name.endswith(".npz")][0]
        with zf.open(npz_name) as f:
            buf = io.BytesIO(f.read())
            data = np.load(buf)
            depth = data["depth"]
    return depth

def visualize_first_trajectory(depth, out_path, fps=20):
    """
    depth: (N, 480, 640)
    Assume first 180 frames belong to trajectory 0
    """
    traj0 = depth[:180]
    frames = []
    for d in traj0:
        d = np.clip(d, -0.5, 0.0)
        d_min, d_max = d.min(), d.max()
        d_vis = (d - d_min) / (d_max - d_min + 1e-8)
        d_vis = (d_vis * 255).astype(np.uint8)
        d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_PLASMA)
        frames.append(d_color)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"🎥 Saved trajectory 0 video → {out_path}")

if __name__ == "__main__":
    zip_path = "data/depth_relative_multitask_zip/depth_0_part1.zip"
    print(f"Loading from: {zip_path}")

    depth = load_depth_from_zip(zip_path)
    print(f"✅ Loaded depth shape: {depth.shape}")  # should be (1080, 480, 640)

    # visualize first trajectory (first 180 frames)
    out_path = "depth_traj0_preview.mp4"
    visualize_first_trajectory(depth, out_path)
