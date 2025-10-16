import numpy as np
import cv2
import os
import imageio

# path to your depth files
depth_dir = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/depth_relative"
out_path = "depth_video.mp4"
fps = 20

# load file list (0 → 179)
files = [f"depth_{i}.npy" for i in range(180)]

frames = []

for f in files:
    depth = np.load(os.path.join(depth_dir, f))
    
    # ✅ clip invalid far-away depth values
    depth = np.clip(depth, -0.5, 0.0)   # keep range [-0.5, 0]
    
    # normalize to 0-255 for visualization
    depth_min, depth_max = depth.min(), depth.max()
    depth_vis = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    
    # apply colormap for better visualization
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
    frames.append(depth_color)

# save as mp4 video
imageio.mimsave(out_path, frames, fps=fps)
print(f"Saved depth video to {out_path}")
