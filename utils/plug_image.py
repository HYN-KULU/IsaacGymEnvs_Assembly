import h5py
import numpy as np
from diffusion_utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform
import torch
import plotly.graph_objects as go
import os 
from PIL import Image
from pointcloud_rgbd import render_top_down_custom
from pointcloud_rgbd_bottom_up import render_bottom_up_custom
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
        # for key in ["mask"]:
        for key in ["fingertip_centered_pos", "fingertip_centered_quat", "actions", "init_plug_pos", "plug_pos", "init_plug_quat", "plug_quat", "camera3_vinv", "camera3_proj", "force", "init_socket_photo_depth", "init_socket_photo_rgb", "mask", "init_socket_photo_top", "init_plug_photo_top", "init_plug_photo_rgb", "init_plug_photo_depth"]:
            try:
                data[key] = f[key][()]   # load as numpy array
            except Exception as e:
                print(f"Could not read {key}: {e}, filename: {filename}")
    return data


import torch

import torch.nn.functional as F

def depth_to_pointcloud_with_rgb(
    depth,
    rgb,
    cam_vinv,
    cam_proj,
    mask=None,       # Added mask argument
    depth_bar=10.0,
    device="cuda",
):
    # ... (Existing tensor conversion code) ...
    depth = torch.tensor(depth, dtype=torch.float32, device=device)
    rgb = torch.tensor(rgb, dtype=torch.float32, device=device) / 255.0
    cam_vinv = torch.tensor(cam_vinv, dtype=torch.float32, device=device)
    cam_proj = torch.tensor(cam_proj, dtype=torch.float32, device=device)

    H, W = depth.shape

    # 1. Handle Masking and Surroundings
    if mask is not None:
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        # Dilation: Expand the mask by ~50 pixels to include the surrounding table
        # kernel_size controls how much of the "surroundings" you see
        kernel_size = 51 
        padding = kernel_size // 2
        dilated_mask = F.max_pool2d(mask_t, kernel_size=kernel_size, stride=1, padding=padding)
        valid_mask = dilated_mask.squeeze() > 0
    else:
        valid_mask = torch.ones_like(depth, dtype=torch.bool)

    # 2. Meshgrid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    # 3. Intrinsics and Camera Coords
    fu = 2.0 / cam_proj[0, 0]
    fv = 2.0 / cam_proj[1, 1]
    centerU, centerV = W / 2.0, H / 2.0

    Z = depth
    X = -(u - centerU) / W * Z * fu
    Y =  (v - centerV) / H * Z * fv

    # 4. Apply combined Mask (Depth + Socket Mask)
    # This filters out the background/faraway points early
    valid = (Z > -depth_bar) & valid_mask

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    rgb = rgb.reshape(-1, 3)[valid.reshape(-1)]

    # 5. Transform to World
    ones = torch.ones_like(X)
    position = torch.stack([X, Y, Z, ones], dim=1)
    position_world = position @ cam_vinv
    points = position_world[:, :3]

    return points, rgb

def render_custom_view(points, colors, H, W, fov_deg=60, device="cuda"):
    """
    Render point cloud with:
    - Image right = World +Y
    - Image up = World -X  
    - Camera looks toward World -Z
    """
    import torch
    import numpy as np
    
    # Convert to tensors
    points = torch.tensor(points, dtype=torch.float32, device=device)
    colors = torch.tensor(colors, dtype=torch.float32, device=device)
    
    # 1. Build camera-to-world matrix (cam_vinv)
    # This matrix transforms points from camera space to world space
    cam_vinv = torch.eye(4, device=device)
    
    # Camera axes in world coordinates
    right_world = torch.tensor([0.0, 1.0, 0.0], device=device)   # +Y
    up_world = torch.tensor([-1.0, 0.0, 0.0], device=device)     # -X
    forward_world = torch.tensor([0.0, 0.0, -1.0], device=device) # -Z
    
    # Normalize
    right_world = right_world / torch.norm(right_world)
    up_world = up_world / torch.norm(up_world)
    forward_world = forward_world / torch.norm(forward_world)
    
    # Set rotation (camera axes in world coordinates)
    cam_vinv[0, :3] = right_world   # Camera X axis (right) in world
    cam_vinv[1, :3] = up_world      # Camera Y axis (up) in world
    cam_vinv[2, :3] = forward_world # Camera Z axis (forward) in world
    
    # Set camera position (where to place the camera)
    # Let's place it at (0, 0, 5) looking at origin
    camera_pos = torch.tensor([0.0, 0.0, 5.0], device=device)
    cam_vinv[:3, 3] = camera_pos
    
    # 2. Build projection matrix
    aspect = W / H
    fov_rad = torch.tensor(fov_deg * np.pi / 180.0, device=device)
    f = 1.0 / torch.tan(fov_rad / 2.0)
    
    cam_proj = torch.zeros(4, 4, device=device)
    cam_proj[0, 0] = f / aspect
    cam_proj[1, 1] = f
    cam_proj[2, 2] = -1.0  # Different sign for OpenGL style
    cam_proj[2, 3] = -0.1  # Near plane
    cam_proj[3, 2] = -1.0
    
    # 3. Transform points: world -> camera
    ones = torch.ones(points.shape[0], 1, device=device)
    points_world_h = torch.cat([points, ones], dim=1)
    
    # Inverse of cam_vinv (world to camera)
    cam_vinv_inv = torch.inverse(cam_vinv)
    points_cam_h = points_world_h @ cam_vinv_inv
    
    # 4. Project to NDC
    points_ndc = points_cam_h @ cam_proj
    
    # 5. Convert to pixel coordinates
    u = (points_ndc[:, 0] / points_ndc[:, 3] + 1.0) * 0.5 * W
    v = (points_ndc[:, 1] / points_ndc[:, 3] + 1.0) * 0.5 * H
    Z = points_cam_h[:, 2]
    
    # 6. Filter valid points
    valid = (Z > 0) & (Z < 10.0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    if valid.sum() == 0:
        print("No valid points!")
        return None, None, None
    
    u = u[valid].round().long()
    v = v[valid].round().long()
    Z = Z[valid]
    colors = colors[valid]
    
    # 7. Render with z-buffer
    depth_img = torch.full((H, W), 10.0, device=device)
    rgb_img = torch.zeros(H, W, 3, device=device)
    mask_img = torch.zeros(H, W, dtype=torch.bool, device=device)
    
    for i in range(len(u)):
        vi, ui = v[i], u[i]
        if Z[i] < depth_img[vi, ui]:  # Keep closest
            depth_img[vi, ui] = Z[i]
            rgb_img[vi, ui] = colors[i]
            mask_img[vi, ui] = True
    
    return depth_img, rgb_img, mask_img



def save_depth_visualization(depth_array, save_path):
    if hasattr(depth_array, "detach"):
        depth_np = depth_array.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth_array)

    valid_mask = np.isfinite(depth_np)

    if not valid_mask.any():
        depth_viz = np.zeros(depth_np.shape, dtype=np.uint8)
    else:
        valid_depth = depth_np[valid_mask]
        d_min = valid_depth.min()
        d_max = valid_depth.max()

        depth_viz = np.zeros(depth_np.shape, dtype=np.uint8)
        depth_norm = (valid_depth - d_min) / (d_max - d_min + 1e-8)
        depth_viz_valid = (255 - depth_norm * 255).astype(np.uint8)
        depth_viz[valid_mask] = depth_viz_valid

    Image.fromarray(depth_viz).save(save_path)

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image



def save_rgb_float_image(rgb_img, save_path):
    """
    Save (H, W, 3) float image in [0,1] to png.
    """
    if hasattr(rgb_img, "detach"):
        rgb_np = rgb_img.detach().cpu().numpy()
    else:
        rgb_np = np.asarray(rgb_img)

    rgb_np = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_np).save(save_path)
    print(f"RGB image saved to {save_path}")


def save_rgb_image(rgb, save_path):
    """
    Saves a NumPy array RGB image to a file.
    
    Args:
        rgb: numpy array of shape (H, W, 3)
        save_path: string path where the image will be saved
    """
    # Ensure it's a numpy array (in case a torch tensor was passed)
    if hasattr(rgb, 'detach'):
        rgb = rgb.detach().cpu().numpy()
    
    # Ensure the data type is uint8 (standard for 0-255 RGB images)
    if rgb.dtype != np.uint8:
        # If data is in [0, 1] range, scale it
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

    # Convert to PIL Image and save
    img = Image.fromarray(rgb)
    img.save(save_path)
    print(f"Original RGB image saved to {save_path}")

def save_depth_visualization(depth_array, save_path):
    if hasattr(depth_array, "detach"):
        depth_np = depth_array.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth_array)

    # 同时排除 +inf, -inf, nan
    valid_mask = np.isfinite(depth_np)

    if not valid_mask.any():
        print("Warning: no valid depth values")
        depth_viz = np.zeros(depth_np.shape, dtype=np.uint8)
    else:
        valid_depth = depth_np[valid_mask]
        d_min = valid_depth.min()
        d_max = valid_depth.max()

        depth_viz = np.zeros(depth_np.shape, dtype=np.uint8)

        # 只对有效像素做归一化
        depth_norm = (valid_depth - d_min) / (d_max - d_min + 1e-8)

        # 近的亮，远的暗
        depth_viz_valid = (255 - depth_norm * 255).astype(np.uint8)

        depth_viz[valid_mask] = depth_viz_valid

    Image.fromarray(depth_viz).save(save_path)
    print(f"Depth image saved to {save_path}")

def visualize_pointcloud_plotly(points, colors, max_points=20000, save_path="pc.html"):
    import numpy as np
    import plotly.graph_objects as go
    import torch

    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.detach().cpu().numpy()

    N = points.shape[0]

    # downsample (same idx!)
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        pts = points[idx]
        cols = colors[idx]
    else:
        pts = points
        cols = colors

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # plotly needs color as string or rgb
    cols_plotly = [
        f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
        for r, g, b in cols
    ]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=cols_plotly,
                    opacity=0.9,
                )
            )
        ]
    )

    fig.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40),
        title="Colored Point Cloud"
    )

    fig.write_html(save_path)
    print(f"Saved to {save_path}")

import torch
import numpy as np


if __name__=="__main__":
    train_tasks = [
       "40009"
    ]
    hdf_ids = 0
    timestep_index = {}
    force_list = []

    for task_id in train_tasks:
        for hdf_id in range(6):
            data = read_from_hdf5(
                f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_0416_forward_force_photo_debug/asset_{task_id}/disassembly_traj_{hdf_id}.h5"
            )
            env_id = 1
            depth = data["init_plug_photo_depth"][env_id]   # (960, 1280)
            rgb = data["init_plug_photo_rgb"][env_id]       # (960, 1280, 3)
            cam_vinv_tensor = torch.tensor([[ 9.9504e-02,  9.9504e-01,  0.0000e+00,  0.0000e+00],
            [ 9.9463e-01, -9.9463e-02,  2.8702e-02,  0.0000e+00],
            [ 2.8560e-02, -2.8560e-03, -9.9959e-01,  0.0000e+00],
            [ 2.7940e-09, -2.0000e-01,  4.5000e-01,  1.0000e+00]])
            cam_vinv = cam_vinv_tensor.cpu().numpy()        # (4, 4) # 350 12 4 4
            cam_proj = data["camera3_proj"][0][env_id]        # (4, 4) # 350 12 4 4
            mask = data["mask"][0][env_id]                       #  uint8 (960, 1280) # 350 12 960 1280
            save_depth_visualization(data["init_plug_photo_depth"][env_id]  , "init_plug_depth.png")
            orig_rgb_path= f"./task_{task_id}_traj_{hdf_id}_orig_rgb_plug_bottom.png"
            save_rgb_image(data["init_plug_photo_rgb"][env_id]  , orig_rgb_path)
            points, colors = depth_to_pointcloud_with_rgb(
                depth,
                rgb,
                cam_vinv,
                cam_proj,
                mask=None, # Pass the mask here
                depth_bar=10.0,
                device="cuda"
            )

            # point cloud html
            save_path = f"./task_{task_id}_traj_{hdf_id}.html"
            visualize_pointcloud_plotly(
                points,
                colors,
                max_points=150000,
                save_path=save_path
            )
            torch.save({
                "points": points,
                "colors": colors
            }, f"./pointcloud.pth")
            rgb_img, depth_img = render_bottom_up_custom(points, colors, H=960, W=1280, camera_height_offset=0.08, fov_deg=30, brightness_scale=2, point_radius=3)
            if rgb_img is not None:
                # Convert to numpy and save
                rgb_np = (rgb_img.cpu().numpy() * 255).astype(np.uint8)
                
                # Save RGB image
                Image.fromarray(rgb_np).save("plug_bottom_up.png")
                print("\n✓ Saved: plug_bottom_up.png")
                
                # Also save depth visualization
                depth_np = depth_img.cpu().numpy()
                depth_normalized = (depth_np / depth_np.max() * 255).astype(np.uint8)
                Image.fromarray(depth_normalized).save("plug_bottom_up_depth.png")
                print("✓ Saved: plug_bottom_up_depth.png")
                
                # Print some debug info
                print("\nDebug Info:")
                print(f"  Image size: {rgb_np.shape}")
                print(f"  Non-black pixels: {(rgb_np.sum(axis=2) > 0).sum()}")
                
                # Check colors at specific pixels (center, edges)
                center_u, center_v = 320, 240
                print(f"\n  Color at center (u={center_u}, v={center_v}): {rgb_np[center_v, center_u]}")
                
                # Expected: Center x≈1, y≈0 should have red≈0.5, blue≈0.5
                print("  Expected center: red~128, blue~128 (since x=1→red=0.5, y=0→blue=0.5)")
                
            else:
                print("Failed to render!")

            os._exit(0)