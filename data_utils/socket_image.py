import h5py
import numpy as np
from diffusion_utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform
import torch
import plotly.graph_objects as go
import os 
from PIL import Image
from data_utils.pointcloud_rgbd import render_top_down_custom

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
        for key in ["fingertip_centered_pos", "fingertip_centered_quat", "actions", "init_plug_pos", "plug_pos", "init_plug_quat", "plug_quat", "camera3_vinv", "camera3_proj", "force", "init_socket_photo_depth", "init_socket_photo_rgb", "init_socket_photo_top_rgb", "init_socket_photo_top_depth", "init_plug_photo_top", "init_plug_photo_rgb", "init_plug_photo_depth", ]:
            try:
                data[key] = f[key][()]   # load as numpy array
            except Exception as e:
                print(f"Could not read {key}: {e}, filename: {filename}")
        try:
            grp = f["init_point_list"]
            point_list = []
            for idx in range(len(grp.keys())):
                point_list.append(grp[f"{idx}"][()])  # each is (N, D)

            data["init_point_list"] = point_list

        except Exception as e:
            print(f"Could not read init_point_list: {e}, filename: {filename}")
    return data


import torch

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


# @torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor: torch.Tensor,
                                   colors: torch.Tensor,
                                   camera_view_matrix_inv: torch.Tensor,
                                   camera_proj_matrix: torch.Tensor,
                                   u: torch.Tensor,
                                   v: torch.Tensor,
                                   width: float,
                                   height: float,
                                   depth_bar: float,
                                   device: torch.device,
                                   mask: torch.Tensor = None
                                   ) -> torch.Tensor:
    # Depth and intrinsics
    depth_buffer = camera_tensor.to(device)
    vinv = camera_view_matrix_inv
    proj = camera_proj_matrix
    fu = 2.0 / proj[0, 0]
    fv = 2.0 / proj[1, 1]

    centerU = width / 2.0
    centerV = height / 2.0
    # Project into 3D
    Z = depth_buffer
    X = -(u - centerU) / width * Z * fu
    Y =  (v - centerV) / height * Z * fv

    # Flatten
    Z = Z.reshape(-1)
    X = X.reshape(-1)
    Y = Y.reshape(-1)

    # Flatten colors (H, W, 3) -> (N, 3)
    colors = colors.reshape(-1, 3)

    # Mask valid points
    valid = Z > -depth_bar
        # Apply mask
    if mask is not None:
        mask = mask.reshape(-1).to(torch.bool)
        valid = valid & mask
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]
    colors = colors[valid]

    # Homogeneous coords
    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device))).permute(1, 0)
    position = position @ vinv
    points = position[:, 0:3]  # (N, 3)

    # Concatenate with colors -> (N, 6)
    points_rgb = torch.cat((points, colors), dim=1)
    return points_rgb

def filter_workspace(points: torch.Tensor,
                     x_range: tuple,
                     y_range: tuple,
                     z_range: tuple) -> torch.Tensor:
    """
    Filter points within a 3D workspace box.

    Args:
        points: (N, 3) torch.Tensor, world coordinates
        x_range: (min_x, max_x)
        y_range: (min_y, max_y)
        z_range: (min_z, max_z)

    Returns:
        filtered_points: (M, 3) torch.Tensor
    """
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    return points[mask]

def quat_xyzw_to_matrix(quat):
    """
    quat: (..., 4) in (x, y, z, w) order
    returns: (..., 3, 3)
    """
    if not torch.is_tensor(quat):
        quat = torch.tensor(quat, dtype=torch.float32)
    else:
        quat = quat.float()

    x, y, z, w = quat.unbind(-1)

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w

    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    R = torch.stack([
        ww + xx - yy - zz,  2 * (xy - zw),      2 * (xz + yw),
        2 * (xy + zw),      ww - xx + yy - zz,  2 * (yz - xw),
        2 * (xz - yw),      2 * (yz + xw),      ww - xx - yy + zz
    ], dim=-1).reshape(quat.shape[:-1] + (3, 3))

    return R

def get_inverse_relative_rotation(q_init, q_final):
    """
    q_init, q_final: (4,) quaternions in (x, y, z, w)
    returns:
        R_rel: init -> final rotation
        R_inv: inverse rotation to apply to socket
    """
    R_init = quat_xyzw_to_matrix(q_init)   # (3, 3)
    R_final = quat_xyzw_to_matrix(q_final) # (3, 3)

    R_rel = R_final @ R_init.T
    R_inv = R_rel.T
    return R_rel, R_inv

def rotate_pointcloud(points, R, center=None):
    """
    points: (N, 3)
    R: (3, 3)
    center: (3,) rotation center. If None, use centroid.

    returns:
        rotated_points: (N, 3)
    """
    if not torch.is_tensor(points):
        points = torch.tensor(points, dtype=torch.float32)
    else:
        points = points.float()

    if not torch.is_tensor(R):
        R = torch.tensor(R, dtype=torch.float32)
    else:
        R = R.float()

    if center is None:
        center = points.mean(dim=0)
    else:
        if not torch.is_tensor(center):
            center = torch.tensor(center, dtype=torch.float32)
        else:
            center = center.float()

    points_centered = points - center[None, :]
    rotated = points_centered @ R.T
    rotated = rotated + center[None, :]
    return rotated

def extract_yaw_from_rotation_matrix(R):
    """
    R: (3, 3)
    returns yaw angle in radians
    Assumes yaw around world Z.
    """
    yaw = torch.atan2(R[1, 0], R[0, 0])
    return yaw


def yaw_matrix(theta):
    c = torch.cos(theta)
    s = torch.sin(theta)
    R = torch.tensor([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=theta.device if torch.is_tensor(theta) else None)
    return R

def get_inverse_yaw_only_rotation(q_init, q_final):
    R_init = quat_xyzw_to_matrix(q_init)
    R_final = quat_xyzw_to_matrix(q_final)

    R_rel = R_final @ R_init.T
    yaw = extract_yaw_from_rotation_matrix(R_rel)

    R_yaw = yaw_matrix(yaw)
    R_inv = R_yaw.T
    return R_yaw, R_inv, yaw

def quat_xyzw_to_yaw(q):
    x, y, z, w = q
    return np.arctan2(
        2*(w*z + x*y),
        1 - 2*(y*y + z*z)
    )

def relative_yaw_from_quats(q_init, q_final):
    yaw_init = quat_xyzw_to_yaw(q_init)
    yaw_final = quat_xyzw_to_yaw(q_final)
    delta = yaw_final - yaw_init
    return np.arctan2(np.sin(delta), np.cos(delta))
import cv2

def rotate_image(image, angle_deg, interp=cv2.INTER_LINEAR, border_value=0):
    """
    image: HxW or HxWxC, numpy array or torch tensor
    angle_deg: positive means CCW in OpenCV
    """
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated

def rotate_socket_image_opposite_gripper(
    init_socket_img,
    gripper_init_quat,
    gripper_final_quat,
):
    delta_yaw = relative_yaw_from_quats(gripper_init_quat, gripper_final_quat)
    delta_yaw_deg = float(delta_yaw * 180.0 / np.pi)
    print(f"Rotating socket image by {-delta_yaw_deg:.2f} degrees to compensate for gripper yaw change of {delta_yaw_deg:.2f} degrees")
    if torch.is_tensor(init_socket_img):
        init_socket_img = init_socket_img.detach().cpu().numpy()

    init_socket_img = np.asarray(init_socket_img)

    rotated_img = rotate_image(init_socket_img, -delta_yaw_deg)

    return rotated_img, delta_yaw_deg

def rotate_socket_depth_opposite_gripper(
    init_socket_depth,
    gripper_init_quat,
    gripper_final_quat,
    invalid_val=0.0,
):
    """
    init_socket_depth: (H, W) numpy array or torch tensor
    quats: (x, y, z, w)

    Returns:
        rotated_depth, delta_yaw_deg
    """
    delta_yaw = relative_yaw_from_quats(gripper_init_quat, gripper_final_quat)
    delta_yaw_deg = float(delta_yaw * 180.0 / np.pi)

    if torch.is_tensor(init_socket_depth):
        init_socket_depth = init_socket_depth.detach().cpu().numpy()

    init_socket_depth = np.asarray(init_socket_depth, dtype=np.float32)

    h, w = init_socket_depth.shape
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, -delta_yaw_deg, 1.0)

    rotated_depth = cv2.warpAffine(
        init_socket_depth,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(invalid_val),
    )

    return rotated_depth, delta_yaw_deg

if __name__=="__main__":
    train_tasks = [
       "100003"
    ]
    hdf_ids = 0
    timestep_index = {}
    force_list = []

    for task_id in train_tasks:
        for hdf_id in range(1):
            data = read_from_hdf5(
                f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/sim2real_0513/asset_{task_id}/disassembly_traj_{hdf_id}.h5"
            )
            env_id = 1
            depth = data["init_socket_photo_depth"][env_id]   # (960, 1280)
            rgb = data["init_socket_photo_rgb"][env_id]       # (960, 1280, 3)
            cam_proj = data["camera3_proj"][0][env_id]        # (4, 4) # 350 12 4 4
            cam_proj = torch.from_numpy(cam_proj).float().to("cuda")
            # import pdb;pdb.set_trace()
            orig_rgb_path = f"./task_{task_id}_traj_{hdf_id}_orig_rgb.png"
            save_rgb_image(rgb, orig_rgb_path)
            save_rgb_image(data["init_socket_photo_top_rgb"][env_id], f"./task_{task_id}_traj_{hdf_id}_orig_top_rgb.png")
            save_rgb_image(data["init_plug_photo_rgb"][env_id], f"./task_{task_id}_traj_{hdf_id}_orig_plug_rgb.png")
            save_rgb_image(np.flipud(data["init_plug_photo_rgb"][env_id]), f"./task_{task_id}_traj_{hdf_id}_orig_plug_rgb_flipped.png")
            save_depth_visualization(data["init_plug_photo_depth"][env_id], f"./task_{task_id}_traj_{hdf_id}_orig_plug_depth.png")
            points = data["init_point_list"][env_id]   # list of (N, D), D>=6, last 3 are RGB
            # point cloud html
            save_path = f"./task_{task_id}_traj_{hdf_id}.html"
            visualize_pointcloud_plotly(
                points = points[:,:3],
                colors = points[:,3:],
                max_points=150000,
                save_path=save_path
            )
            points=torch.from_numpy(points).float().to("cpu")
            q_init = data["fingertip_centered_quat"][env_id,0]
            rgb_init, depth_init = render_top_down_custom(points[:,:3], points[:,3:], H=480, W=640, camera_height_offset=0.02, fov_deg=73.73979365244269, point_radius=5)
            for t in range(data["fingertip_centered_quat"].shape[1]):
                if t ==0:
                    rgb_img = rgb_init.clone().detach().cpu().numpy()
                    depth_img = depth_init.clone().detach().cpu().numpy()
                else:
                    q_final = data["fingertip_centered_quat"][env_id,t]
                    # import pdb;pdb.set_trace()
                    # R_yaw, R_inv, yaw = get_inverse_yaw_only_rotation(q_init, q_final)
                    # socket_center = points[:,:3].mean(dim=0)
                    # rotated_socket_points = rotate_pointcloud(points[:,:3], R_inv, center=socket_center)
                    # rgb_img, depth_img = render_top_down_custom(rotated_socket_points, points[:,3:], H=960, W=1280, camera_height_offset=0.08, fov_deg=30, point_radius=3)
                    rgb_img, _ = rotate_socket_image_opposite_gripper(rgb_init, q_init, q_final)
                    depth_img,_ = rotate_socket_depth_opposite_gripper(depth_init, q_init, q_final, invalid_val=0.0)
                if rgb_img is not None:
                    rgb_np = (rgb_img * 255).astype(np.uint8)
                    Image.fromarray(rgb_np).save(f"rotated_socket_image/socket_top_down_{t}.png")
                    depth_np = depth_img
                    depth_normalized = (depth_np / depth_np.max() * 255).astype(np.uint8)
                    Image.fromarray(depth_normalized).save(f"rotated_socket_depth_image/socket_top_down_depth_{t}.png")
                    
                    
                    
                else:
                    print("Failed to render!")

            os._exit(0)