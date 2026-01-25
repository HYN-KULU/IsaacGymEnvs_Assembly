import argparse
import os
from collections import OrderedDict

import torch
import numpy as np
import cv2
import imageio
import numpy as np
import torch
import cv2
from dataset.flow_dataset import DepthActionDataset
from policy.flow_policy import FlowPolicy
import h5py

def read_from_hdf5(filename):
    with h5py.File(filename, "r") as f:
        return {"rgb":f["camera3_rgb"][()][::-1,...].copy(),"mask": f["mask"][()][::-1,...].copy(),"depth":f["camera3_depth"][()][::-1,...].copy(), "gripper_pos":f["fingertip_centered_pos"][()][:,::-1,...].copy(),"gripper_rot":f["fingertip_centered_quat"][()][:,::-1,...].copy(), "camera3_vinv": f["camera3_vinv"][()][::-1,...].copy(), "camera3_proj": f["camera3_proj"][()][::-1,...].copy()}     # (T, 12, 480, 640, 3)

def shift_np_integer(x, dx, dy):
    """
    x: (C,H,W)
    dx, dy: integers
    """
    dx = int(round(dx))
    dy = int(round(dy))

    y = np.roll(x, shift=(dy, dx), axis=(1, 2))

    # zero out wrapped region (important!)
    if dy > 0:
        y[:, :dy, :] = 0
    elif dy < 0:
        y[:, dy:, :] = 0

    if dx > 0:
        y[:, :, :dx] = 0
    elif dx < 0:
        y[:, :, dx:] = 0

    return y
# ------------------------------------------------------
# Optical flow → HSV color
# ------------------------------------------------------
def flow_to_hsv(flow):
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()

    u = flow[0]
    v = flow[1]

    mag = np.sqrt(u * u + v * v)
    ang = np.arctan2(v, u)

    H, W = u.shape
    hsv = np.zeros((H, W, 3), dtype=np.uint8)

    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ------------------------------------------------------
# Draw sparse arrows
# ------------------------------------------------------
def draw_arrows(img, flow, stride=12, scale=4):
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()

    out = img.copy()
    u = flow[0]
    v = flow[1]
    H, W = u.shape

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            dx = u[y, x]
            dy = v[y, x]

            x2 = int(x + dx * scale)
            y2 = int(y + dy * scale)

            cv2.arrowedLine(
                out,
                (x, y),
                (x2, y2),
                color=(0, 0, 255),
                thickness=1,
                tipLength=0.3,
            )
    return out


# ------------------------------------------------------
# MAIN visualization
# ------------------------------------------------------
def visualize_flow_and_mask(flow, mask, rgb, rate=2, prefix="vis"):
    """
    flow: (2, 480, 640)
    mask: (1, 1, 240, 320) or (240, 320)
    rgb : (3, 480, 640) or (480, 640, 3)

    Output:
    - Left : RGB image
    - Right: white-mask background + flow arrows
    """

    # -------------------------
    # Prepare RGB
    # -------------------------
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()

    if rgb.shape[0] == 3:
        rgb_img = np.transpose(rgb, (1, 2, 0))  # (H,W,3)
    else:
        rgb_img = rgb.copy()

    rgb_img = rgb_img.astype(np.uint8)
    rgb_img = rgb_img[..., ::-1]

    # -------------------------
    # Prepare flow
    # -------------------------
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()

    u = flow[0]
    v = flow[1]
    H, W = u.shape

    # -------------------------
    # Prepare mask
    # -------------------------
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    mask_2d = mask.squeeze()

    if mask_2d.shape != (H, W):
        mask_2d = cv2.resize(
            mask_2d.astype(np.uint8),
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_2d = mask_2d > 0

    # -------------------------
    # Right panel: white bg + arrows
    # -------------------------
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[mask_2d] = 255

    stride = rate * 6
    scale = 12

    for y in range(0, H, stride):
        for x in range(0, W, stride):

            if not mask_2d[y, x]:
                continue

            dx = u[y, x]
            dy = v[y, x]

            if dx == 0 and dy == 0:
                continue

            x2 = int(x + dx * scale)
            y2 = int(y - dy * scale)

            cv2.arrowedLine(
                canvas,
                (x, y),
                (x2, y2),
                color=(0, 0, 255),  # red (BGR)
                thickness=1,
                tipLength=0.35,
                line_type=cv2.LINE_AA,
            )

    # -------------------------
    # Concatenate: left RGB | right flow
    # -------------------------
    vis = np.concatenate([rgb_img, canvas], axis=1)

    # -------------------------
    # Save
    # -------------------------
    cv2.imwrite(f"{prefix}_rgb_flow_mask_arrow.png", vis)
    print(f"[✓] Saved: {prefix}_rgb_flow_mask_arrow.png")


import torch.nn.functional as F

# ============================================================
# Final egomotion optical flow implementation (Isaac Gym)
# ------------------------------------------------------------
# - Uses instantaneous egomotion (twist)
# - Converts gripper motion from world frame -> camera frame
# - Handles Isaac Gym negative depth correctly
# - All assumptions & explanations are in comments only
# ============================================================

import numpy as np
from scipy.spatial.transform import Rotation as R


# ------------------------------------------------------------
# Compute camera-frame twist (t, omega) from gripper world poses
# ------------------------------------------------------------
import numpy as np
from scipy.spatial.transform import Rotation as R

import cv2
import numpy as np
import plotly.graph_objects as go


def plot_frame_plotly(
    fig,
    R_wf,          # (3,3) frame axes expressed in world
    p_wf,          # (3,)
    name,
    axis_len=0.05,
):
    """
    Add a coordinate frame to a plotly 3D figure.

    Convention:
      X = red
      Y = green
      Z = blue
    """

    colors = ["red", "green", "blue"]
    labels = ["X", "Y", "Z"]

    for i in range(3):
        axis = R_wf[:, i]

        fig.add_trace(go.Scatter3d(
            x=[p_wf[0], p_wf[0] + axis[0] * axis_len],
            y=[p_wf[1], p_wf[1] + axis[1] * axis_len],
            z=[p_wf[2], p_wf[2] + axis[2] * axis_len],
            mode="lines+markers+text",
            line=dict(color=colors[i], width=6),
            marker=dict(size=4),
            text=[None, f"{name}:{labels[i]}"],
            textposition="top center",
            name=f"{name}-{labels[i]}",
            showlegend=False,
        ))

    # origin label
    fig.add_trace(go.Scatter3d(
        x=[p_wf[0]],
        y=[p_wf[1]],
        z=[p_wf[2]],
        mode="text",
        text=[name],
        textposition="bottom center",
        showlegend=False,
    ))
import numpy as np
import matplotlib.pyplot as plt
def visualize_frames_in_world_plotly(
    R_wg, p_wg,
    R_wc, p_wc,
    save_html="frames_in_world.html",
):
    """
    Interactive 3D visualization using Plotly.
    """

    fig = go.Figure()

    # World frame
    plot_frame_plotly(
        fig,
        R_wf=np.eye(3),
        p_wf=np.zeros(3),
        name="World",
        axis_len=0.08,
    )

    # Gripper frame
    plot_frame_plotly(
        fig,
        R_wf=R_wg,
        p_wf=p_wg,
        name="Gripper",
        axis_len=0.06,
    )

    # Camera frame
    plot_frame_plotly(
        fig,
        R_wf=R_wc,
        p_wf=p_wc,
        name="Camera",
        axis_len=0.06,
    )

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="World X",
            yaxis_title="World Y",
            zaxis_title="World Z",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title="World / Gripper / Camera Frames",
    )

    fig.write_html(save_html)
    print(f"[✓] Saved interactive plot to: {save_html}")

def world_pose_to_camera_twist(p0, q0, p1, q1,vinv, dt=1.0):
    """
    p*: (3,) world-frame position
    q*: (4,) world-frame quaternion (x, y, z, w)

    Returns:
        t_cam     : (3,) camera-frame linear velocity
        omega_cam : (3,) camera-frame angular velocity

    Notes:
    - Egomotion flow formula assumes BOTH t and omega are expressed
      in the camera (body) frame.
    - We therefore:
        1) compute world-frame twist
        2) rotate it into camera frame using R_wc^T
    """

    # world-frame linear velocity
    t_world = (p1 - p0) / dt
    
    # world-frame angular velocity (via relative rotation)
    r0 = R.from_quat(q0)
    r1 = R.from_quat(q1)
    r_rel = r1 * r0.inv()
    omega_world = r_rel.as_rotvec() / dt
    R_wc = vinv[:3, :3]
    t_cam =R_wc@t_world 
    print("p0:",p0)
    print("p1:",p1)
    print("t world:",t_world)
    print("t cam:",t_cam)
    print("***************************")
    omega_cam=R_wc@omega_world

    return t_cam, omega_cam
# ------------------------------------------------------------
# Egomotion optical flow (motion field)
# ------------------------------------------------------------
def egomotion_flow_from_depth(
    depth, t_cam, omega_cam,
    fx, fy, cx, cy
):
    """
    depth: (H, W) Isaac Gym depth (NEGATIVE values!)
    t_cam: (3,) camera-frame translation velocity
    omega_cam: (3,) camera-frame angular velocity

    Returns:
        flow: (2, H, W)  [u, v]

    Notes:
    - Isaac Gym depth is negative along camera optical axis
      -> true Z = -depth
    - Formula is instantaneous (first-order), no camera position needed
    """

    H, W = depth.shape
    # Convert Isaac Gym depth to positive Z
    Z = -depth
    Z = np.maximum(Z, 1e-6)  # avoid division by zero

    # Pixel -> normalized image coordinates
    xs = (np.arange(W) - cx) / fx
    ys = (np.arange(H) - cy) / fy
    x, y = np.meshgrid(xs, ys)

    tx, ty, tz = t_cam
    tz = -tz
    wx, wy, wz = omega_cam
    # print(tx,ty,tz)
    # Translation-induced flow (depth dependent)
    u_t = (-tx + x * tz) / Z
    v_t = (-ty + y * tz) / Z
    # import pdb;pdb.set_trace()
    # Rotation-induced flow (depth independent)
    u_r = x * y * wx - (1 + x**2) * wy + y * wz
    v_r = (1 + y**2) * wx - x * y * wy - x * wz

    # Total egomotion flow
    u = u_t + u_r
    v = v_t + v_r
    # u = u_t
    # v = v_t

    return np.stack([u, v], axis=0)


# ------------------------------------------------------------
# Main driver: compute flow for all trajectories & timesteps
# ------------------------------------------------------------
def compute_all_egomotion_flows(
    data, fx, fy, cx, cy, dt=1.0
):
    """
    data keys expected:
        data["depth"]        : (T, N, H, W)
        data["gripper_pos"]  : (N, T, 3)
        data["gripper_rot"]  : (N, T, 4)

    Returns:
        flows: (T-1, N, 2, H, W)
    """
    # Reverse the whole trajectory
    depth = data["depth"] # (T, N, H, W)
    gripper_pos = data["gripper_pos"] # N, T, 3
    gripper_rot = data["gripper_rot"] # N, T, 4
    vinv = data["camera3_vinv"] # (T, N, 4, 4)
    T, N, H, W = depth.shape
    flows = np.zeros((T - 1, N, 2, H, W), dtype=np.float32)

    for traj_id in range(N):
        for t in range(T - 1):

            # camera-frame twist from gripper motion
            t_cam, omega_cam = world_pose_to_camera_twist(
                gripper_pos[traj_id, t],
                gripper_rot[traj_id, t],
                gripper_pos[traj_id, t + 1],
                gripper_rot[traj_id, t + 1],
                vinv[t, traj_id],
                dt=dt
            )

            # egomotion flow for this frame
            flow = egomotion_flow_from_depth(
                depth[t, traj_id],
                t_cam, omega_cam,
                fx, fy, cx, cy
            )

            flows[t, traj_id] = flow

    return flows

def process_dataset_with_egomotion_flow(
    data,
    fx, fy, cx, cy,
    dt=1.0,
    mask_invalid_depth=True,
    min_valid_z=1e-4,
):
    """
    Args:
        data dict contains:
            rgb         : (T, N, H, W, 3)
            depth       : (T, N, H, W)   # Isaac Gym (negative)
            mask        : (T, N, H, W)    # optional binary mask
            gripper_pos : (N, T, 3)
            gripper_rot : (N, T, 4)

    Returns:
        out dict:
            flow        : (T-1, N, 2, H, W)
            flow_mask   : (T-1, N, H, W)
    """

    T, N, H, W = data["depth"].shape

    # --------------------------------------------------------
    # 1. Compute egomotion flow (pure geometry)
    # --------------------------------------------------------
    flows = compute_all_egomotion_flows(
        data,
        fx=fx, fy=fy, cx=cx, cy=cy,
        dt=dt
    )  # (T-1, N, 2, H, W)

    # --------------------------------------------------------
    # 2. Build flow validity mask
    # --------------------------------------------------------
    flow_mask = np.ones((T - 1, N, H, W), dtype=bool)

    # (a) mask out invalid / missing depth
    if mask_invalid_depth:
        # true Z = -depth
        Z = -data["depth"][:-1]
        flow_mask &= Z > min_valid_z

    # (b) apply task/object mask if provided
    if "mask" in data and data["mask"] is not None:
        flow_mask &= data["mask"][:-1] > 0

    # --------------------------------------------------------
    # 3. Zero out invalid flow (safer for training)
    # --------------------------------------------------------
    flows = flows.copy()
    mag = np.linalg.norm(flows, axis=2, keepdims=True)  # (T, N, 1, H, W)

    # Normalize, keep zero where magnitude is zero
    flows = flows / (mag + 1e-8)
    flows[:, :, 0][~flow_mask] = 0.0
    flows[:, :, 1][~flow_mask] = 0.0

    return {
        "flow": flows,
        "flow_mask": flow_mask,
    }

def normalize_flow_direction(flow, eps=1e-8):
    """
    flow: (T, N, 2, H, W)

    Returns:
        flow_dir: (T, N, 2, H, W)  # unit direction
        flow_mag: (T, N, H, W)     # original magnitude (optional use)
    """

    # L2 magnitude per pixel
    mag = np.linalg.norm(flow, axis=2, keepdims=True)  # (T, N, 1, H, W)

    # Normalize, keep zero where magnitude is zero
    flow_dir = flow / (mag + eps)

    return flow_dir, mag.squeeze(2)




# ------------------------------------------------------
# Example usage
# ------------------------------------------------------
if __name__ == "__main__":
    """
    假设你在 pdb 里已经看到：
      flow.shape == (2,480,640)
      mask.shape == (1,1,240,320)
    """

    train_tasks = ["00681"]
    for task in train_tasks:
        print(f"[Eval] Processing task {task} ...")
        results=[]
        for traj_id in range(1):
            hdf_id = 100
            vis=True
            if vis:
                data = read_from_hdf5(
                    f"IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_0125_gt/asset_{task}/disassembly_traj_{hdf_id}.h5"
                )
                for keys in data.keys():
                    print(f"{keys}: {data[keys].shape}")
                out = process_dataset_with_egomotion_flow(
                    data=data,
                    fx=320, fy=320, cx=320, cy=240,
                    dt=1.0
                )
                flow = out["flow"]          # (119, 12, 2, 480, 640)
                flow_mask = out["flow_mask"]  # (119, 12, 480, 640)
                rgb = data["rgb"]
                env_id = 1
                for i in range(119):
                    visualize_flow_and_mask(
                        flow=flow[i,env_id],
                        mask=flow_mask[i,env_id],
                        rgb=rgb[i,env_id],
                        rate=2,
                        prefix=f"ego_vis/vis_task_{task}_traj_{hdf_id}_frame_{i:03d}"
                    )
                
