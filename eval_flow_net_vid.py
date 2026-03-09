import argparse
import os
from collections import OrderedDict

import torch
import numpy as np
import cv2
import imageio

from dataset.flow_dataset import DepthActionDataset
from policy.flow_policy import FlowPolicy


# ------------------------------------------------------
# Optical flow → HSV color
# ------------------------------------------------------
def flow_to_hsv(flow):
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()

    u = flow[0]
    v = flow[1]

    mag = np.sqrt(u*u + v*v)
    ang = np.arctan2(v, u)

    H, W = u.shape
    hsv = np.zeros((H, W, 3), dtype=np.uint8)

    hsv[..., 0] = ((ang + np.pi) / (2*np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# ---------------------------------------------
# Draw sparse arrows on flow map
# ---------------------------------------------
def draw_arrows(img, flow, stride=6, scale=5):
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

            cv2.arrowedLine(out, (x, y), (x2, y2),
                            color=(255, 0, 0),
                            thickness=1,
                            tipLength=0.3)
    return out

def flow_to_direction(flow, eps=1e-6):
    """
    flow: (2, H, W)
    return: (2, H, W), unit direction field
    """
    mag = np.linalg.norm(flow, axis=0, keepdims=True)  # (1,H,W)
    return flow / (mag + eps)
# # ---------------------------------------------------
# # Render trajectory into video: GT | RGB | Pred
# # ---------------------------------------------------
# def render_traj_video(rgb_rev, flows_gt, flows_pred, mask, out_path):
#     """
#     rgb_rev:   (160, H, W, 3) uint8  – your rotated + reversed RGB frames
#     flows_gt:  list of 160 items, each (2,H,W)
#     flows_pred: list of 160 items, each (2,H,W)
#     mask: (1,1,H,W) torch or numpy
#     out_path: output mp4 path
#     """
#     # convert mask to numpy
#     # if hasattr(mask, "detach"):
#     #     mask = mask.detach().cpu().numpy()
#     # mask_2d = mask.squeeze()              # (H,W)
#     # mask_3d = np.stack([mask_2d]*2, axis=0)

#     frames = []

#     for t in range(120):
#         flow_gt = flows_gt[t] 
#         flow_pd = flows_pred[t] 
#         # flow_pd = flows_pred[t] * mask_3d
#         flow_gt_dir = flow_to_direction(flow_gt)
#         # flow_pd_dir = flow_to_direction(flow_pd)
#         flow_pd_dir = flow_pd
#         # print(np.linalg.norm(flow_pd_dir, axis=0, keepdims=True))
#         norm = np.linalg.norm(flow_pd_dir, axis=0)  # (H, W)

#         nonzero = norm > 1e-6
#         # print("Non-zero count:", nonzero.sum())
#         # print("Min norm:", norm[nonzero].min())
#         # print("Max norm:", norm[nonzero].max())
#         # print("Mean norm:", norm[nonzero].mean())
#         # ---- flow visualizations ----
#         gt_hsv = flow_to_hsv(flow_gt)
#         pd_hsv = flow_to_hsv(flow_pd)

#         # arrows
#         gt_arrow = draw_arrows(gt_hsv, flow_gt_dir)
#         pd_arrow = draw_arrows(pd_hsv, flow_pd_dir)

#         # ---- RGB ----
#         rgb = rgb_rev[t]          # already uint8 (H,W,3)

#         # ---- combine horizontally ----
#         frame = np.concatenate([gt_arrow, rgb, pd_arrow], axis=1)

#         frames.append(frame)

#     # ---- save as video ----
#     imageio.mimsave(out_path, frames, fps=20)
#     print(f"[✓] Saved video → {out_path}")

# def render_traj_video(rgb_rev, flows_gt, flows_pred, mask, out_path):
#     """
#     rgb_rev:    (160, H, W, 3) uint8  – rotated + reversed RGB frames
#     flows_gt:   list of 160 items, each (2,H,W)
#     flows_pred: unused (kept for interface compatibility)
#     mask:       unused (kept for interface compatibility)
#     out_path:   output mp4 path
#     """

#     frames = []

#     for t in range(120):
#         # ---- GT flow ----
#         flow_gt = flows_gt[t]                  # (2, H, W)
#         flow_gt_dir = flow_to_direction(flow_gt)

#         # HSV + arrows
#         gt_hsv = flow_to_hsv(flow_gt)
#         gt_arrow = draw_arrows(gt_hsv, flow_gt_dir)

#         # ---- RGB ----
#         rgb = rgb_rev[t]                       # (H, W, 3), uint8

#         # ---- combine horizontally: RGB | GT Flow ----
#         frame = np.concatenate([rgb, gt_arrow], axis=1)
#         frames.append(frame)

#     # ---- save video ----
#     imageio.mimsave(out_path, frames, fps=20)
#     print(f"[✓] Saved video → {out_path}")
def save_mask_image(mask, out_path, scale=3):
    """
    Save mask as a visualizable image.

    mask: torch.Tensor or np.ndarray, shape (H,W) or (1,H,W) or (1,1,H,W)
    out_path: path to save png
    scale: upsample factor for clarity
    """
    import numpy as np
    import cv2

    # ---- convert to numpy ----
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(mask)  # -> (H, W)

    # ---- normalize to [0,255] ----
    mask_img = (mask > 0).astype(np.uint8) * 255

    # ---- upsample for visibility ----
    if scale != 1:
        mask_img = cv2.resize(
            mask_img, None,
            fx=scale, fy=scale,
            interpolation=cv2.INTER_NEAREST
        )
    
    # ---- save ----
    cv2.imwrite(out_path, mask_img)
    print(f"[✓] Saved mask image → {out_path}")

# def render_traj_video(rgb_rev, flows_gt, flows_pred, mask, out_path):
#     """
#     rgb_rev:    (160, H, W, 3) uint8  – rotated + reversed RGB frames
#     flows_gt:   list of 160 items, each (2,H,W)
#     flows_pred: list of 160 items, each (2,H,W)
#     mask:       unused (kept for interface compatibility)
#     out_path:   output mp4 path
#     """

#     import cv2
#     import numpy as np
#     import imageio

#     frames = []

#     # =========================
#     # visualization parameters
#     # =========================
#     scale = 3            # upsample factor (clarity)
#     stride = 12          # arrow sparsity
#     arrow_scale = 4.0    # arrow length scaling

#     for t in range(len(flows_pred)):
#         # ---------------------
#         # RGB
#         # ---------------------
#         rgb = rgb_rev[t]  # (H, W, 3)
#         rgb = cv2.resize(
#             rgb, None, fx=scale, fy=scale,
#             interpolation=cv2.INTER_LINEAR
#         )

#         # =====================
#         # GT FLOW
#         # =====================
#         flow_gt = flows_gt[t]                  # (2, H, W)
#         flow_gt_dir = flow_to_direction(flow_gt)

#         flow_gt_dir = cv2.resize(
#             flow_gt_dir.transpose(1, 2, 0),
#             None, fx=scale, fy=scale,
#             interpolation=cv2.INTER_LINEAR
#         ).transpose(2, 0, 1)

#         sparse_gt = np.zeros_like(flow_gt_dir)
#         sparse_gt[:, ::stride, ::stride] = flow_gt_dir[:, ::stride, ::stride]
#         sparse_gt *= arrow_scale

#         gt_hsv = flow_to_hsv(flow_gt)
#         gt_hsv = cv2.resize(
#             gt_hsv, None, fx=scale, fy=scale,
#             interpolation=cv2.INTER_LINEAR
#         )

#         gt_arrow = draw_arrows(gt_hsv, sparse_gt)

#         # =====================
#         # PREDICTED FLOW
#         # =====================
#         flow_pd = flows_pred[t]                # (2, H, W)
#         flow_pd_dir = flow_to_direction(flow_pd)

#         flow_pd_dir = cv2.resize(
#             flow_pd_dir.transpose(1, 2, 0),
#             None, fx=scale, fy=scale,
#             interpolation=cv2.INTER_LINEAR
#         ).transpose(2, 0, 1)

#         sparse_pd = np.zeros_like(flow_pd_dir)
#         sparse_pd[:, ::stride, ::stride] = flow_pd_dir[:, ::stride, ::stride]
#         sparse_pd *= arrow_scale

#         pd_hsv = flow_to_hsv(flow_pd)
#         pd_hsv = cv2.resize(
#             pd_hsv, None, fx=scale, fy=scale,
#             interpolation=cv2.INTER_LINEAR
#         )

#         pd_arrow = draw_arrows(pd_hsv, sparse_pd)

#         # =====================
#         # COMBINE: GT | RGB | PRED
#         # =====================
#         frame = np.concatenate([rgb, pd_arrow], axis=1)
#         # frame = np.concatenate([gt_arrow, rgb, pd_arrow], axis=1)
#         frames.append(frame)

#     # ---- save video (high quality) ----
#     imageio.mimsave(
#         out_path,
#         frames,
#         fps=20,
#         codec="libx264",
#         bitrate="20M"
#     )
#     print(f"[✓] Saved video → {out_path}")
def render_traj_video(rgb_rev, flows_gt, flows_pred, mask, out_path):
    """
    rgb_rev:    (T, H, W, 3) uint8
    flows_gt:   unused
    flows_pred: list of (2, H, W)
    mask:       (1, H, W) or (H, W)
    out_path:   output mp4 path
    """

    import cv2
    import numpy as np
    import imageio

    frames = []

    # =========================
    # visualization parameters
    # =========================
    scale = 3
    stride = 12
    arrow_scale = 4.0

    # -------------------------
    # prepare mask
    # -------------------------
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    # mask_2d = mask[0,0]       # (H, W)
    H, W = mask.shape

    for t in range(len(flows_pred)):
        # =====================
        # LEFT: RGB
        # =====================
        rgb = rgb_rev[t]
        rgb = cv2.resize(
            rgb, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_LINEAR
        )

        # =====================
        # RIGHT: mask background
        # =====================
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        bg[mask > 0] = 255   # white inside mask

        bg = cv2.resize(
            bg, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_NEAREST
        )

        # =====================
        # FLOW (PRED) — use YOUR template
        # =====================
        flow = flows_pred[t]                    # (2, H, W)
        flow_dir = flow_to_direction(flow)

        flow_dir = cv2.resize(
            flow_dir.transpose(1, 2, 0),
            None, fx=scale, fy=scale,
            interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)

        sparse = np.zeros_like(flow_dir)
        sparse[:, ::stride, ::stride] = flow_dir[:, ::stride, ::stride]
        sparse *= arrow_scale

        flow_arrow = draw_arrows(bg, sparse)

        # =====================
        # COMBINE
        # =====================
        frame = np.concatenate([rgb, flow_arrow], axis=1)
        frames.append(frame)

    # ---- save video ----
    imageio.mimsave(
        out_path,
        frames,
        fps=20,
        codec="libx264",
        bitrate="20M"
    )
    print(f"[✓] Saved video → {out_path}")


# def render_traj_video(rgb_rev, flows_gt, flows_pred, mask, out_path):
#     """
#     rgb_rev:    (160, H, W, 3) uint8  – rotated + reversed RGB frames
#     flows_gt:   list of 160 items, each (2,H,W)
#     flows_pred: unused (kept for interface compatibility)
#     mask:       unused (kept for interface compatibility)
#     out_path:   output mp4 path
#     """

#     import cv2
#     import numpy as np
#     import imageio

#     frames = []

#     # =========================
#     # visualization parameters
#     # =========================
#     scale = 3            # upsample factor (clarity)
#     stride = 12          # arrow sparsity (IMPORTANT)
#     arrow_scale = 4.0    # arrow length scaling (IMPORTANT)

#     for t in range(120):
#         # ---- RGB ----
#         rgb = rgb_rev[t]  # (H, W, 3)
#         rgb = cv2.resize(
#             rgb, None, fx=scale, fy=scale,
#             interpolation=cv2.INTER_LINEAR
#         )

#         # ---- GT flow ----
#         flow_gt = flows_gt[t]                  # (2, H, W)
#         flow_dir = flow_to_direction(flow_gt)  # (2, H, W)

#         # upsample flow direction
#         flow_dir = cv2.resize(
#             flow_dir.transpose(1, 2, 0),
#             None, fx=scale, fy=scale,
#             interpolation=cv2.INTER_LINEAR
#         ).transpose(2, 0, 1)

#         # ---- sparsify arrows ----
#         sparse_flow = np.zeros_like(flow_dir)
#         sparse_flow[:, ::stride, ::stride] = flow_dir[:, ::stride, ::stride]

#         # ---- make arrows longer ----
#         sparse_flow = sparse_flow * arrow_scale

#         # ---- HSV visualization ----
#         gt_hsv = flow_to_hsv(flow_gt)
#         gt_hsv = cv2.resize(
#             gt_hsv, None, fx=scale, fy=scale,
#             interpolation=cv2.INTER_LINEAR
#         )

#         # ---- draw arrows ----
#         gt_arrow = draw_arrows(gt_hsv, sparse_flow)

#         # ---- combine horizontally: RGB | GT Flow ----
#         frame = np.concatenate([rgb, gt_arrow], axis=1)
#         frames.append(frame)
    
#     # ---- save video (high quality) ----
#     imageio.mimsave(
#         out_path,
#         frames,
#         fps=20,
#         codec="libx264",
#         bitrate="20M"
#     )
#     print(f"[✓] Saved video → {out_path}")



def rewrite_monai_keys(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        nk = nk.replace(".sub0", ".submodule.0")
        nk = nk.replace(".sub1", ".submodule.1")
        nk = nk.replace(".sub2", ".submodule.2")
        nk = nk.replace(".subconv", ".submodule.conv")
        nk = nk.replace(".subadn", ".submodule.adn")
        new_sd[nk] = v
    return new_sd


def apply_mask_to_flow(flow, mask):
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    mask_2d = mask.squeeze()         # (H,W)
    mask_3d = np.stack([mask_2d, mask_2d], axis=0)

    return flow * mask_3d

import h5py
def read_from_hdf5(filename):
    with h5py.File(filename, "r") as f:
        return {"rgb":f["camera3_rgb"][()],"mask": f["mask"][()],"depth":f["camera3_depth"][()]}     # (T, 12, 480, 640, 3)

def read_from_hdf5_flow(filename):
    with h5py.File(filename, "r") as f:
        return {"flow":f["flow"][()]}     # (T, 12, 480, 640, 3)

def evaluate_directional_success_v2(
    score_gt_x_list,
    score_gt_y_list,
    score_pred_x_list,
    score_pred_y_list,
    threshold=0.1
):
    """
    Improved directional correctness metric.

    For success_x: only count frames where |GT_x| >= threshold.
    For success_y: only count frames where |GT_y| >= threshold.
    For success_xy: count frames where at least one GT direction is meaningful,
                    and the prediction is correct on BOTH axes.
    """

    import numpy as np

    # Convert to numpy
    gt_x = np.array(score_gt_x_list)[:40]
    gt_y = np.array(score_gt_y_list)[:40]
    pred_x = np.array(score_pred_x_list)[:40]
    pred_y = np.array(score_pred_y_list)[:40]

    # ---------- Convert flows into {-1,0,+1} direction ----------
    def to_dir(arr):
        d = np.zeros_like(arr)
        d[arr >  threshold] = 1
        d[arr < -threshold] = -1
        return d

    gt_dir_x = to_dir(gt_x)
    gt_dir_y = to_dir(gt_y)
    pred_dir_x = to_dir(pred_x)
    pred_dir_y = to_dir(pred_y)
    # ---------- Masks for meaningful frames ----------
    mask_x = np.abs(gt_x) >= threshold          # meaningful X motion
    mask_y = np.abs(gt_y) >= threshold          # meaningful Y motion
    mask_xy = mask_x | mask_y                   # at least one meaningful motion

    # ---------- Compute correctness ----------
    correct_x = (gt_dir_x == pred_dir_x)
    correct_y = (gt_dir_y == pred_dir_y)
    correct_xy = correct_x & correct_y          # both must be correct
    # ---------- Success Rates ----------
    success_x = correct_x[mask_x].mean() if mask_x.any() else 0.0
    success_y = correct_y[mask_y].mean() if mask_y.any() else 0.0
    success_xy = correct_xy[mask_xy].mean() if mask_xy.any() else 0.0

    return {
        "success_x": float(success_x),
        "success_y": float(success_y),
        "success_xy": float(success_xy),
        "num_x_frames": int(mask_x.sum()),
        "num_y_frames": int(mask_y.sum()),
        "num_xy_frames": int(mask_xy.sum()),
        "total_frames": len(gt_x),
    }

def evaluate_directional_success_v3(
    score_gt_x_list,
    score_gt_y_list,
    score_pred_x_list,
    score_pred_y_list,
    mag_threshold=1,
    angle_threshold_deg=20.0,
    max_frames=40,
):
    """
    Directional success metric based on angular error.

    - GT and prediction are treated as 2D vectors (x, y)
    - Both are normalized to represent directions
    - Angular error (degrees) is computed
    - Success = angle_error <= angle_threshold_deg
    - Frames with small GT magnitude are ignored

    Returns:
        success_rate, mean_angle_error, stats
    """

    import numpy as np

    # -----------------------------
    # Convert to numpy and truncate
    # -----------------------------
    gt_x = np.array(score_gt_x_list)[:max_frames]
    gt_y = np.array(score_gt_y_list)[:max_frames]
    pr_x = np.array(score_pred_x_list)[:max_frames]
    pr_y = np.array(score_pred_y_list)[:max_frames]

    gt_vec = np.stack([gt_x, gt_y], axis=1)   # (T, 2)
    pr_vec = np.stack([pr_x, pr_y], axis=1)   # (T, 2)

    # -----------------------------
    # GT magnitude mask
    # -----------------------------
    gt_mag = np.linalg.norm(gt_vec, axis=1)
    valid_mask = gt_mag >= mag_threshold

    if not valid_mask.any():
        return {
            "success_rate": 0.0,
            "mean_angle_error_deg": None,
            "num_valid_frames": 0,
            "total_frames": len(gt_vec),
        }

    gt_vec = gt_vec[valid_mask]
    pr_vec = pr_vec[valid_mask]

    # -----------------------------
    # Normalize vectors
    # -----------------------------
    def normalize(v, eps=1e-8):
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + eps)

    gt_dir = normalize(gt_vec)
    pr_dir = normalize(pr_vec)

    # -----------------------------
    # Angle error (degrees)
    # -----------------------------
    dot = np.sum(gt_dir * pr_dir, axis=1)
    dot = np.clip(dot, -1.0, 1.0)  # numerical safety

    angle_error_deg = np.degrees(np.arccos(dot))  # (N,)

    # -----------------------------
    # Success criterion
    # -----------------------------
    success = angle_error_deg <= angle_threshold_deg
    # import pdb;pdb.set_trace()
    return {
        "success_rate": float(success.mean()),
        "mean_angle_error_deg": float(angle_error_deg.mean()),
        "median_angle_error_deg": float(np.median(angle_error_deg)),
        "num_success_frames": int(success.sum()),
        "num_valid_frames": int(len(angle_error_deg)),
        "total_frames": int(len(gt_x)),
        "angle_threshold_deg": float(angle_threshold_deg),
    }


def evaluate_directional_success(
    score_gt_x_list,
    score_gt_y_list,
    score_pred_x_list,
    score_pred_y_list,
    threshold=0.1
):
    """
    Evaluate directional correctness for socket flow prediction.

    Args:
        score_gt_x_list: list of floats (GT avg flow_x)
        score_gt_y_list: list of floats (GT avg flow_y)
        score_pred_x_list: list of floats (Pred avg flow_x)
        score_pred_y_list: list of floats (Pred avg flow_y)
        threshold: float, |flow| above this → considered significant

    Returns:
        dict with success rates for x, y, xy
    """

    T = len(score_gt_x_list)

    # Convert lists to numpy
    gt_x = np.array(score_gt_x_list)
    gt_y = np.array(score_gt_y_list)
    pred_x = np.array(score_pred_x_list)
    pred_y = np.array(score_pred_y_list)

    # ----------- Determine GT direction class --------------
    # +1 = positive direction
    # -1 = negative direction
    #  0 = insignificant
    def to_dir(arr):
        d = np.zeros_like(arr)
        d[arr > threshold] = 1
        d[arr < -threshold] = -1
        return d

    gt_dir_x = to_dir(gt_x)
    gt_dir_y = to_dir(gt_y)
    pred_dir_x = to_dir(pred_x)
    pred_dir_y = to_dir(pred_y)

    # ----------- Compute correctness --------------
    correct_x = (gt_dir_x == pred_dir_x)
    correct_y = (gt_dir_y == pred_dir_y)

    # XY must BOTH be correct
    correct_xy = correct_x & correct_y

    success_x = correct_x.mean()
    success_y = correct_y.mean()
    success_xy = correct_xy.mean()

    return {
        "success_x": float(success_x),
        "success_y": float(success_y),
        "success_xy": float(success_xy),
        "frames": T,
    }

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
# MAIN: Predict entire trajectory & save video
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=int, default=0)     # which trajectory id
    parser.add_argument("--ckpt", type=str, default="logs/automate/flow_net_ckpt/epoch_low_resolution.pt")
    parser.add_argument("--out", type=str, default="traj_flow_pred")
    parser.add_argument("--task_id", type=str, default="")
    args = parser.parse_args()

    device = "cuda"

    # ============================================================
    # 1. Load Flow Model
    # ============================================================
    policy = FlowPolicy().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)

    sd = ckpt["model"]
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    sd = rewrite_monai_keys(sd)
    policy.load_state_dict(sd, strict=True)
    policy.eval()

    # print("[Eval] Loaded model from:", args.ckpt)

    # ============================================================
    # 2. Load RGB trajectory
    # ============================================================
    # train_tasks=['00004', '00016', '00021', '00030', '00042', '00074', '00077', '00078', '00110', '00117', '00133', '00138', '00141', '00163', '00175', '00186', '00187', '00192', '00211', '00213', '00255', '00256', '00271', '00293', '00296', '00301', '00318', '00319', '00320', '00329', '00345', '00346', '00388', '00410', '00417', '00422', '00426', '00437', '00444', '00446', '00471', '00480', '00499', '00514', '00537', '00559', '00581', '00614', '00615', '00638', '00649', '00659', '00681', '00686', '00700', '00703', '00726', '00768', '00783', '00855', '00860', '01026', '01029', '01036', '01041', '01079', '01092', '01102', '01129', '01132', '01136']
    log_result={}
    train_tasks = ["00028"]
    for task in train_tasks:
        print(f"[Eval] Processing task {task} ...")
        args.task_id = task
        results=[]
        for traj_id in range(1):
            # traj_id = 2
            hdf_id = 2
            traj_id = 2
            vis=True
            if vis:
                # data = read_from_hdf5(
                    # f"/tmp/flow_1206/asset_{args.task_id}/disassembly_traj_{hdf_id}.h5"
                # )
                data = read_from_hdf5(
                     f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_1206/asset_{args.task_id}/disassembly_traj_{hdf_id}.h5"
                )
                # data: (160, 12, 480, 640, 3)
                # processing mask
                mask = data["mask"][traj_id, ...]                  # (160,480,640)
                save_mask_image(np.rot90(mask, 2), "saved_mask.png")
                print("Save mask")
                rgbs = data["rgb"][:, traj_id, ...]                      # (160,480,640,3)
                rgb_rot = np.rot90(rgbs, 2, axes=(1,2)).copy()    # rotate 180°
                rgb_rev = rgb_rot[::-1].copy()                    # reverse in time
                depths=data["depth"]
                depths_rot=np.rot90(depths,2,axes=(2,3))
                depth_rot_rev=torch.from_numpy(depths_rot[::-1].copy()).cuda()
                depth_rot_rev = torch.clamp(depth_rot_rev, min=-0.5, max=0.0)
                depth_rot_rev = (depth_rot_rev - (-0.1890)) / 0.0795
            # rgb_rev shape: (160,480,640,3)

            # ============================================================
            # 3. Load dataset (flows + masks + depths)
            # ============================================================
            # dataset = DepthActionDataset(task_id=args.task_id)

            # ============================================================
            # 4. Predict all flows
            # ============================================================
            flows_gt_list = []       # store GT flows
            flows_pred_list = []     # store predicted flows
            masks = None
            # offset=0
            # start_idx = traj_id * 160 + offset
            # end_idx   = traj_id * 160 + 160 + offset    # not inclusive; 160 frames

            # print(f"[Eval] Processing trajectory {traj_id}, frames {start_idx} → {end_idx-1}")
            score_gt_x_list=[]
            score_gt_y_list=[]
            score_pred_x_list=[]
            score_pred_y_list=[]
            flow_data=read_from_hdf5_flow(f"/tmp/data/flow_mask_data_1206/{args.task_id}/flow_asset_{args.task_id}_{hdf_id}.h5")
            # flow_data=read_from_hdf5_flow(f"/home/ubuntu/automate/alltracker/flow_mask_data_1206/{args.task_id}/flow_asset_{args.task_id}_{hdf_id}.h5")
            rate = 2
            masks = torch.from_numpy(
                np.rot90(data["mask"], 2, axes=(1, 2)).copy()  # rotate 180° on H,W
            ).cuda().unsqueeze(0).unsqueeze(0)
            mask=masks[:,:,traj_id,:,:][:,:,::rate,::rate]
            # for t, idx in enumerate(range(start_idx, end_idx)):
            mask_np = mask[0].detach().cpu().numpy()
            mask_2d = mask_np[0]
            H,W = mask_2d.shape
            ys, xs = np.where(mask_2d > 0)
            cy = ys.mean()
            cx = xs.mean()
            target_y = H / 2.0
            target_x = W / 2.0
            dy = target_y - cy
            dx = target_x - cx
            mask_shifted_np = shift_np_integer(mask_np, dx, dy)
            mask = torch.from_numpy(mask_shifted_np).unsqueeze(0).to(device)
            # rendered_mask=np.rot90(data["mask"][traj_id],2).copy()[::rate,::rate]
            rendered_mask = np.rot90(mask_shifted_np[0],2).copy()

            for t in range(100):
                # sample = dataset[idx]
                # import pdb;pdb.set_trace()
                # mask = sample["mask"].unsqueeze(0).to(device)      # (1,1,H,W)

                depth = depth_rot_rev[t,traj_id].unsqueeze(0).unsqueeze(0).to(device)[:,:,::rate,::rate]    # (1,1,H,W)
                # depth normalize
                flow_gt = flow_data["flow"][traj_id,t][:,::rate, ::rate]

                if masks is None:
                    masks = mask      # store mask for entire traj
                
                # model prediction
                with torch.no_grad():
                    pred_flow = policy(mask, depth)[0].cpu().numpy()
                # Mask flows
                flow_gt_masked = apply_mask_to_flow(flow_gt, rendered_mask)
                pred_flow_masked = apply_mask_to_flow(pred_flow, rendered_mask)
                # pred_flow_masked = pred_flow
                gt_score_x=flow_gt_masked[0].sum()/mask.sum()
                gt_score_y=flow_gt_masked[1].sum()/mask.sum()
                pred_score_x=pred_flow_masked[0].sum()/mask.sum()
                pred_score_y=pred_flow_masked[1].sum()/mask.sum()
                score_gt_x_list.append(gt_score_x.detach().cpu().numpy())
                score_gt_y_list.append(gt_score_y.detach().cpu().numpy())
                score_pred_x_list.append(pred_score_x.detach().cpu().numpy())
                score_pred_y_list.append(pred_score_y.detach().cpu().numpy())
                # store
                flows_gt_list.append(flow_gt_masked)
                flows_pred_list.append(pred_flow_masked)

                # print(f"Frame {t:03d} / 119 processed")
            # result=evaluate_directional_success_v3(
            #     score_gt_x_list,
            #     score_gt_y_list,
            #     score_pred_x_list,
            #     score_pred_y_list,
            # )
            # import pdb;pdb;pdb.set_trace()
            # ============================================================
            # 5. Render video (GT | RGB | Pred)
            # ============================================================
            vis = True
            if vis:
                os.makedirs("eval_vis", exist_ok=True)
                out_path = f"eval_vis/{args.out}_{traj_id}.mp4"

                print("[Eval] Rendering video ...")
                rate = 2
                render_traj_video(
                    rgb_rev=rgb_rev[:100, ::rate, ::rate],
                    flows_gt=flows_gt_list,
                    flows_pred=flows_pred_list,
                    mask=rendered_mask,
                    # mask=mask,
                    out_path=out_path
                )

                print(f"[✓] Saved final trajectory video to: {out_path}")
            # print(result)
            # results.append(result)
        log_result[task] = results
        import json 
        with open(f"eval_vis/results_update_metric.json", "w") as f:
            json.dump(log_result, f)

if __name__ == "__main__":
    main()
