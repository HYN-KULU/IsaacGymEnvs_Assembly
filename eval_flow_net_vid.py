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
def draw_arrows(img, flow, stride=20, scale=30):
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
                            color=(0, 255, 0),
                            thickness=1,
                            tipLength=0.3)
    return out


# ---------------------------------------------------
# Render trajectory into video: GT | RGB | Pred
# ---------------------------------------------------
def render_traj_video(rgb_rev, flows_gt, flows_pred, mask, out_path):
    """
    rgb_rev:   (160, H, W, 3) uint8  – your rotated + reversed RGB frames
    flows_gt:  list of 160 items, each (2,H,W)
    flows_pred: list of 160 items, each (2,H,W)
    mask: (1,1,H,W) torch or numpy
    out_path: output mp4 path
    """
    # convert mask to numpy
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask_2d = mask.squeeze()              # (H,W)
    mask_3d = np.stack([mask_2d]*2, axis=0)

    frames = []

    for t in range(160):
        flow_gt = flows_gt[t] * mask_3d
        flow_pd = flows_pred[t] 
        # flow_pd = flows_pred[t] * mask_3d

        # ---- flow visualizations ----
        gt_hsv = flow_to_hsv(flow_gt)
        pd_hsv = flow_to_hsv(flow_pd)

        # arrows
        gt_arrow = draw_arrows(gt_hsv, flow_gt)
        pd_arrow = draw_arrows(pd_hsv, flow_pd)

        # ---- RGB ----
        rgb = rgb_rev[t]          # already uint8 (H,W,3)

        # ---- combine horizontally ----
        frame = np.concatenate([gt_arrow, rgb, pd_arrow], axis=1)

        frames.append(frame)

    # ---- save as video ----
    imageio.mimsave(out_path, frames, fps=20)
    print(f"[✓] Saved video → {out_path}")


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
        return f["camera3_rgb"][()]     # (T, 12, 480, 640, 3)

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

# ------------------------------------------------------
# MAIN: Predict entire trajectory & save video
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=int, default=0)     # which trajectory id
    parser.add_argument("--ckpt", type=str, default="logs/automate/flow_net_ckpt/epoch_low_resolution.pt")
    parser.add_argument("--out", type=str, default="traj_flow_pred.mp4")
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

    print("[Eval] Loaded model from:", args.ckpt)

    # ============================================================
    # 2. Load RGB trajectory
    # ============================================================
    traj_id = args.traj
    hdf_id = 51
    vis=False
    if vis:
        data = read_from_hdf5(
            f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow/asset_00681/disassembly_traj_{hdf_id}.h5"
        )
        # data: (160, 12, 480, 640, 3)

        rgbs = data[:, traj_id, ...]                      # (160,480,640,3)
        rgb_rot = np.rot90(rgbs, 2, axes=(1,2)).copy()    # rotate 180°
        rgb_rev = rgb_rot[::-1].copy()                    # reverse in time
    # rgb_rev shape: (160,480,640,3)

    # ============================================================
    # 3. Load dataset (flows + masks + depths)
    # ============================================================
    dataset = DepthActionDataset()

    # ============================================================
    # 4. Predict all flows
    # ============================================================
    flows_gt_list = []       # store GT flows
    flows_pred_list = []     # store predicted flows
    masks = None
    offset=1920
    start_idx = traj_id * 160 + offset
    end_idx   = traj_id * 160 + 160 + offset    # not inclusive; 160 frames

    print(f"[Eval] Processing trajectory {traj_id}, frames {start_idx} → {end_idx-1}")
    score_gt_x_list=[]
    score_gt_y_list=[]
    score_pred_x_list=[]
    score_pred_y_list=[]
    for t, idx in enumerate(range(start_idx, end_idx)):
        sample = dataset[idx]

        mask = sample["mask"].unsqueeze(0).to(device)      # (1,1,H,W)
        depth = sample["depth"].unsqueeze(0).to(device)    # (1,1,H,W)
        flow_gt = sample["flow"]                           # (2,H,W) numpy

        if masks is None:
            masks = mask      # store mask for entire traj

        # model prediction
        with torch.no_grad():
            pred_flow = policy(mask, depth)[0].cpu().numpy()

        # Mask flows
        flow_gt_masked = apply_mask_to_flow(flow_gt, mask)
        pred_flow_masked = apply_mask_to_flow(pred_flow, mask)
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

        print(f"Frame {t:03d} / 159 processed")
    result=evaluate_directional_success(
        score_gt_x_list,
        score_gt_y_list,
        score_pred_x_list,
        score_pred_y_list,
        threshold=0.2
    )
    import pdb;pdb;pdb.set_trace()
    # ============================================================
    # 5. Render video (GT | RGB | Pred)
    # ============================================================
    if vis:
        os.makedirs("eval_vis", exist_ok=True)
        out_path = f"eval_vis/{args.out}"

        print("[Eval] Rendering video ...")
        render_traj_video(
            rgb_rev=rgb_rev,
            flows_gt=flows_gt_list,
            flows_pred=flows_pred_list,
            mask=masks,
            out_path=out_path
        )

        print(f"[✓] Saved final trajectory video to: {out_path}")



if __name__ == "__main__":
    main()
