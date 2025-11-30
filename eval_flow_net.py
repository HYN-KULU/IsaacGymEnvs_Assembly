import argparse
import os
from collections import OrderedDict

import torch
import numpy as np
import cv2

from dataset.flow_dataset import DepthActionDataset
from policy.flow_policy import FlowPolicy


# ------------------------------------------------------
# Load checkpoint (mimic InsertionNet ckpt loading)
# ------------------------------------------------------
def load_flow_policy_ckpt(model, ckpt_path, device="cuda"):
    print(f"[Eval] Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # Most training scripts save {"model": state_dict}
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt  # fallback

    # Remove "module." (DDP prefix)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)
    print("[Eval] Checkpoint loaded successfully.")


# ------------------------------------------------------
# Optical flow → Color for visualization
# ------------------------------------------------------
def flow_to_color(flow):
    """
    flow: (2, H, W)
    returns: (H, W, 3) uint8
    """

    flow = flow.copy()
    u = flow[0]
    v = flow[1]

    magnitude, angle = cv2.cartToPolar(u, v)

    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
    hsv[..., 1] = 255                     # Saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def rewrite_monai_keys(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_k = k

        # generic submodule index replacement
        new_k = new_k.replace(".sub0", ".submodule.0")
        new_k = new_k.replace(".sub1", ".submodule.1")
        new_k = new_k.replace(".sub2", ".submodule.2")
        new_k = new_k.replace(".sub3", ".submodule.3")

        # *** FIX THIS PART ***
        # old MONAI style: subconv → new style: submodule.conv
        new_k = new_k.replace(".subconv", ".submodule.conv")

        # old MONAI style: subadn → new style: submodule.adn
        new_k = new_k.replace(".subadn", ".submodule.adn")

        new_sd[new_k] = v

    return new_sd
import cv2


def flow_to_hsv(flow):
    """
    flow: (2, H, W) numpy array or torch tensor
    returns: (H, W, 3) RGB uint8 image
    """
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()

    u = flow[0]
    v = flow[1]

    mag = np.sqrt(u*u + v*v)
    ang = np.arctan2(v, u)  # [-pi, pi]

    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)

    # Hue: angle mapped to [0,180]
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)

    # Saturation: full
    hsv[..., 1] = 255

    # Value: magnitude normalized
    hsv[..., 2] = np.clip(mag / (mag.max() + 1e-8) * 255, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def draw_arrows(img, flow, stride=20, color=(0,255,0), scale=1.0):
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()

    u = flow[0]
    v = flow[1]

    H, W = u.shape
    out = img.copy()

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            dx = u[y, x]
            dy = v[y, x]
            x2 = int(x + dx * scale)
            y2 = int(y + dy * scale)

            cv2.arrowedLine(
                out,
                (x, y), (x2, y2),
                color,
                1,
                tipLength=0.3
            )

    return out


def visualize_gt_pred_flow_with_arrows(flow_gt, flow_pred, out_path,
                                       stride=20, scale=1.0):
    """
    flow_gt, flow_pred: (2,H,W) torch or numpy
    Saves one image:
    [ GT (HSV+arrows) | PRED (HSV+arrows) ]
    """

    # Convert flows → HSV color
    gt_rgb = flow_to_hsv(flow_gt)      # (H,W,3)
    pred_rgb = flow_to_hsv(flow_pred)

    # Draw arrows
    gt_arrow = draw_arrows(gt_rgb, flow_gt, stride=stride, scale=scale)
    pred_arrow = draw_arrows(pred_rgb, flow_pred, stride=stride, scale=scale)

    # Concatenate horizontally
    combined = np.concatenate([gt_arrow, pred_arrow], axis=1)

    # Save
    cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"[✓] Saved → {out_path}")
def apply_mask_to_flow(flow, mask):
    """
    flow: (2, H, W) numpy array
    mask: torch tensor (1,1,H,W) or numpy (1,1,H,W), bool

    returns: (2, H, W) numpy array
    """
    if hasattr(flow, "detach"):   # if user passes torch
        flow = flow.detach().cpu().numpy()

    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    # mask: (1,1,H,W) → (H,W)
    mask_2d = mask.squeeze()

    # expand mask to (2,H,W)
    mask_3d = np.stack([mask_2d, mask_2d], axis=0)

    flow_masked = flow * mask_3d

    return flow_masked
# ------------------------------------------------------
# Main eval
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------
    # Build dataset & load one sample
    # ------------------------------
    dataset = DepthActionDataset()
    sample = dataset[args.idx]

    mask = sample["mask"].unsqueeze(0).to(device)     # (1,1,H,W)
    depth = sample["depth"].unsqueeze(0).to(device)   # (1,1,H,W)
    flow_gt = sample["flow"].numpy()                  # (2,H,W)

    print("Sample loaded.")
    print("Mask:", mask.shape)
    print("Depth:", depth.shape)
    print("Flow GT:", flow_gt.shape)

    # ------------------------------
    # Build model
    # ------------------------------
    policy = FlowPolicy().to(device)
    policy.eval()
    ckpt = torch.load("logs/automate/flow_net_ckpt/epoch_low_resolution.pt", map_location="cuda")

    state_dict = ckpt["model"]

    # remove DDP prefix
    cleaned = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())

    # repair MONAI UNet key names
    cleaned = rewrite_monai_keys(cleaned)

    policy.load_state_dict(cleaned, strict=True)
    # Load checkpoint

    # ------------------------------
    # Forward
    # ------------------------------
    with torch.no_grad():
        pred_flow = policy(mask, depth)[0].cpu().numpy()   # (2,H,W)

    print("Predicted flow:", pred_flow.shape)

    # ------------------------------
    # Save visualization
    # ------------------------------
    os.makedirs("eval_vis", exist_ok=True)
    flow_gt_masked  = apply_mask_to_flow(flow_gt,  mask)
    # flow_pred_masked = apply_mask_to_flow(pred_flow, mask)
    visualize_gt_pred_flow_with_arrows(
        flow_gt=flow_gt_masked,         # (2,H,W)
        flow_pred=pred_flow,     # (2,H,W)
        # flow_pred=flow_pred_masked,     # (2,H,W)
        out_path=f"eval_vis/flow_compare_{args.idx}.png",
        stride=15,
        scale=55.0
    )

    print(f"[Eval] Saved visualization to: eval_vis/gt_flow_{args.idx}.png")
    print(f"[Eval] Saved visualization to: eval_vis/pred_flow_{args.idx}.png")


if __name__ == "__main__":
    main()
