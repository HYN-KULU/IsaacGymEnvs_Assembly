import torch
from torch.utils.data import DataLoader
from dataset.diffusion_policy_dataset_rgb import DepthActionDataset
from policy.diffusion_policy_rgb import Diffusion_Policy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# -------------------
# Utility: unnormalize delta pos
# -------------------
def unnormalize_actions(actions, pos_min, pos_max, scale_to_unit=True):
    """
    Undo the normalization from DepthActionDataset.
    
    Args:
        actions: (B, K, 9) torch.Tensor, with normalized delta_pos and raw rot6d
        pos_min: np.ndarray or torch.Tensor, shape (3,)
        pos_max: np.ndarray or torch.Tensor, shape (3,)
        scale_to_unit: bool, whether dataset was scaled to [-1, 1] or just [0, 1]
    Returns:
        unnorm_actions: (B, K, 9) torch.Tensor, with real delta_pos + rot6d
    """
    # device = actions.device
    # pos_min = torch.tensor(pos_min, dtype=torch.float32, device=device)
    # pos_max = torch.tensor(pos_max, dtype=torch.float32, device=device)

    delta_norm = actions[..., :3]

    if scale_to_unit:
        # [-1,1] -> [0,1]
        delta_norm = (delta_norm + 1.0) / 2.0

    # [0,1] -> original range
    delta_pos = delta_norm * (pos_max - pos_min) + pos_min

    rot6d = actions[..., 3:]  # unchanged
    return torch.cat([delta_pos, rot6d], dim=-1)


def plot_trajectories(gt_actions, pred_actions, save_path="trajectory_3d.png"):
    """
    gt_actions: (K, D)
    pred_actions: (K, D)
    """

    # Convert to numpy
    if isinstance(gt_actions, torch.Tensor):
        gt_actions = gt_actions.cpu().numpy()
    if isinstance(pred_actions, torch.Tensor):
        pred_actions = pred_actions.cpu().numpy()

    # Take only xyz
    gt_xyz = gt_actions[0, :, :3]
    pred_xyz = pred_actions[0, :, :3]

    steps = np.arange(gt_xyz.shape[0])

    # Normalize steps to [0,1] for colormap
    norm = plt.Normalize(steps.min(), steps.max())
    cmap_gt = plt.cm.Blues  # Ground truth
    cmap_pred = plt.cm.Reds # Predicted

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    # --- Plot GT as scatter ---
    ax.scatter(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2],
               c=[cmap_gt(norm(i)) for i in steps],
               s=40, label="GT")

    # --- Plot Pred as scatter ---
    ax.scatter(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2],
               c=[cmap_pred(norm(i)) for i in steps],
               s=40, label="Pred")

    # Mark start and end points
    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], 
               c="navy", s=100, marker="o", label="GT Start")
    ax.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], 
               c="cyan", s=100, marker="^", label="GT End")

    ax.scatter(pred_xyz[0, 0], pred_xyz[0, 1], pred_xyz[0, 2], 
               c="darkred", s=100, marker="o", label="Pred Start")
    ax.scatter(pred_xyz[-1, 0], pred_xyz[-1, 1], pred_xyz[-1, 2], 
               c="orange", s=100, marker="^", label="Pred End")

    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("GT vs Predicted Trajectories")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"3D trajectory plot saved to {save_path}")

def inspect_sample(idx,pos_min,pos_max):
    sample = dataset[idx]
    rgb = torch.tensor(sample["rgb"], dtype=torch.float32).unsqueeze(0).to(device)
    proprio = torch.tensor(sample["proprioception"], dtype=torch.float32).unsqueeze(0).to(device)
    gt_actions = torch.tensor(sample["actions"], dtype=torch.float32).unsqueeze(0).to(device)
    import pdb;pdb.set_trace()
    with torch.no_grad():
        pred_actions = policy(rgb=rgb, proprioception=proprio, actions=None)
    pred_actions = unnormalize_actions(pred_actions, pos_min, pos_max)
    gt_actions = unnormalize_actions(gt_actions.cuda(), pos_min, pos_max)
    # B, K, D = pred_actions.shape
    B, K, D = gt_actions.shape
    for b in range(B):
        # print(f"\n=== Sample {b} ===")
        for k in range(min(K, 10)):  # only show first 5 steps for clarity
            gt = gt_actions[b, k].cpu().numpy()
            pred = pred_actions[b, k].cpu().numpy()

            # format with 5 decimals, no scientific notation
            gt_str = np.array2string(gt, precision=5, suppress_small=True, floatmode="fixed")
            pred_str = np.array2string(pred, precision=5, suppress_small=True, floatmode="fixed")

            print(f"Step {k:02d} | GT: {gt_str} | Pred: {pred_str}")
    rgb_np = sample["rgb"]  # shape (H, W, 3), dtype uint8 [0,255]
    save_path = "inspect_test_rgb.png"

    img = Image.fromarray(rgb_np.astype(np.uint8), mode="RGB")
    img.save(save_path)
    print(f"RGB image saved to {save_path}")
    plot_trajectories(gt_actions,pred_actions,save_path="trajectory_3d.png")
    



# -------------------
# Test Code
# -------------------
if __name__ == "__main__":
    # Load dataset
    dataset = DepthActionDataset("processed_dataset_rgb_absolute_actions_0927.h5")
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    device="cuda"
    # # Take one batch
    # batch = next(iter(dataloader))
    # rgb = batch["rgb"]               # (B, 1, H, W)
    # proprio = batch["proprioception"]    # (B, 9)
    # gt_actions = batch["actions"]        # (B, K, 9)


    ckpt_path = "logs/automate/diffusion_policy_ckpt_rgb_absolute_actions_0927/policy_epoch_1000.ckpt"  # or policy_last.ckpt
    policy = Diffusion_Policy(
        num_action=10,
        obs_feature_dim=512,
        hidden_dim=512,
        proprio_dim=9,
        action_dim=9
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    pos_min = torch.tensor(dataset.pos_min, dtype=torch.float32).cuda()
    pos_max = torch.tensor(dataset.pos_max, dtype=torch.float32).cuda()

    # import pdb;pdb.set_trace()
    # idx=0
    inspect_sample(0,pos_min,pos_max)

    

    print("\nDone.")
