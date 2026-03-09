import torch
from torch.utils.data import DataLoader
from dataset.diffusion_policy_dataset import DepthActionDataset
from policy.diffusion_policy import Diffusion_Policy
import numpy as np
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



# -------------------
# Test Code
# -------------------
if __name__ == "__main__":
    # Load dataset
    dataset = DepthActionDataset("processed_dataset.h5")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    device="cuda"
    # Take one batch
    batch = next(iter(dataloader))
    depth = batch["depth"]               # (B, 1, H, W)
    proprio = batch["proprioception"]    # (B, 9)
    gt_actions = batch["actions"]        # (B, K, 9)

    ckpt_path = "logs/automate/diffusion_policy_ckpt/policy_epoch_1000.ckpt"  # or policy_last.ckpt
    policy = Diffusion_Policy(
        num_action=20,
        obs_feature_dim=512,
        hidden_dim=512,
        proprio_dim=9,
        action_dim=9
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    # Predict actions
    with torch.no_grad():
        pred_actions = policy(depth=depth.cuda(), proprioception=proprio.cuda(), actions=None).cuda()
    # import pdb;pdb.set_trace()
    # Unnormalize if dataset provides min/max
    if hasattr(dataset, "pos_min") and hasattr(dataset, "pos_max"):
        pos_min = torch.tensor(dataset.pos_min, dtype=torch.float32).cuda()
        pos_max = torch.tensor(dataset.pos_max, dtype=torch.float32).cuda()
        pred_actions = unnormalize_actions(pred_actions, pos_min, pos_max)
        gt_actions = unnormalize_actions(gt_actions.cuda(), pos_min, pos_max)

    # -------------------
    # Print results
    # -------------------
    B, K, D = pred_actions.shape
    for b in range(B):
        print(f"\n=== Sample {b} ===")
        for k in range(min(K, 5)):  # only show first 5 steps for clarity
            gt = gt_actions[b, k].cpu().numpy()
            pred = pred_actions[b, k].cpu().numpy()

            # format with 5 decimals, no scientific notation
            gt_str = np.array2string(gt, precision=5, suppress_small=True, floatmode="fixed")
            pred_str = np.array2string(pred, precision=5, suppress_small=True, floatmode="fixed")

            print(f"Step {k:02d} | GT: {gt_str} | Pred: {pred_str}")

    print("\nDone.")
    import pdb;pdb.set_trace()
