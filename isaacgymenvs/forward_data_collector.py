import isaacgym
import isaacgymenvs
import hydra
from omegaconf import DictConfig
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.tasks import isaacgym_task_map
import os
import torch
# from policy.diffusion_policy import Diffusion_Policy
from dataset.diffusion_policy_dataset import DepthActionDataset
from policy.diffusion_policy import Diffusion_Policy
from diffusion_utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform
import cv2
import random
def load_processed_dataset(filename):
    import h5py
    data = {}
    with h5py.File(filename, "r") as f:
        # data["depth"] = f["depth"][()]               # (N, H, W)
        data["proprioception"] = f["proprioception"][()]  # (N, 9)
        data["actions"] = f["actions"][()]           # (N, K, 9)
    return data
def unnormalize_actions(actions, pos_min, pos_max, rot_min,rot_max, scale_to_unit=True):
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
    delta_norm_rot=actions[...,3:]
    if scale_to_unit:
        # [-1,1] -> [0,1]
        delta_norm = (delta_norm + 1.0) / 2.0
        delta_norm_rot=(delta_norm_rot + 1.0) / 2.0
    # [0,1] -> original range
    delta_pos = delta_norm * (pos_max - pos_min) + pos_min
    delta_rot = delta_norm_rot * (rot_max-rot_min) + rot_min
    return torch.cat([delta_pos, delta_rot], dim=-1)

import imageio
import numpy as np

def normalize_xy_min_length(delta_pos, i, min_length=0.0005):
    """
    Normalize xy direction of delta_pos while keeping magnitude >= min_length.

    Args:
        delta_pos: (N, 3) torch.Tensor
        min_length: float, minimum magnitude for xy plane

    Returns:
        delta_pos_new: (N, 3) torch.Tensor
    """
    # 提取 xy 分量
    min_length=0.001 if i < 100 else 0.0005
    delta_xy = delta_pos[:, :2]

    # 计算 xy 平面上的长度
    xy_norm = torch.norm(delta_xy, dim=1, keepdim=True) + 1e-8  # 防止除零

    # 计算需要放大的比例：如果原始长度小于 min_length，就放大
    scale = torch.where(xy_norm < min_length, min_length / xy_norm, torch.ones_like(xy_norm))

    # 应用缩放到 xy 部分
    delta_xy_scaled = delta_xy * scale

    # 拼回 z 分量
    delta_pos_new = torch.cat([delta_xy_scaled, delta_pos[:, 2:]], dim=1)
    return delta_pos_new


def save_video_from_images(image_list, out_path="output_video.mp4", fps=10):
    """
    Convert a list of numpy images (HxW or HxWx3/4) into a video.
    
    Args:
        image_list: list of np.ndarray images (uint8 or float in [0,255])
        out_path: path to save the video
        fps: frames per second
    """
    # ensure images are uint8
    frames = []
    for img in image_list:
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.rotate(img, cv2.ROTATE_180)
        frames.append(img)

    # write video
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved video: {out_path}")

@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def run_env(cfg: DictConfig):
    device="cuda"
    # === mimic train.py preamble ===
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    set_np_formatting()

    # global rank is 0 for single GPU case
    global_rank = 0
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)
    cfg.task.seed = cfg.seed

    # === mimic train.py: create env ===
    envs = isaacgymenvs.make(
        cfg.seed,
        cfg.task_name,
        cfg.task.env.numEnvs,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
    )
   

    # import pdb;pdb.set_trace()
    env_ids=torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11], device='cuda:0')
    envs.reset_idx(env_ids)
    init_plug_pos = envs.plug_pos.clone()
    envs.disassemble_plug_from_socket_eval_init()
    save_dir = "eval_visual"
    os.makedirs(save_dir, exist_ok=True)
    envs.visualize_top_camera(0, f"eval_visual/visualize_eval_top_camera.png")
    torch.set_printoptions(precision=5, sci_mode=False)
    image_list=[[] for _ in range(12)]
    target_final_pos=torch.tensor([4.3678e-04, 1.3084e-03, 4.5973e-01], device='cuda:0')
    offset = torch.zeros(12, 3)
    # -0.01 ~ 0.01 for task 00042
    offset[:, :2] = (torch.rand(12, 2) * 0.02) - 0.01 
    
    nums = random.sample(range(200), 50)
    # Test Min Height
    # 00861: 0.418
    min_height = 0
    if envs.cfg_task.env.desired_subassemblies[0] == "asset_00681":
        min_height = 0.418
    if envs.cfg_task.env.desired_subassemblies[0] == "asset_00028":
        min_height = 0.4234
    if envs.cfg_task.env.desired_subassemblies[0] == "asset_00030":
        min_height = 0.413
    action_list=[]
    depth_list=[]
    for i in range(60):
        print(f"Step {i}")
        log_step=False
        delta_pos = init_plug_pos - envs.plug_pos 
        delta_norm = torch.norm(delta_pos, dim=1, keepdim=True) + 1e-8
        target_length = 0.004 * (3 ** 0.5)
        delta_pos = delta_pos / delta_norm * target_length
        delta_pos[:,2] = -0.0004
        plug_height = envs.plug_pos[:, 2]  # shape [N]
        freeze_mask = (plug_height <= min_height).unsqueeze(1)  # shape [N,1] for broadcasting
        # freeze delta_pos to 0 where height <= min_height
        delta_pos = torch.where(freeze_mask, torch.zeros_like(delta_pos), delta_pos)
        envs.gym.fetch_results(envs.sim, True)
        envs.gym.sync_frame_time(envs.sim)
        envs.refresh_base_tensors()
        envs.refresh_env_tensors()
        next_tgt_pos=envs.fingertip_centered_pos.clone() + delta_pos
        next_tgt_quat=envs.fingertip_centered_quat.clone()
        action_list.append(delta_pos.detach().cpu().numpy())
        envs._move_gripper_to_eef_pose(env_ids, 
                                        ctrl_tgt_pos=next_tgt_pos, 
                                        ctrl_tgt_quat=next_tgt_quat, 
                                        sim_steps=10, 
                                        if_log=log_step, 
                                        close_gripper=True,
                                        log_freq=10,
                                        log_extra=False,
                                        log_first_only=True
                                        )
    envs._log_robot_state(envs.success_env_ids)
    envs._log_object_state(envs.success_env_ids)

    # Success Checker
    
    action_array=np.stack(action_list)
    zeros = np.zeros((action_array.shape[0], action_array.shape[1], 6))
    # cond_z = envs.plug_pos[:, 2] < 0.425
    # Condition 2: XY distance < 0.0005
    # diff = init_plug_pos - envs.plug_pos                # shape (N, 3)
    # xy_dist = torch.norm(diff[:, :2], dim=1)            # shape (N,)
    # cond_xy = xy_dist < 0.001

    # Combine conditions
    # valid_idx = torch.nonzero(cond_z & cond_xy, as_tuple=False).squeeze(1)
    action_array_padded = np.concatenate([action_array, zeros], axis=-1)
    # envs.success_env_ids=np.intersect1d(envs.success_env_ids, valid_idx.cpu().numpy())
     # for env_id in range(12):
    #     envs.save_first_env_images(out_dir="rollout", index=env_id, reverse=False) 
    envs._save_log_traj(action_array_padded)  
    os._exit(0)


if __name__ == "__main__":
    run_env()
