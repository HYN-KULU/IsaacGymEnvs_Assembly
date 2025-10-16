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
from dataset.diffusion_policy_dataset_rgb import DepthActionDataset
from policy.diffusion_policy_rgb import Diffusion_Policy
from diffusion_utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform
import cv2

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

import imageio
import numpy as np

def save_video_from_images(image_list, out_path="output_video.mp4", fps=20):
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
    pos=np.load("error_pos_gripper_1014.npy")
    quat=np.load("error_quat_gripper_1014.npy")
    plug_pos=np.load("error_pos_plug_1014.npy")
    plug_quat=np.load("error_quat_plug_1014.npy")
    # test_id=12
    # ctrl_tgt_pos = torch.from_numpy(np.repeat(pos[test_id:test_id+1, :], 12, axis=0)).cuda()
    # ctrl_tgt_quat = torch.from_numpy(np.repeat(quat[test_id:test_id+1, :], 12, axis=0)).cuda()
    # ctrl_plug_pos= torch.from_numpy(np.repeat(plug_pos[test_id:test_id+1, :], 12, axis=0)).cuda()
    # ctrl_plug_quat= torch.from_numpy(np.repeat(plug_quat[test_id:test_id+1, :], 12, axis=0)).cuda()
    ctrl_tgt_pos=torch.from_numpy(pos[cfg.seed*12:cfg.seed*12+12,:]).cuda()
    ctrl_tgt_quat=torch.from_numpy(quat[cfg.seed*12:cfg.seed*12+12,:]).cuda()
    ctrl_plug_pos=torch.from_numpy(plug_pos[cfg.seed*12:cfg.seed*12+12,:]).cuda()
    ctrl_plug_quat=torch.from_numpy(plug_quat[cfg.seed*12:cfg.seed*12+12,:]).cuda()
    # envs.disassemble_plug_dagger(ctrl_tgt_pos=ctrl_tgt_pos ,ctrl_tgt_quat=ctrl_quat_pos)
    # current_env_plug_pos=envs.plug_pos.clone()
    # current_env_plug_quat=envs.plug_quat.clone()
    # current_env_plug_pos[:,2]=0.4189 
    # current_env_plug_pos[:,2]=current_env_plug_pos[:,2]+0.018198      
    envs.disassemble_plug_dagger_generate_trajectory(ctrl_tgt_pos,ctrl_tgt_quat, ctrl_plug_pos, ctrl_plug_quat)
    # import pdb;pdb.set_trace()
    os._exit(0)


if __name__ == "__main__":
    run_env()
