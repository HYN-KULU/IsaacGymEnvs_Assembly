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
from isaacgym import gymapi, gymtorch, torch_utils
import cv2
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
    data=load_processed_dataset("/home/ubuntu/automate/IsaacGymEnvs_Assembly/processed_dataset_depth_relative_1107_01053.h5")
    all_actions = data['actions'][()]  # (N, K, 9)
    delta_pos = all_actions[..., 0:3]  # (N, K, 3)
    delta_rot = all_actions[...,3:]
    # delta_pos = all_actions[..., 0:3]  # (N, K, 3)
    # delta_rot = all_actions[...,3:]
    pos_min = torch.from_numpy(delta_pos.min(axis=(0, 1))).cuda()
    pos_max = torch.from_numpy(delta_pos.max(axis=(0, 1))).cuda()
    rot_min=torch.from_numpy(delta_rot.min(axis=(0,1))).cuda()
    rot_max=torch.from_numpy(delta_rot.max(axis=(0,1))).cuda()
    ### Load Policy
    # ckpt_path = "../logs/automate/diffusion_policy_depth_relative_1012_dagger/policy_epoch_300.ckpt"  # or policy_last.ckpt
    ckpt_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/diffusion_policy_depth_relative_1107_01053/policy_last.ckpt"  # or policy_last.ckpt
    policy = Diffusion_Policy(
        num_action=10,
        obs_feature_dim=512,
        hidden_dim=512,
        proprio_dim=7,
        action_dim=9
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    # import pdb;pdb.set_trace()
    env_ids=torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11], device='cuda:0')
    envs.reset_idx(env_ids)
    envs.disassemble_plug_from_socket_eval_init()
    init_plug_pos = envs.plug_pos.clone()
    save_dir = "eval_visual"
    os.makedirs(save_dir, exist_ok=True)
    envs.visualize_top_camera(0, f"eval_visual/visualize_eval_top_camera.png")
    torch.set_printoptions(precision=5, sci_mode=False)
    image_list=[[] for _ in range(12)]
    target_final_pos=torch.tensor([4.3678e-04, 1.3084e-03, 4.5973e-01], device='cuda:0')
    def quat_axis_angle_error(q_current: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        """
        Compute quaternion orientation error between q_current and q_target
        and return it as a 3D rotation vector (axis * angle).
        Compatible with Isaac Gym torch_utils quaternions.
        """
        # Normalize quaternions
        q_current = q_current / q_current.norm(p=2, dim=-1, keepdim=True)
        q_target  = q_target / q_target.norm(p=2, dim=-1, keepdim=True)

        # Relative rotation: q_rel = q_target * conj(q_current)
        q_conj = torch.cat([-q_current[..., :3], q_current[..., 3:]], dim=-1)
        q_rel  = torch_utils.quat_mul(q_target, q_conj)

        # Clamp scalar part to valid range
        qw = torch.clamp(q_rel[..., 3], -1.0, 1.0)
        angle = 2.0 * torch.acos(qw)

        # Compute normalized axis
        sin_half_angle = torch.sqrt(1.0 - qw * qw)
        axis = torch.zeros_like(q_rel[..., :3])
        mask = sin_half_angle > 1e-6
        axis[mask] = q_rel[..., :3][mask] / sin_half_angle[mask].unsqueeze(-1)
        axis[~mask] = torch.tensor([1.0, 0.0, 0.0], device=q_rel.device)  # arbitrary

        # Rotation vector
        rot_vec = axis * angle.unsqueeze(-1)
        return rot_vec
    dof_state_tensor = envs.gym.acquire_dof_state_tensor(envs.sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    envs.gym.refresh_jacobian_tensors(envs.sim)
    envs.gym.refresh_rigid_body_state_tensor(envs.sim)

    # Get current end-effector pose
    current_pos = envs.fingertip_centered_pos.clone()
    current_quat = envs.fingertip_centered_quat.clone()
    body_names = envs.gym.get_actor_rigid_body_names(envs.env_ptrs[0], 0)
    for i, name in enumerate(body_names): 
        print(i,name)
    # Define target 2 cm forward along +X
    target_pos = current_pos.clone()
    target_pos[:, 2] += 0.02
    target_quat = current_quat.clone()
    franka_jacobian=gymtorch.wrap_tensor(envs.gym.acquire_jacobian_tensor(envs.sim, "franka"))
    # Extract Jacobian for fingertip_centered link
    hand_idx = 11
    jacobian = franka_jacobian[:, hand_idx, :, :7]  # [12, 6, 7]

    # Compute damped least squares IK
    damping = 0.05
    step_size = 0.5

    dof_state_tensor = envs.gym.acquire_dof_state_tensor(envs.sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    q = dof_state[:, 0].view(envs.num_envs, -1)[:, :7].clone()

    pos_err = target_pos - current_pos
    orn_err = quat_axis_angle_error(current_quat, target_quat)
    dpose = torch.cat([pos_err, orn_err], dim=-1).unsqueeze(-1)  # [12, 6, 1]

    j_T = torch.transpose(jacobian, 1, 2)
    I6 = torch.eye(6, device=q.device).unsqueeze(0)
    dq = j_T @ (torch.inverse(jacobian @ j_T + damping * I6) @ dpose)
    dq = dq.squeeze(-1)

    # Update joint positions
    q[:, :7] += step_size * dq
    target_joint_pos = dof_state[:, 0].clone().view(envs.num_envs, -1)
    target_joint_pos[:, :7] = q
    target_joint_pos = target_joint_pos.flatten()

    # Apply to simulator
    for _ in range(20):
        envs.gym.set_dof_position_target_tensor(envs.sim, gymtorch.unwrap_tensor(target_joint_pos))
        envs.gym.simulate(envs.sim)
        envs.gym.fetch_results(envs.sim, True)
        envs.gym.refresh_rigid_body_state_tensor(envs.sim)

    print("EE before:", current_pos[0])
    print("EE after :", envs.fingertip_centered_pos[0])
    import pdb;pdb.set_trace()
   

if __name__ == "__main__":
    run_env()
