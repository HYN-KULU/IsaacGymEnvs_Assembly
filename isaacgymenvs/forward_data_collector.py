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
from quat_helper import *
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

def compute_insert_action(delta_pos):
    """
    delta_pos: (N,3) tensor, plug_pos error (target - current)

    return:
        action: (N,3)
    """

    action = torch.zeros_like(delta_pos)

    delta_xy = delta_pos[:, :2]
    delta_xy_norm = torch.norm(delta_xy, dim=1, keepdim=True)

    tmp = delta_pos.clone()
    tmp[:, 2] = 0
    delta_norm = torch.norm(tmp, dim=1, keepdim=True) + 1e-8
    # print(delta_pos)
    # print(delta_xy_norm)
    mask1 = (delta_xy_norm[:,0] > 0.002)
    mask2 = (delta_xy_norm[:,0] <= 0.002) & (delta_xy_norm[:,0] > 0.001)
    mask3 = (delta_xy_norm[:,0] <= 0.001) & (delta_xy_norm[:,0] > 0.0003)
    mask4 = (delta_xy_norm[:,0] <= 0.0003) & (delta_xy_norm[:,0] > 0.0001)
    mask5 = (delta_xy_norm[:,0] < 0.0001)

    # branch1
    action[mask1.squeeze()] = tmp[mask1.squeeze()] / delta_norm[mask1] * 0.0003

    # branch2
    action[mask2.squeeze()] = tmp[mask2.squeeze()] / delta_norm[mask2] * 0.00015

    # branch3
    action[mask3.squeeze()] = tmp[mask3.squeeze()] / 5 
    action[mask3.squeeze(), 2] = -0.0001

    # branch4
    action[mask4.squeeze()] = tmp[mask4.squeeze()] 
    action[mask4.squeeze(), 2] = -0.0002

    # branch5
    action[mask5.squeeze()] = tmp[mask5.squeeze()] 
    action[mask5.squeeze(), 2] = -0.0004
    action_xy = action[:, :2]
    action_xy_norm = torch.norm(action_xy, dim=1, keepdim=True)

    # min_norm = 0.0005

    # scale = torch.clamp(min_norm / (action_xy_norm + 1e-8), min=1.0)

    # action[:, :2] = action[:, :2] * scale
    # print(action)
    return action

TRAIN_TASKS = {
    "00004", "00015", "00016", "00021", "00028", "00030", "00042",
    "00074", "00077", "00078", "00081", "00103", "00110", "00117",
    "00133", "00138", "00141", "00163", "00175", "00186", "00187",
    "00192", "00211", "00213", "00255", "00256", "00271", "00293",
    "00301", "00318", "00319", "00320", "00329", "00345", "00346",
    "00360", "00388", "00410", "00417", "00422", "00426", "00437",
    "00444", "00446", "00471", "00480", "00499", "00506", "00514",
    "00537", "00553", "00559", "00581", "00597", "00614", "00615",
    "00638", "00648", "00649", "00659", "00681", "00686", "00700",
    "00703", "00726", "00731", "00768", "00783", "00855", "00860",
    "01026", "01029", "01036", "01041", "01079", "01092", "01102",
    "01129", "01132", "01136"
}

from quat_utils import *
from diffusion_utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform
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
   
    num_envs = 12
    # import pdb;pdb.set_trace()
    env_ids=torch.tensor(range(num_envs), device='cuda:0')
    envs.reset_idx(env_ids)
    for run_seed in range(6):
        envs.run_id = run_seed
        envs.visualize_rgb = False
        envs.reset_idx(env_ids)
        init_plug_pos = envs.init_root_pose[:, :3].clone()
        init_plug_quat = envs.init_root_pose[:, 3:7].clone()
        envs.disassemble_plug_from_socket_eval_init()
        save_dir = "eval_visual"
        os.makedirs(save_dir, exist_ok=True)
        # envs.visualize_top_camera(0, f"eval_visual/visualize_eval_top_camera.png")
        torch.set_printoptions(precision=5, sci_mode=False)
        image_list=[[] for _ in range(num_envs)]
        offset = torch.zeros(num_envs, 3)
        # -0.01 ~ 0.01 for task 00042
        # 1. Define the shape
        shape = (num_envs, 2)

        # 2. Generate magnitudes in range [0.003, 0.005]
        magnitudes = (torch.rand(shape) * 0.002) + 0.003

        # 3. Create a random sign mask (-1 or 1)
        signs = torch.where(torch.rand(shape) > 0.5, 1.0, -1.0)
        target_plug_quat = init_plug_quat.clone()
        # 4. Apply to your offset
        offset[:, :2] = magnitudes * signs

        nums = random.sample(range(200), 50)
        action_list=[]
        depth_list=[]
        if run_seed in [0,1,2]:
            recovery = False
        else:
            recovery = True
        if run_seed in [3]:
            hard_recovery = True
            min_angle_error = 91
            max_angle_error = 179
        else:
            hard_recovery = False
            min_angle_error = 5
            max_angle_error = 90
        if recovery:
            disturb_steps = 200
            if hard_recovery:
                steps = 600
            else:
                steps = 500
        else:
            steps = 350
        
        target_plug_quat_disturbed, yaw_deg = disturb_quat_in_yaw(
            target_plug_quat,
            deg_min=min_angle_error,
            deg_max=max_angle_error,
            both_directions=True,
            world_frame=True,
        )
        ###
        force_traj = None
        recovery_steps_after_spike = 10
        recovery_counter = np.zeros(num_envs, dtype=np.int32)
        for i in range(steps):
        # for i in range(200):
            print(f"Step {i}")
            # log_step=False
            delta_pos = init_plug_pos - envs.plug_pos
            target_plug_quat = init_plug_quat
            if recovery:
                if i < disturb_steps:
                    delta_pos = delta_pos + offset.cuda()
                    delta_pos[:,2] = random.uniform(-0.0002,-0.0001)
                    target_plug_quat = target_plug_quat_disturbed
                    log_step = False
                else:
                    log_step = True
            else:
                log_step = True
            # if i > 100:
                # delta_pos[:,2] = -0.001 * (i-100)
            ############################################################################################################# For debugging
            # delta_pos = envs.socket_pos - envs.plug_pos
            # print("envs.socket_pos: ", envs.socket_pos)
            # print("envs.plug_pos: ", envs.plug_pos)
            # print("delta_pos: ", delta_pos)
            delta_xy = delta_pos[:, :2]
            action = compute_insert_action(delta_pos)
            # action[:] = 0
            envs.gym.fetch_results(envs.sim, True)
            envs.gym.sync_frame_time(envs.sim)
            envs.refresh_base_tensors()
            envs.refresh_env_tensors()
            next_tgt_quat=envs.fingertip_centered_quat.clone()
            previous_tgt_quat = next_tgt_quat.clone()
            previous_tgt_pos = envs.fingertip_centered_pos.clone()
            # log_step = True
            if i > 200:
                delta = force_traj[-1] - force_traj[0]
                vals = delta[:, 0, 2]
                idx = np.where(vals > 0.05)[0]
                if len(idx) > 0:
                    recovery_counter[idx] = recovery_steps_after_spike
            recovering_idx = np.where(recovery_counter > 0)[0]
            recovery_counter[recovering_idx] -= 1
            # print(recovering_idx)
            vals = torch.tensor([0.0, 1e-4, 2e-4], device=action.device)
            rand_idx = torch.randint(0, 3, (len(recovering_idx),), device=action.device)
            action[recovering_idx, 2] = vals[rand_idx]
            # print(action[4])
            next_tgt_pos= previous_tgt_pos + action
            if i > 50:
                q_step, angle_err = quat_step_toward(
                    current_q=envs.plug_quat,
                    target_q=target_plug_quat,
                    max_angle_step_deg=0.3,
                )

                # print("quat_angle_err_deg:", torch.rad2deg(angle_err))

                next_tgt_quat = quat_mul(q_step, envs.fingertip_centered_quat.clone())
                next_tgt_quat = quat_normalize(next_tgt_quat)
            
            if log_step:
                # previous / next target pos, quat
                prev_pos_np = previous_tgt_pos.detach().cpu().numpy()[:, None, :]   # [N, 1, 3]
                prev_quat_np = previous_tgt_quat.detach().cpu().numpy()[:, None, :] # [N, 1, 4]

                next_pos_np = next_tgt_pos.detach().cpu().numpy()[:, None, :]       # [N, 1, 3]
                next_quat_np = next_tgt_quat.detach().cpu().numpy()[:, None, :]     # [N, 1, 4]

                # build tcp = [x, y, z, qx, qy, qz, qw]
                prev_tcp = np.concatenate([prev_pos_np, prev_quat_np], axis=2)      # [N, 1, 7]
                next_tcp = np.concatenate([next_pos_np, next_quat_np], axis=2)      # [N, 1, 7]

                # convert quaternion -> rotation_6d
                prev_tcp_rotation_6d = xyz_rot_transform(
                    prev_tcp,
                    from_rep="quaternion",
                    to_rep="rotation_6d"
                )  # expected [N, 1, 9] = [xyz + rot6d]

                next_tcp_rotation_6d = xyz_rot_transform(
                    next_tcp,
                    from_rep="quaternion",
                    to_rep="rotation_6d"
                )  # expected [N, 1, 9]

                # only take the rotation_6d part, not xyz
                prev_rot6d = prev_tcp_rotation_6d[:, 0, 3:]   # [N, 6]
                next_rot6d = next_tcp_rotation_6d[:, 0, 3:]   # [N, 6]

                action_rot6d = next_rot6d - prev_rot6d        # [N, 6]

                # action is delta_pos, shape [N, 3]
                action_9d = torch.cat([
                    action,
                    torch.from_numpy(action_rot6d).to(action.device, dtype=action.dtype)
                ], dim=-1)                                    # [N, 9]

                action_list.append(action_9d.detach().cpu().numpy())
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
            # print(envs.vec_sensor_tensor.shape,envs.vec_sensor_tensor[0])
            force_np = envs.vec_sensor_tensor.view(-1, 3, 6).detach().cpu().numpy()
            if force_traj is None:
                # 第一次初始化
                force_traj = force_np[None]   # shape: (1, N, 3, 6)
            else:
                # 在时间维度拼接
                force_traj = np.concatenate([force_traj, force_np[None]], axis=0)
        # envs.success_env_ids = torch.arange(0, num_envs, device=envs.device)
        envs._log_robot_state(envs.success_env_ids)
        envs._log_object_state(envs.success_env_ids)
        # force_array = torch.stack(force_traj,dim=0).detach().cpu().numpy()
        force_array = force_traj
        # Success Checker
        action_array=np.stack(action_list)
        # if True:
        #     folder = "rollout_orient"
        #     for env_id in range(12):
        #             folder= "rollout_orient"
        #             if recovery:
        #                 folder = "rollout_recovery"
        #             envs.save_first_env_images(out_dir=f"{folder}/{envs.cfg_task.env.desired_subassemblies[0]}", index=env_id, reverse=False) 
        envs._save_log_traj(action_array[:,envs.success_env_ids,:], force_array[:,envs.success_env_ids,:,:])  


    os._exit(0)


if __name__ == "__main__":
    run_env()