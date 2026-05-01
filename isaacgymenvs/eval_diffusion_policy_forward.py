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
    # data=load_processed_dataset("/home/ubuntu/automate/IsaacGymEnvs_Assembly/utils/preprocess/processed_dataset_forward_0314.h5")
    data=load_processed_dataset("/home/ubuntu/automate/IsaacGymEnvs_Assembly/data_utils/preprocess/processed_dataset_forward_0421_force_feedback.h5")
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
    # ckpt_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/finetune_diffusion_policy/00345_policy_epoch_100.ckpt"  # or policy_last.ckpt
    ckpt_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/diffusion_policy_forward_20004/policy_epoch_199.ckpt"  # or policy_last.ckpt
    # ckpt_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/diffusion_policy_forward/policy_epoch_270.ckpt"  # or policy_last.ckpt
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
    dists=[]
    # import pdb;pdb.set_trace()
    env_ids=torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11], device='cuda:0')
    for trial in range(5):
        envs.visualize_rgb = True
        # if trial ==0:
            # envs.visualize_rgb = True
        # else:
        #     envs.visualize_rgb = False
        print("===== Trial ", trial, " =====")
        envs.reset_idx(env_ids)
        init_plug_pos = envs.plug_pos.clone()
        
        envs.disassemble_plug_from_socket_eval_init()
        save_dir = "eval_visual"
        os.makedirs(save_dir, exist_ok=True)
        envs.visualize_top_camera(0, f"eval_visual/visualize_eval_top_camera.png")
        torch.set_printoptions(precision=5, sci_mode=False)
        rate = 2
        
        for t in range(50):
            print("Timestep: ",t)
            fingertip_centered_pos=envs.fingertip_centered_pos.clone()
            fingertip_centered_quat=envs.fingertip_centered_quat.clone()
            ## Process the depth data
            depth = envs.get_wrist_camera_depth()
            rotated_depth_list=[]
            camera3_vinv_list=[]
            for i in range(12):
                rotated_depth_list.append(depth[i].copy())
                # rotated_depth_list.append(np.rot90(depth[i], 2))
                camera3_vinv_list.append(torch.inverse(torch.tensor(envs.gym.get_camera_view_matrix(envs.sim,envs.env_ptrs[i],envs.camera_handles[i]["panda"]))).to(envs.device).detach().cpu().numpy())
            camera3_vinv = np.array(camera3_vinv_list)
            depth_rot = np.stack(rotated_depth_list)
            depth_rot = torch.from_numpy(depth_rot).cuda()
            depth_rot = torch.clamp(depth_rot, min=-0.5, max=0.0)
            depth_rot = (depth_rot - (-0.1890)) / 0.0795

            tcp=torch.concatenate([fingertip_centered_pos,fingertip_centered_quat],axis=1).cpu().numpy()
            proprioception=torch.from_numpy(xyz_rot_transform(tcp,from_rep="quaternion", to_rep="rotation_6d")[:,:]).cuda()
            raw_actions=policy(depth_rot.unsqueeze(dim=1)[...,::rate,::rate][...,30:-30,40:-40],proprioception[...,2:],actions=None)
            predict_actions=unnormalize_actions(raw_actions,pos_min,pos_max,rot_min,rot_max)
            next_tgt_pos=envs.fingertip_centered_pos.clone()
            next_tgt_quat=envs.fingertip_centered_quat.clone()
            # Freeze the simulation
            envs.gym.fetch_results(envs.sim, True)
            envs.gym.sync_frame_time(envs.sim)
            envs.refresh_base_tensors()
            envs.refresh_env_tensors()
            current_gripper_pos = envs.fingertip_centered_pos.clone()
            # Add the k steps together, and launch a move
            for i in range(10):    
                if_log = True if i==0 else False   
                tcp=torch.concatenate([next_tgt_pos,next_tgt_quat],axis=1).cpu().numpy()
                proprioception=torch.from_numpy(xyz_rot_transform(tcp,from_rep="quaternion", to_rep="rotation_6d")).cuda()
                delta_pos=predict_actions[:,i,:3] 
                # print("Action Delta Pose: ",delta_pos)
                camera3_vinv_torch = torch.from_numpy(camera3_vinv).to(delta_pos.device)
                R_wc = camera3_vinv_torch[:, :3, :3] 
                R_cw = R_wc.transpose(1, 2) 
                delta_world = torch.matmul(R_cw,delta_pos.unsqueeze(-1)).squeeze(-1)
                delta_rot6d=predict_actions[:,i,3:]
                curr_rot6d=proprioception[:,3:]
                next_rot6d=curr_rot6d + delta_rot6d
                next_tcp=torch.concatenate([next_tgt_pos + delta_world,next_rot6d],axis=1).cpu().numpy()
                next_tgt_pose=torch.from_numpy(xyz_rot_transform(next_tcp,from_rep="rotation_6d",to_rep="quaternion")).cuda()
                next_tgt_pos=next_tgt_pose[:,:3]
                next_tgt_quat=next_tgt_pose[:,3:]
                next_tgt_pos=next_tgt_pose[:,:3].clone()
                next_tgt_quat=next_tgt_pose[:,3:].clone()

                envs._move_gripper_to_eef_pose(env_ids, 
                                                    ctrl_tgt_pos=next_tgt_pos, 
                                                    ctrl_tgt_quat=next_tgt_quat, 
                                                    sim_steps=10, 
                                                    if_log=if_log, 
                                                    close_gripper=True,
                                                    log_freq=10,
                                                    log_extra=True,
                                                    log_first_only = True
                                                    )
        # Visualize
        # for env_id in range(12):
        #     envs.save_first_env_images(out_dir="rollout", index=env_id, reverse=False)
        # quat_success = envs.plug_quat[envs.success_env_ids]
        # w = quat_success[:, 3]
        # vertical_mask = w > 0.85
        valid_ids = envs.success_env_ids
        plug_pos = envs.plug_pos[valid_ids]
        init_pos = envs.init_plug_pos[valid_ids]
        delta_xy = plug_pos[:, :2] - init_pos[:, :2]
        dist_xy = torch.norm(delta_xy, dim=1)
        dist_z = torch.abs(plug_pos[:, 2] - init_pos[:, 2])
        dist_list = list(
            zip(
                dist_xy.detach().cpu().tolist(),
                dist_z.detach().cpu().tolist()
            )
        )
        print(dist_list)
        for env_id in range(12):
            envs.save_first_env_images(out_dir=f"rollout_eval_forward/{envs.cfg_task.env.desired_subassemblies[0]}/{trial}", index=env_id, reverse=False)
        # os._exit(0)
        # print("Final (XY dist, Z dist) for successful vertical plugs:", dist_list)
        dists.extend(dist_list)
        np.save(f"eval_dp_forward_transform/{envs.cfg_task.env.desired_subassemblies[0]}.npy", np.array(dists))
        # np.save(f"eval_result_baseline_diffusion_policy/{envs.cfg_task.env.desired_subassemblies[0]}.npy", np.array(dists))
    
    os._exit(0)


if __name__ == "__main__":
    run_env()
