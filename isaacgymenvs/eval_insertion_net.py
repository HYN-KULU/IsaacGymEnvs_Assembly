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
from policy.insertion_net import InsertionNet
import h5py
def load_processed_dataset(filename):
    import h5py
    data = {}
    with h5py.File(filename, "r") as f:
        # data["depth"] = f["depth"][()]               # (N, H, W)
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
    data=load_processed_dataset("/home/ubuntu/automate/IsaacGymEnvs_Assembly/processed_dataset_depth_relative_insertion_net_action.h5")
    h5_init_depth=h5py.File("/home/ubuntu/automate/IsaacGymEnvs_Assembly/processed_dataset_depth_relative_insertion_net_init_depth.h5", "r")
    task_id=envs.cfg_task.env.desired_subassemblies[0]
    task_id_mapping={"asset_00021":0, "asset_00028":1, "asset_00030":2, "asset_00042":3, "asset_00110":4, "asset_00681":5}
    # init_depth=h5_init_depth["init_depth"][task_id_mapping[task_id]]
    # init_depth=torch.from_numpy(init_depth).cuda()
    # init_depth = torch.clamp(init_depth, min=-0.5, max=0.0)
    # init_depth = (init_depth - (-0.1890)) / 0.0795
    # init_depth = init_depth.repeat(12,1,1,1).cuda()   # (12,1,480,640)
    all_actions = data['actions'][()]  # (N, K, 9)
    delta_pos = all_actions[..., 0:3]  # (N, K, 3)
    delta_rot = all_actions[...,3:]
    # delta_pos = all_actions[..., 0:3]  # (N, K, 3)
    # delta_rot = all_actions[...,3:]

    pos_min =delta_pos.min(axis=(0,1))
    pos_max = delta_pos.max(axis=(0,1))
    rot_min=delta_rot.min(axis=(0,1))
    rot_max=delta_rot.max(axis=(0,1))
    ckpt_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/insertion_net_ckpt/epoch_60.pt"  # or policy_last.ckpt
    policy = InsertionNet(
        feature_dim=512,
        hidden_dim=512,
        action_dim=9,
        shared_encoder=True  # 必须和训练时一致
    ).cuda()

    ckpt = torch.load(ckpt_path, map_location="cuda")

    state_dict = ckpt["model"]

    # --- Strip module. ---
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    policy.load_state_dict(new_state_dict, strict=True)
    print("Loaded epoch_60")
    env_ids=torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11], device='cuda:0')
    envs.reset_idx(env_ids)
    init_plug_pos = envs.plug_pos.clone()
    init_depth=torch.from_numpy(envs.get_wrist_camera_depth()).cuda()
    init_depth = torch.clamp(init_depth, min=-0.5, max=0.0)
    init_depth = (init_depth - (-0.1890)) / 0.0795
    init_depth = init_depth.unsqueeze(1).cuda()  
    envs.disassemble_plug_from_socket_eval_init()
    save_dir = "eval_visual"
    os.makedirs(save_dir, exist_ok=True)
    envs.visualize_top_camera(0, f"eval_visual/visualize_eval_top_camera.png")
    torch.set_printoptions(precision=5, sci_mode=False)
    # init depth
    for t in range(100):
        print("Timestep: ",t)
        fingertip_centered_pos=envs.fingertip_centered_pos.clone()
        fingertip_centered_quat=envs.fingertip_centered_quat.clone()
        ## Process the depth data
        # Current Depth
        depth=torch.from_numpy(envs.get_wrist_camera_depth()).cuda()
        depth = torch.clamp(depth, min=-0.5, max=0.0)
        depth = (depth - (-0.1890)) / 0.0795
        depth = depth.squeeze(0).unsqueeze(1).cuda()          # (12,1,480,640)

        raw_actions=policy(depth,init_depth)
        predict_actions=unnormalize_actions(raw_actions,pos_min,pos_max,rot_min,rot_max)
        next_tgt_pos=envs.fingertip_centered_pos.clone()
        next_tgt_quat=envs.fingertip_centered_quat.clone()
        tcp=torch.concatenate([next_tgt_pos,next_tgt_quat],axis=1).cpu().numpy()
        tcp_9d=torch.from_numpy(xyz_rot_transform(tcp,from_rep="quaternion", to_rep="rotation_6d")).cuda()
        # Freeze the simulation
        envs.gym.fetch_results(envs.sim, True)
        envs.gym.sync_frame_time(envs.sim)
        envs.refresh_base_tensors()
        envs.refresh_env_tensors()
        # delta_pos = predict_actions[:,:3]
        # delta_pos[:,2] = -0.0005
        # next_tgt_pos = next_tgt_pos + delta_pos     
        # delta_quat=predict_actions[:,3:]
        print(predict_actions[:,:3])
        predict_actions[:,2] = -0.0005
        next_tcp_9d = tcp_9d + predict_actions
        next_tcp = torch.from_numpy(xyz_rot_transform(next_tcp_9d.detach().cpu().numpy(),from_rep="rotation_6d", to_rep="quaternion")).cuda()
        envs._move_gripper_to_eef_pose(env_ids, 
                                            ctrl_tgt_pos=next_tcp[:,:3], 
                                            ctrl_tgt_quat=envs.fingertip_centered_quat.clone(), 
                                            sim_steps=50, 
                                            if_log=True, 
                                            close_gripper=True,
                                            log_freq=30,
                                            log_extra=True
                                            )
    # Visualize
    for env_id in range(12):
        envs.save_first_env_images(out_dir="rollout", index=env_id, reverse=False)
    import pdb;pdb.set_trace()
    
    os._exit(0)


if __name__ == "__main__":
    run_env()
