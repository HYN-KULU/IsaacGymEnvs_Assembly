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
    ### Load Policy
    ckpt_path = "../logs/automate/diffusion_policy_depth_absolute_1003_ckpt/policy_epoch_100.ckpt"  # or policy_last.ckpt
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
    target_final_pos=torch.tensor([4.3678e-04, 1.3084e-03, 4.5973e-01], device='cuda:0')
    ###
    pos_min=torch.tensor([-0.04752755, -0.04311813,  0.45951718]).cuda()
    pos_max=torch.tensor([0.0455904,  0.04908895, 0.599305  ]).cuda()
    rot_min=torch.tensor([ 0.975577 ,  -0.17272623, -0.17862518, -0.16902371,  0.96979964, -0.18132125]).cuda()
    rot_max=torch.tensor([1.  ,       0.18020557, 0.1827667 , 0.16844484 ,0.99999994 ,0.17435487]).cuda()

    # import pdb;pdb.set_trace()
    env_ids=torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11], device='cuda:0')
    save_dir = "eval_visual"
    os.makedirs(save_dir, exist_ok=True)
    id_list=[]
    for _ in range(1):
        envs.reset_idx(env_ids)
        envs.disassemble_plug_from_socket_eval_init()
        # envs.visualize_top_camera(0, f"eval_visual/visualize_eval_top_camera.png")
        torch.set_printoptions(precision=5, sci_mode=False)
        image_list=[[] for _ in range(12)]
        ###
        for t in range(30):
            print("Timestep: ",t)
            fingertip_centered_pos=envs.fingertip_centered_pos.clone()
            fingertip_centered_quat=envs.fingertip_centered_quat.clone()
            ## Process the depth data
            depth=torch.from_numpy(envs.get_wrist_camera_depth()).cuda()
            depth = torch.clamp(depth, min=-0.5, max=0.0)
            depth = (depth - (-0.1890)) / 0.0795

            tcp=torch.concatenate([fingertip_centered_pos,fingertip_centered_quat],axis=1).cpu().numpy()
            proprioception=torch.from_numpy(xyz_rot_transform(tcp,from_rep="quaternion", to_rep="rotation_6d")).cuda()
            raw_actions=policy(depth.unsqueeze(dim=1),proprioception,actions=None)
            predict_actions=unnormalize_actions(raw_actions,pos_min,pos_max,rot_min,rot_max).detach().cpu().numpy()
            actions=torch.from_numpy(xyz_rot_transform(predict_actions,from_rep="rotation_6d", to_rep="quaternion")).cuda()
            for i in range(10):
                fingertip_centered_pos=envs.fingertip_centered_pos.clone()
                fingertip_centered_quat=envs.fingertip_centered_quat.clone()
                envs._apply_absolute_action(env_ids, 
                                        ctrl_tgt_pos=actions[:,i,:3], 
                                        ctrl_tgt_quat=actions[:,i,3:], 
                                        sim_steps=1, 
                                        if_log=False, 
                                        close_gripper=True)
                images=envs.get_wrist_camera_visuals()
                for env_id in range(12):
                    image_list[env_id].append(images[env_id])
        errors=torch.norm(envs.fingertip_centered_pos-target_final_pos,dim=1)
        idx = torch.nonzero(errors >= 0.01, as_tuple=True)[0]
        id_list.append(idx)
    for env_id in range(12):
        save_video_from_images(image_list[env_id], out_path=f"output_video_{env_id}.mp4")
    print(torch.concatenate(id_list).shape)
    
    os._exit(0)


if __name__ == "__main__":
    run_env()
