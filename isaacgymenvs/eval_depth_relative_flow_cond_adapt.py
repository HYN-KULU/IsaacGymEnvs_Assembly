import isaacgym
import isaacgymenvs
from isaacgym import gymapi, gymtorch, torch_utils
import hydra
from omegaconf import DictConfig
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.tasks import isaacgym_task_map
import os
import torch
from collections import OrderedDict
# from policy.diffusion_policy import Diffusion_Policy
from dataset.diffusion_policy_dataset import DepthActionDataset
from policy.diffusion_policy_flow_cond import Diffusion_Policy
from diffusion_utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform
import cv2
from policy.flow_policy import FlowPolicy
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

def rewrite_monai_keys(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        nk = nk.replace(".sub0", ".submodule.0")
        nk = nk.replace(".sub1", ".submodule.1")
        nk = nk.replace(".sub2", ".submodule.2")
        nk = nk.replace(".subconv", ".submodule.conv")
        nk = nk.replace(".subadn", ".submodule.adn")
        new_sd[nk] = v
    return new_sd

def depth_to_vis(depth):
    """
    depth: (H, W), numpy
    return: (H, W, 3) uint8
    """
    d = depth.copy()
    d = np.nan_to_num(d)

    # normalize per-frame for visualization
    d_min, d_max = d.min(), d.max()
    if d_max > d_min:
        d = (d - d_min) / (d_max - d_min)
    else:
        d = np.zeros_like(d)

    d = (d * 255).astype(np.uint8)
    return cv2.cvtColor(d, cv2.COLOR_GRAY2RGB)
def draw_arrows(img, flow, stride=6, scale=5):
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()

    out = img.copy()
    u = flow[0]
    v = flow[1]
    H, W = u.shape

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            dx = u[y, x]
            dy = v[y, x]

            x2 = int(x + dx * scale)
            y2 = int(y + dy * scale)

            cv2.arrowedLine(out, (x, y), (x2, y2),
                            color=(0, 255, 0),
                            thickness=1,
                            tipLength=0.3)
    return out

def flow_to_direction(flow, eps=1e-6):
    """
    flow: (2, H, W)
    return: (2, H, W), unit direction field
    """
    mag = np.linalg.norm(flow, axis=0, keepdims=True)  # (1,H,W)
    return flow / (mag + eps)
def flow_to_hsv(flow):
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()

    u = flow[0]
    v = flow[1]

    mag = np.sqrt(u*u + v*v)
    ang = np.arctan2(v, u)

    H, W = u.shape
    hsv = np.zeros((H, W, 3), dtype=np.uint8)

    hsv[..., 0] = ((ang + np.pi) / (2*np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

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
    data=load_processed_dataset("/home/ubuntu/automate/flow_diffusion_policy_multitask_0113.h5")
    # data=load_processed_dataset("/home/ubuntu/automate/IsaacGymEnvs_Assembly/flow_diffusion_policy_00681_0108.h5")
    # data=load_processed_dataset("/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/flow_diffusion_policy_multitask_00681/flow_diffusion_policy_multitask.h5")
    # data=load_processed_dataset("/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/flow_diffusion_policy_multitask/flow_diffusion_policy_multitask.h5")
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
    ckpt_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/flow_diffusion_policy_multitask_0113/policy_epoch_305.ckpt"  # or policy_last.ckpt
    # ckpt_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/flow_diffusion_policy_multitask_00681/policy_epoch_300.ckpt"  # or policy_last.ckpt
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
    flow_policy = FlowPolicy().to(device)
    # ckpt = torch.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/flow_net_multitask_ckpt/multitask_last.pt", map_location=device)
    ckpt = torch.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/flow_net_adaptation/00015_epoch_30.pt", map_location=device)
    # ckpt = torch.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/flow_net_adaptation/epoch_31.pt", map_location=device)
    # ckpt = torch.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/flow_net_multitask_ckpt/epoch_00360_20.pt", map_location=device)

    sd = ckpt["model"]
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())
    sd = rewrite_monai_keys(sd)
    flow_policy.load_state_dict(sd, strict=True)
    flow_policy.eval()
    dists=[]
    for trial in range(5):
        print("===== Trial ", trial, " =====")
    # import pdb;pdb.set_trace()
        env_ids=torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11], device='cuda:0')
        envs.reset_idx(env_ids)
        init_plug_pos = envs.plug_pos.clone()
        
        envs.disassemble_plug_from_socket_eval_init()
        save_dir = "eval_visual"
        os.makedirs(save_dir, exist_ok=True)
        envs.visualize_top_camera(0, f"eval_visual/visualize_eval_top_camera.png")
        torch.set_printoptions(precision=5, sci_mode=False)
        mask_list = []
        for i in range(12):
                        # index=envs.success_env_ids[i]
                        seg = envs.gym.get_camera_image(
                            envs.sim,
                            envs.env_ptrs[i],
                            envs.camera_handles[i]["panda"],
                            gymapi.IMAGE_SEGMENTATION
                        ).reshape(envs.cam_props.height, envs.cam_props.width)
                        mask_list.append(seg)
        mask = np.stack(mask_list).astype(np.uint8).copy()
        rate = 2
        # mask = np.rot90(mask, k=2, axes=(1, 2))[:,::rate,::rate].copy()
        # mask_tensor = torch.from_numpy(mask).cuda()
        for t in range(50):
            print("Timestep: ",t)
            fingertip_centered_pos=envs.fingertip_centered_pos.clone()
            fingertip_centered_quat=envs.fingertip_centered_quat.clone()
            envs.gym.fetch_results(envs.sim, True)
            envs.gym.sync_frame_time(envs.sim)
            envs.refresh_base_tensors()
            envs.refresh_env_tensors()
            ## Process the depth data
            depth = envs.get_wrist_camera_depth()
            camera_rgb = envs.get_wrist_camera_rgb()
            rotated_depth_list=[]
            rotated_rgb_list=[]
            for i in range(12):
                rotated_depth_list.append(np.rot90(depth[i], 2))
                rotated_rgb_list.append(np.rot90(camera_rgb[i], 2))
            depth_rot = np.stack(rotated_depth_list)
            depth_rot = torch.from_numpy(depth_rot).cuda()
            depth_rot = torch.clamp(depth_rot, min=-0.5, max=0.0)
            depth_rot = (depth_rot - (-0.1890)) / 0.0795
            tcp=torch.concatenate([fingertip_centered_pos,fingertip_centered_quat],axis=1).cpu().numpy()
            proprioception=torch.from_numpy(xyz_rot_transform(tcp,from_rep="quaternion", to_rep="rotation_6d")[:,:]).cuda()
            # depth rot 180 degrees
            # subsample rate 2
            # mask rotate 180 degrees and subsample rate 2
            # flow predicted using processed mask and processed depth
            pred_flow_list=[]
            with torch.no_grad():
                for i in range(12):
                    pred_flow=flow_policy(torch.from_numpy(np.rot90(mask[i], 2)[::rate, ::rate].copy()).unsqueeze(0).unsqueeze(0).cuda(),depth_rot[i][::rate, ::rate].unsqueeze(0).unsqueeze(0).cuda())[0].detach().cpu().numpy()
                    mask_np = (np.rot90(mask[i], 2)[::rate, ::rate] > 0).astype(np.float32)
                    pred_flow = pred_flow * mask_np
                    if i != 12:
                            # rgb = np.rot90(envs.get_wrist_camera_rgb()[i], 2)[::rate, ::rate, :].copy()  # (H, W, 3), uint8 usually
                            rgb = rotated_rgb_list[i][::rate, ::rate, :].copy()  # (H, W, 3), uint8 usually
                            vis_flow = pred_flow  # (2, H, W)

                            # ---- flow ----
                            flow_dir = flow_to_direction(vis_flow)

                            stride = 1
                            arrow_scale = 4.0
                            sparse_flow = np.zeros_like(flow_dir)
                            sparse_flow[:, ::stride, ::stride] = flow_dir[:, ::stride, ::stride]
                            sparse_flow *= arrow_scale

                            flow_hsv = flow_to_hsv(vis_flow)
                            flow_vis = draw_arrows(flow_hsv, sparse_flow)
                            # ---- resize rgb if needed ----
                            if rgb.shape[:2] != flow_vis.shape[:2]:
                                rgb = cv2.resize(
                                    rgb,
                                    (flow_vis.shape[1], flow_vis.shape[0]),
                                    interpolation=cv2.INTER_NEAREST
                                )
                            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                            # ---- concat & save ----
                            vis = np.concatenate([rgb_bgr, flow_vis], axis=1)
                            # vis = flow_vis
                            os.makedirs(f"./flow_vis/{i}", exist_ok=True)
                            out_path = f"./flow_vis/{i}/rgb_flow_{t}.png"
                            cv2.imwrite(out_path, vis)
                        # vis_flow = pred_flow                      # (2, H, W)
                        # depth_small = depth_rot[i][::rate, ::rate].cpu().numpy()  # (H, W)

                        # # ---- depth ----
                        # depth_vis = depth_to_vis(depth_small)

                        # # ---- flow ----
                        # flow_dir = flow_to_direction(vis_flow)

                        # stride = 1
                        # arrow_scale = 4.0
                        # sparse_flow = np.zeros_like(flow_dir)
                        # sparse_flow[:, ::stride, ::stride] = flow_dir[:, ::stride, ::stride]
                        # sparse_flow *= arrow_scale

                        # flow_hsv = flow_to_hsv(vis_flow)
                        # flow_vis = draw_arrows(flow_hsv, sparse_flow)

                        # # ---- resize depth if needed ----
                        # if depth_vis.shape[:2] != flow_vis.shape[:2]:
                        #     depth_vis = cv2.resize(
                        #         depth_vis,
                        #         (flow_vis.shape[1], flow_vis.shape[0]),
                        #         interpolation=cv2.INTER_NEAREST
                        #     )

                        # # ---- concat & save ----
                        # vis = np.concatenate([depth_vis, flow_vis], axis=1)

                        # os.makedirs("./flow_vis", exist_ok=True)
                        # out_path = f"./flow_vis/depth_flow_{t}.png"
                        # cv2.imwrite(out_path, vis)
                    pred_flow_list.append(pred_flow)
            flow = np.stack(pred_flow_list, axis=0) # (B, 2, H, W)
            # predict actions using predicted flow, and rotated + subsampled depth
            proprioception[...,3:] = 0
            raw_actions=policy(depth_rot[:, ::rate, ::rate].unsqueeze(dim=1),proprioception[...,2:],actions=None, flow=torch.from_numpy(flow).cuda())
            predict_actions=unnormalize_actions(raw_actions,pos_min,pos_max,rot_min,rot_max)
            next_tgt_pos=envs.fingertip_centered_pos.clone()
            next_tgt_quat=envs.fingertip_centered_quat.clone()
            # Freeze the simulation

            current_gripper_pos = envs.fingertip_centered_pos.clone()
            # Add the k steps together, and launch a move
            for i in range(5):    
                tcp=torch.concatenate([next_tgt_pos,next_tgt_quat],axis=1).cpu().numpy()
                proprioception=torch.from_numpy(xyz_rot_transform(tcp,from_rep="quaternion", to_rep="rotation_6d")).cuda()
                delta_pos=predict_actions[:,i,:3] 
                # First test with freeze height
                # delta_pos[:,2] = 0
                delta_rot6d=predict_actions[:,i,3:]
                curr_rot6d=proprioception[:,3:]
                next_rot6d=curr_rot6d + delta_rot6d
                next_tcp=torch.concatenate([next_tgt_pos + delta_pos,next_rot6d],axis=1).cpu().numpy()
                next_tgt_pose=torch.from_numpy(xyz_rot_transform(next_tcp,from_rep="rotation_6d",to_rep="quaternion")).cuda()
                next_tgt_pos=next_tgt_pose[:,:3]
                next_tgt_quat=next_tgt_pose[:,3:]
            # Measuring the deviation        
            init_plug_pos_2d = init_plug_pos.clone()
            plug_pos_2d = envs.plug_pos.clone()
            init_plug_pos_2d[:, 2] = 0
            plug_pos_2d[:, 2] = 0
            # raw delta
            deviation = init_plug_pos_2d - plug_pos_2d
            print("Deviation: ", deviation[5])
            # deviation_norm = torch.norm(deviation, dim=1, keepdim=True) + 1e-8
            # # condition mask: True → use raw action
            # mask_raw = deviation_norm < 0.0001

            next_tgt_pose[:,3:] = envs.fingertip_centered_quat.clone()
            next_tgt_pos=next_tgt_pose[:,:3].clone()
            delta_pos = next_tgt_pos - current_gripper_pos   # [num_envs, 3]
            # delta_pos[:,2]=0
            print("Action Delta Pose: ",delta_pos[5])
            # delta_z=delta_pos[:,2].clone()
            # delta_pos[:,2]=0
            # delta_z[:] = -0.0008
            # delta_norm = torch.norm(delta_pos, dim=1, keepdim=True) + 1e-8  # avoid div by 0
            # target_length = 0.002 * (3 ** 0.5)
            # delta_pos = delta_pos / delta_norm * target_length
            # choose: raw when very small, normalized otherwise
            # delta_pos = torch.where(mask_raw, delta_pos, delta_pos_normalized)
            # delta_pos[:,2] = -0.0002
            # print(delta_z)
            next_tgt_pos=current_gripper_pos + delta_pos
            
            next_tgt_quat=envs.fingertip_centered_quat.clone()
            current_pos = envs.fingertip_centered_pos.clone()
            envs._move_gripper_to_eef_pose(env_ids, 
                                                ctrl_tgt_pos=next_tgt_pos, 
                                                ctrl_tgt_quat=next_tgt_quat, 
                                                sim_steps=10, 
                                                if_log=True, 
                                                close_gripper=True,
                                                log_freq=10,
                                                log_extra=False
                                                )
            # for inspect_id in [5]:
            #         print(inspect_id)
            #         print("Current Pose: ",current_pos[inspect_id])
            #         print("Target Pose: ", next_tgt_pos[inspect_id])
            #         print("Target Quat: ", next_tgt_quat[inspect_id])
            #         print("Reach Pose: ", envs.fingertip_centered_pos[inspect_id])
            #         print("Reach Quat: ", envs.fingertip_centered_quat[inspect_id])
        quat_success = envs.plug_quat[envs.success_env_ids]
        w = quat_success[:, 3]

        vertical_mask = w > 0.85
        valid_ids = envs.success_env_ids[vertical_mask.cpu()]
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
        for env_id in range(12):
            envs.save_first_env_images(out_dir="rollout", index=env_id, reverse=False)

        # print("Final (XY dist, Z dist) for successful vertical plugs:", dist_list)
        dists.extend(dist_list)
        # os._exit(0)
        np.save(f"eval_adapt/{envs.cfg_task.env.desired_subassemblies[0]}.npy", np.array(dists))
        #cfg_task.env.desired_subassemblies[0]
    # Visualize
    # envs._save_log_traj()
    # import pdb;pdb.set_trace()
    
    os._exit(0)


if __name__ == "__main__":
    run_env()