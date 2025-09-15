# Copyright (c) 2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""AutoMate: class for disassembly path collection.

Inherits AutoMate environment class and Factory abstract task class (not enforced).

Collect disassembly paths for given assets (no RL training).

Can be executed with python train.py task=AutoMateTaskDisassemble.
"""


import hydra
import numpy as np
import omegaconf
import os
import torch
import warp as wp
import json
import h5py
import imageio
from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import (
    FactorySchemaConfigTask,
)
from isaacgymenvs.tasks.automate.automate_env import AutoMateEnv
from isaacgymenvs.utils import torch_jit_utils
import matplotlib.pyplot as plt
import isaacgymenvs.tasks.factory.factory_control as fc
from concurrent.futures import ThreadPoolExecutor
import math 

def quat_to_matrix(q):
    """Convert Isaac Gym gymapi.Quat to 3x3 rotation matrix"""
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ])
    return R

class AutoMateTaskDisassemble(AutoMateEnv, FactoryABCTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        """Initialize instance variables. Initialize task superclass."""

        self.cfg = cfg
        self._get_task_yaml_params()
        print("Calling This in Automate Disassembly ***************************************")
        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        # load plug grasp poses and disassembly distances 
        self.plug_grasps, self.disassembly_dists = self._load_assembly_info()

        # initialized logging variables for disassembly paths
        self._init_log_data_per_assembly()

        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.viewer != None:
            self._set_viewer_params()

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        # load task configuration
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_task", node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = (
            self.cfg_task.rl.max_episode_length
        )  # required instance var for VecTask

        # load ppo configuration, just as placeholder 
        # since no RL training included in disassembly path collection task
        ppo_path = os.path.join(
            "train/AutoMateTaskDisassemblePPO.yaml"
        )  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

    def create_envs(self):
        # First create the normal environments (Franka, plug, socket, etc.)
        super().create_envs()
        # Then add cameras to each env_ptr
        self.add_cameras()

    def add_cameras(self):
        self.camera_handles = []
        # Shared base properties
        cam_props = gymapi.CameraProperties()
        cam_props.width = 640   # your new resolution
        cam_props.height = 480
        cam_props.enable_tensors = True
        for env_ptr in self.env_ptrs:
            env_cameras = {}
            theta_deg = 120   # example: tilt downward 45°
            theta_rad = math.radians(theta_deg) 
            offset = gymapi.Transform(
                p=gymapi.Vec3(0.00, 0.0, 0.0),
                r=gymapi.Quat.from_euler_zyx(0, -theta_rad, 0)  # tilt downward by theta_deg
            )
            # offset = gymapi.Transform(
            #     p=gymapi.Vec3(0.00, 0.0, 0),  # 5cm above the link
            #     r=gymapi.Quat.from_euler_zyx(0, -1.57, 0)  # tilt downward 90°
            # )
            cam_handle_panda=self.gym.create_camera_sensor(env_ptr,cam_props)
            # self.gym.attach_camera_to_body(cam_handle_panda,env_ptr,self.panda_camera_id,gymapi.Transform(),gymapi.FOLLOW_TRANSFORM)
            self.gym.attach_camera_to_body(cam_handle_panda,env_ptr,self.panda_camera_id,offset,gymapi.FOLLOW_TRANSFORM)
            env_cameras["panda"]=cam_handle_panda
            # self.gym.prepare_sim(self.sim)
            # self.gym.simulate(self.sim)
            # self.gym.fetch_results(self.sim,True)
            # self.gym.step_graphics(self.sim)
            # self.gym.render_all_camera_sensors(self.sim)
            # color_image = self.gym.get_camera_image(self.sim, env_ptr, cam_handle_panda, gymapi.IMAGE_COLOR).reshape(480,640,4)[:,:,3]
            # imageio.imwrite("panda_camera.png", color_image)
            # import pdb;pdb.set_trace()
    
            # --- Top view (slightly shifted, higher)
            cam_handle_top = self.gym.create_camera_sensor(env_ptr, cam_props)
            self.gym.set_camera_location(
                cam_handle_top, env_ptr,
                gymapi.Vec3(0.54, 0.48, 0.7),  # your top cam pos
                gymapi.Vec3(0, 0, 0.7)                                 # look at target
            )
            env_cameras["top"] = cam_handle_top
            # self.gym.prepare_sim(self.sim)
            # self.gym.simulate(self.sim)
            # self.gym.fetch_results(self.sim,True)
            # self.gym.step_graphics(self.sim)
            # self.gym.render_all_camera_sensors(self.sim)
            # color_image = self.gym.get_camera_image(self.sim, env_ptr, cam_handle_top, gymapi.IMAGE_COLOR).reshape(480,640,4)[:,:,3]
            # imageio.imwrite("panda_camera.png", color_image)
            # import pdb;pdb.set_trace()
            # --- Bottom view (angled from below)
            cam_handle_bottom = self.gym.create_camera_sensor(env_ptr, cam_props)
            self.gym.set_camera_location(
                cam_handle_bottom, env_ptr,
                gymapi.Vec3(0.5 * 0.6, -0.5 * 0.6, 0.7),  # your bottom cam pos
                gymapi.Vec3(0, 0, 0.5)                    # look upward-ish
            )
            env_cameras["bottom"] = cam_handle_bottom
            # import pdb;pdb.set_trace()
            # cam_pose_top=self.gym.get_camera_transform(self.sim, env_ptr, cam_handle_top)
            # cam_pose_bottom=self.gym.get_camera_transform(self.sim, env_ptr, cam_handle_bottom)
            # cam_pose_top_R=quat_to_matrix(cam_pose_top.r)
            # cam_pose_bottom_R=quat_to_matrix(cam_pose_bottom.r)
            # # Save both for this env
            self.camera_handles.append(env_cameras)

    def _load_assembly_info(self):
        """Load grasp pose and disassembly distance for plugs in each environment."""

        plug_grasps, disassembly_dists = [], []
        plug_grasp_path = os.path.join(os.getcwd(), self.cfg_task.env.data_dir, self.cfg_task.env.plug_grasp_file)
        if os.path.exists(plug_grasp_path):
            in_file = open(plug_grasp_path, "r")
            plug_grasp_dict = json.load(in_file)
            plug_grasps = [ plug_grasp_dict[self.cfg_env.env.desired_subassemblies[self.asset_indices[i]]] for i in range(self.num_envs)]
        else:
            raise FileNotFoundError(f"{plug_grasp_path} does not exist.")

        disassembly_dist_path = os.path.join(os.getcwd(), self.cfg_task.env.data_dir, self.cfg_task.env.disassembly_dist_file)
        if os.path.exists(disassembly_dist_path):
            in_file = open(disassembly_dist_path, "r")
            disassembly_dist_dict = json.load(in_file)
            disassembly_dists = [ disassembly_dist_dict[self.cfg_env.env.desired_subassemblies[self.asset_indices[i]]] for i in range(self.num_envs)]
        else:
            raise FileNotFoundError(f"{disassembly_dist_path} does not exist.")

        return torch.as_tensor(plug_grasps).to(self.device), torch.as_tensor(disassembly_dists).to(self.device)

    def _acquire_task_tensors(self):
        """Acquire tensors."""
        
        # Grasp pose tensors
        self.palm_to_finger_center = torch.tensor([0.0, 0.0, self.cfg_task.env.palm_to_finger_dist], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.robot_to_gripper_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.plug_grasp_pos_local = self.plug_grasps[:self.num_envs, :3]
        self.plug_grasp_quat_local = torch.roll(self.plug_grasps[:self.num_envs, 3:], -1, 1)

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        
        self.plug_grasp_quat, self.plug_grasp_pos = torch_jit_utils.tf_combine(self.plug_quat,
                                                                               self.plug_pos,
                                                                               self.plug_grasp_quat_local,
                                                                               self.plug_grasp_pos_local)

        self.plug_grasp_quat, self.plug_grasp_pos = torch_jit_utils.tf_combine(self.plug_grasp_quat,
                                                                                 self.plug_grasp_pos,
                                                                                 self.robot_to_gripper_quat,
                                                                                 self.palm_to_finger_center)

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self._actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1

        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            self.close_gripper(sim_steps=self.cfg_task.env.close_gripper_sim_steps)
                
            self._disassemble_plug_from_socket()

            if_intersect = (self.plug_pos[:,2] < self.socket_pos[:, 2] + self.disassembly_dists).cpu().numpy()
            success_env_ids = np.argwhere(if_intersect==0).reshape(-1)

            self._log_robot_state(success_env_ids)
            self._log_object_state(success_env_ids)

            self._save_log_traj()

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        return self.obs_buf  # shape = (num_envs, num_observations)

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        self._update_rew_buf()
        self._update_reset_buf()

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        pass

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""
        self.reset_buf[:] = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka(env_ids)
        self.simulate_and_refresh()

        # Temporarily disable gravity to prevent plugs from dropping before being grasped
        self.disable_gravity()

        self._reset_object(env_ids)
        self.simulate_and_refresh()

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self._move_gripper_to_plug_grasp_pose(env_ids, mode='pre_grasp', sim_steps=self.cfg_task.env.move_gripper_sim_steps)

        self._move_gripper_to_plug_grasp_pose(env_ids, mode='grasp', sim_steps=self.cfg_task.env.move_gripper_sim_steps)

        self.close_gripper(sim_steps=self.cfg_task.env.close_gripper_sim_steps)

        self.enable_gravity()

        self._init_log_data_per_episode()

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        # shape of dof_pos = (num_envs, num_dofs)
        # shape of dof_vel = (num_envs, num_dofs)

        # Initialize Franka 
        self.dof_pos[env_ids] = torch.cat(
                (torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos, device=self.device),
                 torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device),
                 torch.tensor([self.asset_info_franka_table.franka_gripper_width_max], device=self.device)),
                dim=-1).unsqueeze(0).repeat((self.num_envs, 1))  # shape = (num_envs, num_dofs)

        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(torch.zeros_like(self.dof_torque)),
                                                gymtorch.unwrap_tensor(self.franka_actor_ids_sim),
                                                len(self.franka_actor_ids_sim))

    def _reset_object(self, env_ids):
        """Reset root state of plug."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        # Randomize socket position 
        self.socket_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        self.socket_noise_xy = self.socket_noise_xy @ torch.diag(torch.tensor(self.cfg_task.randomize.socket_pos_xy_noise, dtype=torch.float32, device=self.device))
        self.root_pos[env_ids, self.socket_actor_id_env, 0] = self.robot_base_pos[env_ids, 0] + self.cfg_task.randomize.socket_pos_xy_initial[0] + self.socket_noise_xy[env_ids, 0]
        self.root_pos[env_ids, self.socket_actor_id_env, 1] = self.robot_base_pos[env_ids, 1] + self.cfg_task.randomize.socket_pos_xy_initial[1] + self.socket_noise_xy[env_ids, 1]
        self.root_pos[env_ids, self.socket_actor_id_env, 2] = self.cfg_base.env.table_height
        
        # Load plug in assembled state (i.e., move to socket position)
        self.root_pos[env_ids, self.plug_actor_id_env, :] = self.root_pos[env_ids, self.socket_actor_id_env, :]

        # Set plug and socket orientation to be upright
        self.root_quat[env_ids, self.plug_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.root_quat[env_ids, self.socket_actor_id_env] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        plug_socket_actor_ids_sim = torch.cat((self.plug_actor_ids_sim[env_ids], self.socket_actor_ids_sim[env_ids]), dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(plug_socket_actor_ids_sim),
                                                     len(plug_socket_actor_ids_sim))

    def _reset_buffers(self, env_ids):
        """Reset buffers. """

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def quat_slerp(self,q0, q1, alpha):
        """
        Spherical linear interpolation (slerp) between two quaternions q0 and q1.
        q0, q1: [N, 4] (w, x, y, z) format
        alpha: float in [0,1] or tensor broadcastable to [N, 1]
        """
        # Normalize to be safe
        q0 = q0 / q0.norm(dim=-1, keepdim=True)
        q1 = q1 / q1.norm(dim=-1, keepdim=True)

        # Compute cosine of angle
        dot = (q0 * q1).sum(-1, keepdim=True)

        # If negative dot, negate one quaternion to take shorter path
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = torch.abs(dot)

        DOT_THRESHOLD = 0.9995
        if torch.all(dot > DOT_THRESHOLD):
            # Nearly identical → use linear interpolation
            result = q0 + alpha * (q1 - q0)
            return result / result.norm(dim=-1, keepdim=True)

        # Compute angle between them
        theta_0 = torch.acos(dot)        # angle between input vectors
        theta = theta_0 * alpha          # angle between q0 and result

        sin_theta_0 = torch.sin(theta_0)
        sin_theta = torch.sin(theta)

        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return (s0 * q0) + (s1 * q1)

    def _move_gripper_to_eef_pose(self, env_ids, ctrl_tgt_pos, ctrl_tgt_quat, sim_steps, if_log, close_gripper):
        """Move end-effector smoothly along a straight-line trajectory to target pose."""

        # ---- Step 1: Record starting pose ----
        start_pos = self.fingertip_centered_pos.clone()
        start_quat = self.fingertip_centered_quat.clone()

        # ---- Step 2: Interpolate trajectory over sim_steps ----
        for t in range(sim_steps):
            alpha = (t + 1) / sim_steps  # goes from 0 → 1

            # Linear interpolation for position
            interp_pos = (1 - alpha) * start_pos + alpha * ctrl_tgt_pos

            # Spherical linear interpolation (slerp) for rotation
            interp_quat = self.quat_slerp(start_quat, ctrl_tgt_quat, alpha)

            # Update control targets
            self.ctrl_target_fingertip_centered_pos[env_ids] = interp_pos[env_ids]
            self.ctrl_target_fingertip_centered_quat[env_ids] = interp_quat[env_ids]

            # ---- Step sim and logging ----
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            if if_log:
                self._log_robot_state_per_timestep()

            # Compute error relative to interpolated pose
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_centered_pos,
                fingertip_midpoint_quat=self.fingertip_centered_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            # Form action from error
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[env_ids, :6] = delta_hand_pose[env_ids]

            # Control gripper open/close
            if close_gripper:
                self._apply_actions_as_ctrl_targets(actions=actions,
                                                    ctrl_target_gripper_dof_pos=0.0,
                                                    do_scale=False)
            else:
                self._apply_actions_as_ctrl_targets(actions=actions,
                                                    ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                                                    do_scale=False)

            # Step simulation
            self.gym.simulate(self.sim)
            # if close_gripper:
            #     self._apply_actions_as_ctrl_targets(actions=actions,
            #                                         ctrl_target_gripper_dof_pos=0.0,
            #                                         do_scale=False)
            # else:
            #     self._apply_actions_as_ctrl_targets(actions=actions,
            #                                         ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
            #                                         do_scale=False)

            # self.gym.simulate(self.sim)
            if if_log:
                self.gym.fetch_results(self.sim,False)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)

                def fetch_image(env_id, cam_handle, cam_type, height, width):
                    img = self.gym.get_camera_image(self.sim, self.env_ptrs[env_id], cam_handle, cam_type)
                    if cam_type == gymapi.IMAGE_COLOR:
                        return img.reshape(height, width, 4)[:, :, :3]  # RGB
                    elif cam_type == gymapi.IMAGE_DEPTH:
                        img = img.reshape(height, width)
                        return np.where(np.isinf(img), np.nan, img)

                # Storage lists
                camera1_rgb_list = []
                camera2_rgb_list = []
                camera1_depth_list = []
                camera2_depth_list = []
                camera3_rgb_list=[]
                camera3_depth_list=[]
                height, width = 480, 640

                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = {
                        ("top", "color", env_id): executor.submit(fetch_image, env_id, self.camera_handles[env_id]["top"], gymapi.IMAGE_COLOR, height, width)
                        for env_id in range(len(self.env_ptrs))
                    }
                    futures.update({
                        ("top", "depth", env_id): executor.submit(fetch_image, env_id, self.camera_handles[env_id]["top"], gymapi.IMAGE_DEPTH, height, width)
                        for env_id in range(len(self.env_ptrs))
                    })
                    futures.update({
                        ("bottom", "color", env_id): executor.submit(fetch_image, env_id, self.camera_handles[env_id]["bottom"], gymapi.IMAGE_COLOR, height, width)
                        for env_id in range(len(self.env_ptrs))
                    })
                    futures.update({
                        ("bottom", "depth", env_id): executor.submit(fetch_image, env_id, self.camera_handles[env_id]["bottom"], gymapi.IMAGE_DEPTH, height, width)
                        for env_id in range(len(self.env_ptrs))
                    })
                    futures.update({
                        ("panda", "color", env_id): executor.submit(fetch_image, env_id, self.camera_handles[env_id]["panda"], gymapi.IMAGE_COLOR, height, width)
                        for env_id in range(len(self.env_ptrs))
                    })
                    futures.update({
                        ("panda", "depth", env_id): executor.submit(fetch_image, env_id, self.camera_handles[env_id]["panda"], gymapi.IMAGE_DEPTH, height, width)
                        for env_id in range(len(self.env_ptrs))
                    })

                # Collect results in order of env_id
                for env_id in range(len(self.env_ptrs)):
                    camera1_rgb_list.append(futures[("top", "color", env_id)].result())
                    camera1_depth_list.append(futures[("top", "depth", env_id)].result())
                    camera2_rgb_list.append(futures[("bottom", "color", env_id)].result())
                    camera2_depth_list.append(futures[("bottom", "depth", env_id)].result())
                    camera3_rgb_list.append(futures[("panda", "color", env_id)].result())
                    camera3_depth_list.append(futures[("panda", "depth", env_id)].result())

                # Convert to arrays if desired
                self.camera1_rgb_traj.append(np.stack(camera1_rgb_list)  )  # (envs, H, W, 3)
                self.camera2_rgb_traj.append(np.stack(camera2_rgb_list))   # (envs, H, W, 3)
                self.camera3_rgb_traj.append(np.stack(camera3_rgb_list))   # (envs, H, W, 3)
                self.camera1_depth_traj.append(np.stack(camera1_depth_list))   # (envs, H, W)
                self.camera2_depth_traj.append(np.stack(camera2_depth_list))   # (envs, H, W)
                self.camera3_depth_traj.append(np.stack(camera3_depth_list))   # (envs, H, W)
                self.gym.end_access_image_tensors(self.sim)
                # imageio.imwrite("camera_top.png", self.camera1_rgb_traj[0][0].astype(np.uint8))
                # import pdb;pdb.set_trace()
                print(f"Collect Image {len(self.camera1_depth_traj)}")


        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(torch.zeros_like(self.dof_torque)),
                                                gymtorch.unwrap_tensor(self.franka_actor_ids_sim),
                                                len(self.franka_actor_ids_sim))

    def _move_gripper_to_plug_grasp_pose(self, env_ids, mode, sim_steps):
        """Move end-effector to plug grasp pose."""

        ctrl_tgt_pos = torch.empty_like(self.plug_grasp_pos).copy_(self.plug_grasp_pos)

        if mode=='grasp':
            ctrl_tgt_quat = torch.empty_like(self.plug_grasp_quat).copy_(self.plug_grasp_quat)

        elif mode=='pre_grasp':
            ctrl_tgt_pos[:, 2] += self.cfg_task.env.plug_pregrasp_offset
            ctrl_tgt_quat = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self._move_gripper_to_eef_pose(env_ids, ctrl_tgt_pos, ctrl_tgt_quat, sim_steps, if_log=False, close_gripper=False)

    def _disassemble_plug_from_socket(self):
        """Lift plug from socket till disassembly and then randomize end-effector pose."""
        print("Disassembly")
        if_intersect = np.ones(self.num_envs, dtype=np.float32)

        env_ids = np.argwhere(if_intersect==1).reshape(-1)
        self._lift_gripper(self.disassembly_dists * 3.0, self.cfg_task.env.disassemble_sim_steps, env_ids)
        self.simulate_and_refresh()
        if_intersect = (self.plug_pos[:,2] < self.socket_pos[:, 2] + self.disassembly_dists).cpu().numpy()

        env_ids = np.argwhere(if_intersect==0).reshape(-1)
        self._randomize_gripper_pose(env_ids, self.cfg_task.env.move_gripper_sim_steps, if_log=True, close_gripper=True)

    def _lift_gripper(self, lift_distance, sim_steps, env_ids=None):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        ctrl_tgt_pos = torch.empty_like(self.fingertip_centered_pos).copy_(self.fingertip_centered_pos)
        ctrl_tgt_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat((self.num_envs,1))
        ctrl_tgt_pos[:, 2] += lift_distance
        if len(env_ids) == 0:
            env_ids = np.array(range(self.num_envs)).reshape(-1)

        self._move_gripper_to_eef_pose(env_ids, 
                                        ctrl_tgt_pos, 
                                        ctrl_tgt_quat, 
                                        sim_steps, 
                                        if_log=True, 
                                        close_gripper=True)

    # def _randomize_gripper_pose(self, env_ids, sim_steps, if_log, close_gripper):
    #     """Move gripper to random pose."""

    #     ctrl_tgt_pos = torch.empty_like(self.plug_grasp_pos).copy_(self.plug_grasp_pos)
    #     ctrl_tgt_pos[:, 2] += self.cfg_task.randomize.gripper_rand_z_offset

    #     fingertip_centered_pos_noise = \
    #         2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
    #     fingertip_centered_pos_noise = \
    #         fingertip_centered_pos_noise @ torch.diag(torch.tensor(self.cfg_task.randomize.gripper_rand_pos_noise,
    #                                                                device=self.device))
    #     ctrl_tgt_pos += fingertip_centered_pos_noise

    #     # Set target rot
    #     ctrl_target_fingertip_centered_euler = torch.tensor(self.cfg_task.randomize.fingertip_centered_rot_initial,
    #                                                         device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

    #     fingertip_centered_rot_noise = \
    #         2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
    #     fingertip_centered_rot_noise = fingertip_centered_rot_noise @ torch.diag(
    #         torch.tensor(self.cfg_task.randomize.gripper_rand_rot_noise, device=self.device))
    #     ctrl_target_fingertip_centered_euler += fingertip_centered_rot_noise
    #     ctrl_tgt_quat = torch_utils.quat_from_euler_xyz(
    #         ctrl_target_fingertip_centered_euler[:, 0],
    #         ctrl_target_fingertip_centered_euler[:, 1],
    #         ctrl_target_fingertip_centered_euler[:, 2])

    #     self._move_gripper_to_eef_pose(env_ids, ctrl_tgt_pos, ctrl_tgt_quat, sim_steps, if_log, close_gripper)

    def _randomize_gripper_pose(self, env_ids, sim_steps, if_log, close_gripper):
        """Move gripper to a random target pose with smooth motion (no per-step noise)."""

        # ---- Step 1: Randomize target position ----
        ctrl_tgt_pos = torch.empty_like(self.plug_grasp_pos).copy_(self.plug_grasp_pos)
        ctrl_tgt_pos[:, 2] += self.cfg_task.randomize.gripper_rand_z_offset
        # Sample random offset ONCE (no per-step jitter)
        rand_pos_offset = (2 * torch.rand((self.num_envs, 3), device=self.device) - 1.0)
        rand_pos_offset = rand_pos_offset @ torch.diag(
            torch.tensor(self.cfg_task.randomize.gripper_rand_pos_noise, device=self.device)
        )
        ctrl_tgt_pos += rand_pos_offset

        # ---- Step 2: Randomize target rotation ----
        base_euler = torch.tensor(
            self.cfg_task.randomize.fingertip_centered_rot_initial, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        rand_rot_offset = (2 * torch.rand((self.num_envs, 3), device=self.device) - 1.0)
        rand_rot_offset = rand_rot_offset @ torch.diag(
            torch.tensor(self.cfg_task.randomize.gripper_rand_rot_noise, device=self.device)
        )
        ctrl_target_euler = base_euler + rand_rot_offset
        ctrl_tgt_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_euler[:, 0], ctrl_target_euler[:, 1], ctrl_target_euler[:, 2]
        )

        # ---- Step 3: Move smoothly toward target ----
        self._move_gripper_to_eef_pose(env_ids, ctrl_tgt_pos, ctrl_tgt_quat, sim_steps, if_log, close_gripper)


    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets."""
        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))
        self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_centered_quat)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _init_log_data_per_assembly(self):
        
        self.log_assembly_id = []
        self.log_plug_pos = []
        self.log_plug_quat = []
        self.log_init_plug_pos = []
        self.log_init_plug_quat = []
        self.log_plug_grasp_pos = []
        self.log_plug_grasp_quat = []
        self.log_fingertip_centered_pos = []
        self.log_fingertip_centered_quat = []
        self.log_arm_dof_pos = []

    def _init_log_data_per_episode(self):
        
        self.log_fingertip_centered_pos_traj = []
        self.log_fingertip_centered_quat_traj = []
        self.log_arm_dof_pos_traj = []
        self.log_plug_pos_traj = []
        self.log_plug_quat_traj = []
        # Camera 1 rgb
        self.camera1_rgb_traj=[]
        # Camera 1 depth
        self.camera1_depth_traj=[]
        # Camera 2 rgb
        self.camera2_rgb_traj=[]
        # Camera 2 depth
        self.camera2_depth_traj=[]
        # Camera 3 rgb
        self.camera3_rgb_traj=[]
        # Camera 3 depth
        self.camera3_depth_traj=[]

        self.init_plug_grasp_pos = self.plug_grasp_pos.clone().detach()
        self.init_plug_grasp_quat = self.plug_grasp_quat.clone().detach()
        self.init_plug_pos = self.plug_pos.clone().detach()
        self.init_plug_quat = self.plug_quat.clone().detach()

    def _log_robot_state(self, env_ids):

        self.log_plug_pos += torch.stack(self.log_plug_pos_traj, dim=1)[env_ids].cpu().tolist()
        self.log_plug_quat += torch.stack(self.log_plug_quat_traj, dim=1)[env_ids].cpu().tolist()
        self.log_arm_dof_pos += torch.stack(self.log_arm_dof_pos_traj, dim=1)[env_ids].cpu().tolist()
        self.log_fingertip_centered_pos += torch.stack(self.log_fingertip_centered_pos_traj, dim=1)[env_ids].cpu().tolist()
        self.log_fingertip_centered_quat += torch.stack(self.log_fingertip_centered_quat_traj, dim=1)[env_ids].cpu().tolist()

    def _log_robot_state_per_timestep(self):

        self.log_plug_pos_traj.append(self.plug_pos.clone().detach()) # env_nums, 3
        self.log_plug_quat_traj.append(self.plug_quat.clone().detach()) # env_nums, 4
        self.log_arm_dof_pos_traj.append(self.arm_dof_pos.clone().detach()) # env_nums, 7
        self.log_fingertip_centered_pos_traj.append(self.fingertip_centered_pos.clone().detach()) # env_nums, 3
        self.log_fingertip_centered_quat_traj.append(self.fingertip_centered_quat.clone().detach()) # env_nums, 4

    def _log_object_state(self, env_ids):
        
        self.log_plug_grasp_pos += self.init_plug_grasp_pos[env_ids].cpu().tolist()
        self.log_plug_grasp_quat += self.init_plug_grasp_quat[env_ids].cpu().tolist()
        self.log_init_plug_pos += self.init_plug_pos[env_ids].cpu().tolist()
        self.log_init_plug_quat += self.init_plug_quat[env_ids].cpu().tolist()


    def save_first_env_images(self, out_dir="saved_visual"):
        """
        Save first environment's RGB images for camera1 and camera2 in separate folders,
        and generate reverse-order videos for each (disassembly -> assembly).
        """
        env_id = 0  # only save the first environment
        num_steps = len(self.camera1_rgb_traj)  # should be ~180

        # Create output dirs
        cam1_dir = os.path.join(out_dir, "camera1")
        cam2_dir = os.path.join(out_dir, "camera2")
        cam3_dir = os.path.join(out_dir, "camera3")
        os.makedirs(cam1_dir, exist_ok=True)
        os.makedirs(cam2_dir, exist_ok=True)
        os.makedirs(cam3_dir, exist_ok=True)

        # Save frames
        for step in range(num_steps):
            img1 = self.camera1_rgb_traj[step][env_id]  # (H, W, 3)
            img2 = self.camera2_rgb_traj[step][env_id]
            img3 = self.camera3_rgb_traj[step][env_id]

            imageio.imwrite(os.path.join(cam1_dir, f"step{step:03d}.png"), img1.astype(np.uint8))
            imageio.imwrite(os.path.join(cam2_dir, f"step{step:03d}.png"), img2.astype(np.uint8))
            imageio.imwrite(os.path.join(cam3_dir, f"step{step:03d}.png"), img3.astype(np.uint8))
            depth1 = self.camera1_depth_traj[step][env_id] 
            depth2 = self.camera2_depth_traj[step][env_id]
            depth3 = self.camera3_depth_traj[step][env_id]
            np.save(os.path.join(cam1_dir, f"step{step:03d}_depth.npy"), depth1)
            np.save(os.path.join(cam2_dir, f"step{step:03d}_depth.npy"), depth2)
            np.save(os.path.join(cam3_dir, f"step{step:03d}_depth.npy"), depth3)

            # imageio.imwrite(os.path.join(cam1_dir, f"step{step:03d}_depth.png"), depth1_mm)
            # imageio.imwrite(os.path.join(cam2_dir, f"step{step:03d}_depth.png"), depth2_mm)
            # imageio.imwrite(os.path.join(cam3_dir, f"step{step:03d}_depth.png"), depth3_mm)

        print(f"Saved {num_steps} frames each for camera1, camera2 and camera3 into {out_dir}/")

        # Generate reverse videos
        cam1_video = os.path.join(out_dir, "camera1_reverse.mp4")
        cam2_video = os.path.join(out_dir, "camera2_reverse.mp4")
        cam3_video = os.path.join(out_dir, "camera3_reverse.mp4")

        fps = 20  # adjust playback speed

        # Reverse order of steps
        reversed_steps = list(range(num_steps - 1, -1, -1))

        with imageio.get_writer(cam1_video, fps=fps) as writer:
            for step in reversed_steps:
                frame = imageio.imread(os.path.join(cam1_dir, f"step{step:03d}.png"))
                writer.append_data(frame)

        with imageio.get_writer(cam2_video, fps=fps) as writer:
            for step in reversed_steps:
                frame = imageio.imread(os.path.join(cam2_dir, f"step{step:03d}.png"))
                writer.append_data(frame)

        with imageio.get_writer(cam3_video, fps=fps) as writer:
            for step in reversed_steps:
                frame = imageio.imread(os.path.join(cam3_dir, f"step{step:03d}.png"))
                writer.append_data(frame)

        print(f"Generated reverse videos: {cam1_video}, {cam2_video}, {cam3_video}")
    
    def _save_log_traj(self):
        if len(self.log_arm_dof_pos) > self.cfg_task.env.num_log_traj:
            log_filename = os.path.join(
                os.getcwd(), 
                self.cfg_task.env.data_dir, 
                self.cfg_task.env.desired_subassemblies[0] + '_disassembly_traj.h5'
            )

            with h5py.File(log_filename, "w") as f:
                # ---- Convert lists to numpy before saving ----
                f.create_dataset("fingertip_centered_pos", data=np.array(self.log_fingertip_centered_pos))
                f.create_dataset("fingertip_centered_quat", data=np.array(self.log_fingertip_centered_quat))
                f.create_dataset("arm_dof_pos", data=np.array(self.log_arm_dof_pos))
                f.create_dataset("plug_grasp_pos", data=np.array(self.log_plug_grasp_pos))
                f.create_dataset("plug_grasp_quat", data=np.array(self.log_plug_grasp_quat))
                f.create_dataset("init_plug_pos", data=np.array(self.log_init_plug_pos))
                f.create_dataset("init_plug_quat", data=np.array(self.log_init_plug_quat))
                f.create_dataset("plug_pos", data=np.array(self.log_plug_pos))
                f.create_dataset("plug_quat", data=np.array(self.log_plug_quat))

                # # ---- RGBD (already numpy arrays) ----
                # f.create_dataset("camera1_rgb", data=np.array(self.camera1_rgb_traj), compression="gzip", compression_opts=4)
                # f.create_dataset("camera2_rgb", data=np.array(self.camera2_rgb_traj), compression="gzip", compression_opts=4)
                # f.create_dataset("camera1_depth", data=np.array(self.camera1_depth_traj), compression="gzip", compression_opts=4)
                # f.create_dataset("camera2_depth", data=np.array(self.camera2_depth_traj), compression="gzip", compression_opts=4)

            print(f"Saved trajectory to {log_filename}")
            self.save_first_env_images()
            exit(0)
        else:
            print("current logging item num: ", len(self.log_arm_dof_pos))
            
        