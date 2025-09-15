from isaacgym import gymapi, gymtorch
import numpy as np
import imageio
import os
import torch
import sys

# --- Initialization ---
try:
    gym = gymapi.acquire_gym()
except Exception as e:
    print(f"Error acquiring gym: {e}")
    sys.exit()

# --- Simulation Parameters ---
sim_params = gymapi.SimParams()
sim_params.use_gpu_pipeline = True
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

# PhysX params
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.physx.bounce_threshold_velocity = 0.2
sim_params.physx.max_gpu_contact_pairs = 8 * 1024
sim_params.physx.num_threads = 4
sim_params.physx.use_gpu = True
sim_params.physx.max_depenetration_velocity = 100.0

# --- Up axis & gravity ---
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# --- Device IDs ---
sim_device_id = 0
graphics_device_id = 0

# --- Create Simulation ---
sim = gym.create_sim(sim_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("Failed to create sim instance.")
    sys.exit()

# --- Environment ---
spacing = 2.0
env = gym.create_env(sim, gymapi.Vec3(-spacing, -spacing, 0.0), gymapi.Vec3(spacing, spacing, spacing), 1)

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0.0
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0.0
gym.add_ground(sim, plane_params)

# Add cube actor
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, 0.5, 0.5, 0.5, asset_options)
box_pose = gymapi.Transform()
box_pose.p = gymapi.Vec3(0, 0, 0.5)
gym.create_actor(env, box_asset, box_pose, "cube", 0, 1)

# --- Camera ---
cam_props = gymapi.CameraProperties()
cam_props.width = 640
cam_props.height = 480
cam_props.enable_tensors = True   # critical

camera_handle = gym.create_camera_sensor(env, cam_props)
camera_location = gymapi.Vec3(2, 2, 2)
camera_target = gymapi.Vec3(0, 0, 0.5)
gym.set_camera_location(camera_handle, env, camera_location, camera_target)

# Prepare sim
gym.prepare_sim(sim)

# --- Acquire GPU tensor once ---
color_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_COLOR)
torch_color_tensor = gymtorch.wrap_tensor(color_tensor)

# --- Output dir ---
out_dir = "output_images"
os.makedirs(out_dir, exist_ok=True)

# --- Simulation loop ---
for frame in range(30):  # capture 30 frames
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)

    gym.start_access_image_tensors(sim)

    # Tensor is updated in-place each frame
    img_gpu = torch_color_tensor.clone()  # clone if you want to keep frame
    img = img_gpu.cpu().numpy().reshape(cam_props.height, cam_props.width, 4)
    img = img[:, :, :3]  # drop alpha

    # Save to disk
    fname = os.path.join(out_dir, f"frame_{frame:04d}.png")
    imageio.imwrite(fname, img)

    gym.end_access_image_tensors(sim)

print(f"Saved images to {out_dir}/")

