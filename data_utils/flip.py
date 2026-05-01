from PIL import Image

# input and output paths
input_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/data_utils/task_20031_traj_0_orig_plug_depth.png"
output_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/data_utils/task_20031_traj_0_orig_plug_depth_flipped.png"

# open image
img = Image.open(input_path)

# flip vertically (upside down)
flipped = img.transpose(Image.FLIP_TOP_BOTTOM)

# save result
flipped.save(output_path)

print("Saved flipped image to:", output_path)