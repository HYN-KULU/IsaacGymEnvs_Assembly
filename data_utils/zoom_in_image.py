from PIL import Image

img_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/data_utils/task_20031_traj_0_orig_plug_depth.png"
save_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/data_utils/task_20031_traj_0_orig_plug_depth_zoom_center_flip.png"

img = Image.open(img_path)

zoom = 2.7  # larger = more zoom in

w, h = img.size
new_w = int(w / zoom)
new_h = int(h / zoom)

left = (w - new_w) // 2
top = (h - new_h) // 2
right = left + new_w
bottom = top + new_h

cropped = img.crop((left, top, right, bottom))

# resize back to original image size
zoomed = cropped.resize((w, h), Image.NEAREST)  # good for depth / mask images

# flip upside down
zoomed_flipped = zoomed.transpose(Image.FLIP_TOP_BOTTOM)

zoomed_flipped.save(save_path)
print(f"Saved zoomed and flipped image to {save_path}")