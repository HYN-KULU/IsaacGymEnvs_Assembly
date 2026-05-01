import os
from moviepy.editor import ImageSequenceClip
import numpy as np

# Configuration
input_dir = "rotated_socket_image"
output_file = "socket_rotation.mp4"
target_duration = 6
num_steps = 350  # Based on your socket_top_down_0 to 31

def create_video_from_folder():
    # 1. Gather file paths in the correct order
    frames = []
    for step in range(num_steps):
        img_path = os.path.join(input_dir, f"socket_top_down_{step}.png")
        if os.path.exists(img_path):
            frames.append(img_path)
    
    if not frames:
        print("No images found!")
        return

    # 2. Calculate FPS to hit exactly 6 seconds
    # fps = total_frames / duration
    calculated_fps = len(frames) / target_duration

    print(f"Creating video from {len(frames)} frames at {calculated_fps:.2f} FPS...")

    # 3. Create the clip
    clip = ImageSequenceClip(frames, fps=calculated_fps)

    # 4. Write the file
    # We use libx264 for compatibility and specify a standard logger
    clip.write_videofile(output_file, codec="libx264", audio=False)

    print(f"Done! Saved to {output_file}")

if __name__ == "__main__":
    create_video_from_folder()