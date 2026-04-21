import os
import numpy as np

ROOT_DIR = "/tmp/data/diffusion_policy"
OUT_INDEX = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/diffusion_policy/diffusion_policy_index_0421.npy"

index = []

train_tasks = [
        "20031"
    ]
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="", required = False)
args = parser.parse_args()
if args.task !="":
    train_tasks = [f"{args.task}"]
for task in train_tasks:
    ids=[]
    task_dir = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/diffusion_policy_image_condition/asset_{task}"
    for fname in sorted(os.listdir(task_dir),key=lambda x: int(x.replace("depth_", "").replace(".npz", ""))):
        if not fname.startswith("depth_") or not fname.endswith(".npz"):
            continue

        flow_mask_id = int(
            fname.replace("depth_", "").replace(".npz", "")
        )
        ids.append(flow_mask_id)
    max_id = max(ids)
    for i in range(max_id + 1):
        index.append((task,i))
    print(task, len(ids), max_id)
print(f"[INFO] Collected {len(index)} samples")

# convert to numpy array
index = np.asarray(index, dtype=np.int32)

os.makedirs(os.path.dirname(OUT_INDEX), exist_ok=True)
np.save(OUT_INDEX, index)

print(f"[OK] Saved index → {OUT_INDEX}")
print("Index shape:", index.shape)
print("First entry:", index[0])