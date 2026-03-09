import os
import numpy as np

ROOT_DIR = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/flow_net"
OUT_INDEX = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/flow_diffusion_policy_adapt_index_00015_flow.npy"

index = []
task_num=0
# iterate over task directories
for d in sorted(os.listdir(ROOT_DIR)):
    if not d.startswith("asset_"):
        continue
    task_num += 1
    # if task_num > 6:
    #     break
    task_id = int(d.replace("asset_", ""))   # store as int
    print(task_id)
    if task_id != 15:
        continue
    task_dir = os.path.join(ROOT_DIR, d)

    for fname in sorted(os.listdir(task_dir)):
        if not fname.startswith("flow_mask_") or not fname.endswith(".npz"):
            continue

        flow_mask_id = int(
            fname.replace("flow_mask_", "").replace(".npz", "")
        )
        # if len(index) > 1000:
        #     break
        # store only integers, not strings
        index.append((task_id, flow_mask_id))

print(f"[INFO] Collected {len(index)} samples")

# convert to numpy array
index = np.asarray(index, dtype=np.int32)

os.makedirs(os.path.dirname(OUT_INDEX), exist_ok=True)
np.save(OUT_INDEX, index)

print(f"[OK] Saved index → {OUT_INDEX}")
print("Index shape:", index.shape)
print("First entry:", index[0])