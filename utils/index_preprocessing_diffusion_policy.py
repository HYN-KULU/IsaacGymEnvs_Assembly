import os
import numpy as np

ROOT_DIR = "/tmp/data/diffusion_policy"
OUT_INDEX = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/diffusion_policy/diffusion_policy_index.npy"

index = []

train_tasks = [
        # "10001", "10002","10003","10004","10005","10006","10007","10008","10009",
        # "10011","10012","10013","10014","10015","10016","10017","10019","10020","10021",
        # "10022","10023","10024","10025","10027","10028","10030","10031","10033","10034",
        # "10035","10036","10038","10039","10040","10041","10042","10043","10044","10045",
        # "10046","10047","10048","10049","00004","00016","00021","00030","00042",
        # "00110","00138","00163","00175","00187","00192","00211","00213","00256","00271",
        # "00293","00318","00319","00320","00329","00346","00388","00410","00446","00480",
        # "00499","00514","00537","00559","00614","00638","00659","00681","00686","00783",
        # "00860","01041","01079","01092","01102","01129","01132","01136"
        "00103"
    ]
#train_tasks = ["00004"]
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="", required = False)
args = parser.parse_args()
if args.task !="":
    train_tasks = [f"{args.task}"]
for task in train_tasks:
    ids=[]
    task_dir = f"/home/ubuntu/automate/IsaacGymEnvs_Assembly/data/diffusion_policy_multitask_0322/asset_{task}"
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
        if len(index) > 1394: 
            break
    print(task, len(ids), max_id)
print(f"[INFO] Collected {len(index)} samples")

# convert to numpy array
index = np.asarray(index, dtype=np.int32)

os.makedirs(os.path.dirname(OUT_INDEX), exist_ok=True)
np.save(OUT_INDEX, index)

print(f"[OK] Saved index → {OUT_INDEX}")
print("Index shape:", index.shape)
print("First entry:", index[0])