import os

# =====================
# CONFIG
# =====================
TASK_ROOT = "/home/ubuntu/automate/alltracker/flow_mask_data_1206"

ADAPTATION_TASKS = {
    "00081", "00360", "00028", "00597", "00103",
    "00553", "00731", "00015", "00648", "00506",
}

NOT_USED_TASKS = {
    "00007", "00032", "00083", "00143", "00190",
    "00308", "00340", "00470", "00486", "00755",
    "00863",
    "00014",
    "00210",
    "00062",
    "00652",
    "00831",
    "01053",
    "01125",
}

EXCLUDE_TASKS = ADAPTATION_TASKS | NOT_USED_TASKS

# =====================
# GET TASK IDS
# =====================
all_tasks = sorted([
    d for d in os.listdir(TASK_ROOT)
    if os.path.isdir(os.path.join(TASK_ROOT, d))
])

train_tasks = [t for t in all_tasks if t not in EXCLUDE_TASKS]

print(f"[INFO] Total tasks found: {len(all_tasks)}")
print(f"[INFO] Excluded tasks: {len(EXCLUDE_TASKS)}")
print(f"[INFO] Training tasks: {len(train_tasks)}")
print(f"[INFO] Training task IDs: {train_tasks}")
