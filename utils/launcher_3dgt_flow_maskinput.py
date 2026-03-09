import os
import subprocess
from multiprocessing import Pool, cpu_count
import argparse
# =====================
# CONFIG
# =====================
TASK_ROOT = "/tmp/data/ego_flow_data_0130"
NUM_PROCESSES = 8

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


def run_task(task_id):
    print(f"[START] task {task_id}")
    subprocess.run(
        ["python", "preprocess_3dgt_mask_flow.py", "--task_id", task_id],
        check=True
    )
    print(f"[DONE] task {task_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="", required = False)
    args=parser.parse_args()

    # 🔑 FILTER HERE
    #train_tasks = [t for t in all_tasks if t not in EXCLUDE_TASKS]
    # train_tasks=[
                # '00004', '00016', '00021', '00030', '00074', '00078', '00110', '00117', '00133', '00138', '00141', '00163', '00175',
                #  '00186', '00187', '00192', '00211', '00213', '00255', '00256', '00271', '00293', '00301', '00318', '00319', '00320',
                #  '00329', '00346', '00388', '00410', '00417', '00422', '00426', '00437', '00444', '00446', '00471', '00480', '00499' ,
                #  '00514', '00537', '00559', '00615', '00638', '00649', '00659', '00681', '00686', '00700', '00703', '00768', '00783' ,
                #  '00860', '01029', '01041', '01092', '01102', '01132', '01136'
                #  ]
    # train_tasks = ['00004', '00016','00021', '00030', '00074', '00078']
    train_tasks = ['00015']
    # train_tasks = ['00345', '00360', '00028', '00614', '00103', '00553', '00731', '00015', '00648', '00506']
    # print(f"[INFO] Total tasks found: {len(all_tasks)}")
    # print(f"[INFO] Excluded tasks: {len(EXCLUDE_TASKS)}")
    # print(f"[INFO] Training tasks: {len(train_tasks)}")
    # print(f"[INFO] Training task IDs: {train_tasks}")

    nproc = min(NUM_PROCESSES, cpu_count())
    print(f"[INFO] Using {nproc} processes")
    #train_tasks = [f"{args.task}"]
    #train_tasks = ["00021","00028", "00030", "00042", "00110", "00681"]
    with Pool(processes=nproc) as pool:
        pool.map(run_task, train_tasks)