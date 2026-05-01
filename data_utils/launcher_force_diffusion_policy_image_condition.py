import os
import subprocess
from multiprocessing import Pool, cpu_count
import argparse
# =====================
# CONFIG
# =====================


def run_task(task_id, hdf_id):
    print(f"[START] task {task_id}, hdf_id {hdf_id}")
    subprocess.run(
        ["python", "/home/ubuntu/automate/IsaacGymEnvs_Assembly/data_utils/preprocess_data_depth_multitask_forward_image_condition.py", "--task_id", task_id, "--hdf_id", str(hdf_id)],
        check=True
    )
    print(f"[DONE] task {task_id}, hdf_id {hdf_id}")

NUM_PROCESSES = 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="", required = False)
    args=parser.parse_args()
    train_tasks = [
        "20031"
    ]
    print(f"[INFO] Training tasks: {len(train_tasks)}")
    print(f"[INFO] Training task IDs: {train_tasks}")

    nproc = min(NUM_PROCESSES, cpu_count())
    print(f"[INFO] Using {nproc} processes")
    jobs = []
    for task_id in train_tasks:
        for hdf_id in range(12):   # 0 ~ 11
            jobs.append((task_id, hdf_id))
    if args.task != "":
        train_tasks = [f"{args.task}"]
    with Pool(processes=nproc) as pool:
        pool.starmap(run_task, jobs)