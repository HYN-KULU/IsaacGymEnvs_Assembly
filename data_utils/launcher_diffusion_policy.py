import os
import subprocess
from multiprocessing import Pool, cpu_count
import argparse
# =====================
# CONFIG
# =====================


def run_task(task_id):
    print(f"[START] task {task_id}")
    subprocess.run(
        ["python", "preprocess_data_depth_multitask_forward_image_condition.py", "--task_id", task_id],
        check=True
    )
    print(f"[DONE] task {task_id}")

NUM_PROCESSES = 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="", required = False)
    args=parser.parse_args()
    train_tasks = [
        "20031"
    ]
    #train_tasks = ["00004"]
    print(f"[INFO] Training tasks: {len(train_tasks)}")
    print(f"[INFO] Training task IDs: {train_tasks}")

    nproc = min(NUM_PROCESSES, cpu_count())
    print(f"[INFO] Using {nproc} processes")
    if args.task != "":
        train_tasks = [f"{args.task}"]
    with Pool(processes=nproc) as pool:
        pool.map(run_task, train_tasks)