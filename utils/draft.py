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
        ["python", "preprocess_data_depth_multitask_forward.py", "--task_id", task_id],
        check=True
    )
    print(f"[DONE] task {task_id}")

NUM_PROCESSES = 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="", required = False)
    args=parser.parse_args()
    train_tasks = [
        "10001","10002","10003","10004","10005","10006","10007","10008","10009",
        "10011","10012","10013","10014","10015","10016","10017","10019","10020","10021",
        "10022","10023","10024","10025","10027","10028","10030","10031","10033","10034",
        "10035","10036","10038","10039","10040","10041","10042","10043","10044","10045",
        "10046","10047","10048","10049","00004","00016","00021","00030","00042",
        "00110","00138","00163","00175","00187","00192","00211","00213","00256","00271",
        "00293","00318","00319","00320","00329","00346","00388","00410","00446","00480",
        "00499","00514","00537","00559","00614","00638","00659","00681","00686","00783",
        "00860","01041","01079","01092","01102","01129","01132","01136"
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