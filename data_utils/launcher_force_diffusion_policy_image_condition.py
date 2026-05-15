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
        # "40001", "40002", "40003", "40004", "40005", "40006", "40007", "40008", "40009", "40010", "40011",
        # "40012", "40013", "40014", "40015", "40016", "40017", "40019", "40020", "40021", "40022", "40023",
        # "40024", "40025", "40026", "40027", "40028", "40029", "40030", "40031", "40032", "40033", "40034",
        # "40035", "40036", "40037", "40038", "40039", "40040", "40041", "40042", "40043", "40044", "40045",
        # "40046", "40047", "40048", "40049", "40050", "40051", "40052", "40053", "40054", "40055", "40056",
        # "40057", "40058", "40059", "40060", "40061", "40062", "40063", "40064", "40065", "40066", "40067",
        # "40068", "40069", "40070", "40071", "40072", "40073", "40074", "40075", "40076", "40077", "40078",
        # "40079", "40080", "40081", "40082", "40083", "40084", "40085", "40086", "40087", "40088", "40089",
        # "40091", "40092", "40093", "40094", "40095", "40097", "40100", "40102", "40103", "40104", "40105",
        # "40107", "40108", "40109", "40110", "40112", "40113", "40118", "40119", "40120", "40121", "40122",
        # "40123", "40124", "40125", "40126", "40127", "40128", "40130", "40134", "40135", "40136", "40137",
        # "40140", "40141", "40142", "40143", "40144", "40145", "40146", "40147", "40148", "40149", "40150",
        # "40151", "40152", "40153", "40156", "40157", "40158", "40159", "40162", "40167", "40168", "40169",
        # "40172", "40173", "40174", "40175", "40176", "40179", "40180", "40182", "40183", "40185", "40186",
        # "40187", "40188", "40189", "40192", "40195", "40196", "40197", "40198", "40199",
        # "20031"
        "40069"
    ]
    print(f"[INFO] Training tasks: {len(train_tasks)}")
    print(f"[INFO] Training task IDs: {train_tasks}")

    nproc = min(NUM_PROCESSES, cpu_count())
    print(f"[INFO] Using {nproc} processes")
    jobs = []
    for task_id in train_tasks:
        for hdf_id in range(2):   # 0 ~ 5
            jobs.append((task_id, hdf_id))
    if args.task != "":
        train_tasks = [f"{args.task}"]
    with Pool(processes=nproc) as pool:
        pool.starmap(run_task, jobs)