import os
import subprocess
import zipfile
from multiprocessing import Pool, cpu_count

# =======================
# CONFIG
# =======================
BUCKET_PATH = "gs://cmu-gpucloud-yinongh/flow_1206"

EXTRACT_DIR = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_1206"
ZIP_DIR = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_1206_zips"

NUM_PROCESSES = 16
UNZIP_AFTER_DOWNLOAD = True

os.makedirs(ZIP_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)


def list_gcs_zips():
    """List all zip files in the GCS bucket."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", f"{BUCKET_PATH}/*.zip"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines()]


def download_and_extract(gcs_zip_path: str):
    zip_name = os.path.basename(gcs_zip_path)
    local_zip_path = os.path.join(ZIP_DIR, zip_name)

    asset_name = zip_name.replace(".zip", "")
    extract_path = os.path.join(EXTRACT_DIR, asset_name)

    if os.path.exists(extract_path):
        print(f"⏩ Skip (already extracted): {asset_name}")
        return

    try:
        print(f"⬇️ Downloading {zip_name}")
        subprocess.run(
            ["gcloud", "storage", "cp", gcs_zip_path, local_zip_path],
            check=True,
            capture_output=True,
            text=True,
        )

        if UNZIP_AFTER_DOWNLOAD:
            print(f"📂 Extracting {zip_name}")
            with zipfile.ZipFile(local_zip_path, "r") as zipf:
                zipf.extractall(EXTRACT_DIR)

        print(f"✅ Done {asset_name}")

        # Optional cleanup
        # os.remove(local_zip_path)

    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed {zip_name}:\n{e.stderr}")

    except Exception as e:
        print(f"❌ Error processing {zip_name}: {e}")
ADAPTATION_TASKS = {
    "00081", "00360", "00028", "00597", "00103",
    "00553", "00731", "00015", "00648", "00506",
}

NOT_USED_TASKS = {
    "00007", "00032", "00083", "00143", "00190",
    "00308", "00340", "00470", "00486", "00755",
    "00863", "00296",
    "00014",
    "00210",
    "00062",
    "00652",
    "00831", #
    "01053",
    "01125", #
}

import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="", required = False)
    args=parser.parse_args()
    gcs_zips = list_gcs_zips()
    # gcs_zips = [gcs_zips[0]]
    #print(gcs_zips)
    print("Downloading Trajectory Data")
    # gcs_zips = ["gs://cmu-gpucloud-yinongh/flow_1206/asset_00110.zip","gs://cmu-gpucloud-yinongh/flow_1206/asset_00681.zip"]
    #train_tasks = ['00004']
    gcs_zips = [f'gs://cmu-gpucloud-yinongh/flow_1206/asset_{train_task}.zip' for train_task in train_tasks]
    print(len(gcs_zips))
    if args.task != "":
        gcs_zips=[f"gs://cmu-gpucloud-yinongh/flow_1206/asset_{args.task}.zip"]
    print(f"Found {len(gcs_zips)} zip files on GCS")

    nproc = min(NUM_PROCESSES, cpu_count())
    print(f"Using {nproc} processes")

    with Pool(processes=nproc) as pool:
        pool.map(download_and_extract, gcs_zips)


if __name__ == "__main__":
    main()