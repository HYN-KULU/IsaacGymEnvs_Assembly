import os
import subprocess
import zipfile
from multiprocessing import Pool, cpu_count
import argparse
# =======================
# CONFIG
# =======================
BUCKET_PATH = "gs://cmu-gpucloud-yinongh/flow_mask_data_1206"

ZIP_DIR = "/tmp/data/flow_mask_data_1206_zips"
DEST_DIR = "/tmp/data/flow_mask_data_1206"

NUM_PROCESSES = 8

os.makedirs(ZIP_DIR, exist_ok=True)
os.makedirs(DEST_DIR, exist_ok=True)


def list_gcs_zips():
    """List all task zip files on GCS."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", f"{BUCKET_PATH}/*.zip"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines()]


def download_and_extract(gcs_zip_path: str):
    zip_name = os.path.basename(gcs_zip_path)
    task_id = zip_name.replace(".zip", "")

    local_zip_path = os.path.join(ZIP_DIR, zip_name)
    task_dir = os.path.join(DEST_DIR, task_id)

    if os.path.exists(task_dir):
        print(f"⏩ Skip (exists): {task_id}")
        return

    try:
        print(f"⬇️ Downloading {zip_name}")
        subprocess.run(
            ["gcloud", "storage", "cp", gcs_zip_path, local_zip_path],
            check=True,
            capture_output=True,
            text=True,
        )

        print(f"📂 Extracting {zip_name}")
        with zipfile.ZipFile(local_zip_path, "r") as zipf:
            zipf.extractall(DEST_DIR)

        print(f"✅ Done {task_id}")

        # 可选：解压后删除 zip
        # os.remove(local_zip_path)

    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed {task_id}:\n{e.stderr}")

    except Exception as e:
        print(f"❌ Error {task_id}: {e}")
ADAPTATION_TASKS = {
    "00081", "00360", "00028", "00597", "00103",
    "00553", "00731", "00015", "00648", "00506",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="", required = False)
    args=parser.parse_args()
    gcs_zips = list_gcs_zips()
    print("First zip: ", gcs_zips[0])
    gcs_zips = ["gs://cmu-gpucloud-yinongh/flow_mask_data_1206/00028.zip"]
    # gcs_zips = [f"gs://cmu-gpucloud-yinongh/flow_mask_data_1206/{task}.zip" for task in ADAPTATION_TASKS]
    #gcs_zips = [gcs_zips[0]]
    print(f"Found {len(gcs_zips)} tasks on GCS")

    nproc = min(NUM_PROCESSES, cpu_count())
    print(f"Using {nproc} processes")

    with Pool(processes=nproc) as pool:
        pool.map(download_and_extract, gcs_zips)


if __name__ == "__main__":
    main()