import os
import subprocess
import zipfile
from multiprocessing import Pool, cpu_count
import argparse
# =======================
# CONFIG
# =======================
BUCKET_PATH = "gs://cmu-gpucloud-yinongh/ego_flow_data_0130_acc_flow_keyframe"

ZIP_DIR = "/tmp/data/ego_flow_data_0130_zips"
DEST_DIR = "/tmp/data/ego_flow_data_0130"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="", required = False)
    args=parser.parse_args()
    gcs_zips = list_gcs_zips()
    print("First zip: ", gcs_zips[0])
    #gcs_zips = ["gs://cmu-gpucloud-yinongh/flow_mask_data_1206/00021.zip", "gs://cmu-gpucloud-yinongh/flow_mask_data_1206/00028.zip", "gs://cmu-gpucloud-yinongh/flow_mask_data_1206/00030.zip", "gs://cmu-gpucloud-yinongh/flow_mask_data_1206/00042.zip", "gs://cmu-gpucloud-yinongh/flow_mask_data_1206/00110.zip", "gs://cmu-gpucloud-yinongh/flow_mask_data_1206/00681.zip"]
    #gcs_zips = [f"gs://cmu-gpucloud-yinongh/flow_mask_data_1206/{args.task}.zip"]
    train_tasks=[
                '00004', '00016', '00021', '00030', '00074', '00078', '00110', '00117', '00133', '00138', '00141', '00163', '00175',
                 '00186', '00187', '00192', '00211', '00213', '00255', '00256', '00271', '00293', '00301', '00318', '00319', '00320',
                 '00329', '00346', '00388', '00410', '00417', '00422', '00426', '00437', '00444', '00446', '00471', '00480', '00499' ,
                 '00514', '00537', '00559', '00615', '00638', '00649', '00659', '00681', '00686', '00700', '00703', '00768', '00783' ,
                 '00860', '01029', '01041', '01092', '01102', '01132', '01136'
                 ]  
    #train_tasks = ['00004']
    gcs_zips = [f"{BUCKET_PATH}/{task}.zip" for task in train_tasks]
    #gcs_zips = [gcs_zips[0]]
    print(f"Found {len(gcs_zips)} tasks on GCS")

    nproc = min(NUM_PROCESSES, cpu_count())
    print(f"Using {nproc} processes")

    with Pool(processes=nproc) as pool:
        pool.map(download_and_extract, gcs_zips)


if __name__ == "__main__":
    main()