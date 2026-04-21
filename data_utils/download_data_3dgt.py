import os
import subprocess
import zipfile
from multiprocessing import Pool, cpu_count
import argparse
# =======================
# CONFIG
# =======================
BUCKET_PATH = "gs://cmu-gpucloud-yinongh/flow_0125_gt"

ZIP_DIR = "/tmp/data/flow_0125_gt_adapt"
EXTRACT_DIR = "/tmp/data/flow_0125_gt_adapt"

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

        # if UNZIP_AFTER_DOWNLOAD:
        #     print(f"📂 Extracting {zip_name}")
        #     with zipfile.ZipFile(local_zip_path, "r") as zipf:
        #         zipf.extractall(EXTRACT_DIR)

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
    "00831",
    "01053",
    "01125",
}

# train_tasks=['00004', '00016', '00021', '00030']
# train_tasks=[
#                 '00004', '00016', '00021', '00030', '00074', '00078', '00110', '00117', '00133', '00138', '00141', '00163', '00175',
#                  '00186', '00187', '00192', '00211', '00213', '00255', '00256', '00271', '00293', '00301', '00318', '00319', '00320',
#                  '00329', '00346', '00388', '00410', '00417', '00422', '00426', '00437', '00444', '00446', '00471', '00480', '00499' ,
#                  '00514', '00537', '00559', '00615', '00638', '00649', '00659', '00681', '00686', '00700', '00703', '00768', '00783' ,
#                  '00860', '01029', '01041', '01092', '01102', '01132', '01136'
#                  ]
train_tasks = [
    '00345', '00360', '00028', '00614', '00103',
    '00553', '00731', '00015', '00648', '00506'
]
#train_tasks = ['00004']
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
    gcs_zips = [f'{BUCKET_PATH}/asset_{train_task}.zip' for train_task in train_tasks]
    print(len(gcs_zips))
    if args.task != "":
        gcs_zips=[f"{BUCKET_PATH}/asset_{args.task}.zip"]
    print(f"Found {len(gcs_zips)} zip files on GCS")

    nproc = min(NUM_PROCESSES, cpu_count())
    print(f"Using {nproc} processes")

    with Pool(processes=nproc) as pool:
        pool.map(download_and_extract, gcs_zips)


if __name__ == "__main__":
    main()