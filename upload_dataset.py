import os
import subprocess
import zipfile
from multiprocessing import Pool, cpu_count

# =======================
# CONFIG
# =======================
SOURCE_DIR = (
    "/home/ubuntu/automate/IsaacGymEnvs_Assembly/"
    "isaacgymenvs/tasks/automate/data/flow_0125_gt"
)

ZIP_DIR = (
    "/home/ubuntu/automate/IsaacGymEnvs_Assembly/"
    "isaacgymenvs/tasks/automate/data/flow_0125__adapt_zips"
)

BUCKET_PATH = "gs://cmu-gpucloud-yinongh/flow_0125_gt"
NUM_PROCESSES = 32

os.makedirs(ZIP_DIR, exist_ok=True)


def gcs_file_exists(gcs_path: str) -> bool:
    """Check if a file exists in GCS."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def zip_and_upload(asset_dir: str):
    asset_path = os.path.join(SOURCE_DIR, asset_dir)
    zip_path = os.path.join(ZIP_DIR, f"{asset_dir}.zip")
    gcs_path = f"{BUCKET_PATH}/{asset_dir}.zip"

    # if gcs_file_exists(gcs_path):
    #     print(f"⏩ Skip (exists): {asset_dir}.zip")
    #     return

    try:
        print(f"📦 Zipping {asset_dir}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(asset_path):
                for file in files:
                    print(file)
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, SOURCE_DIR)
                    zipf.write(full_path, rel_path)

        # print(f"☁️ Uploading {asset_dir}.zip")
        # subprocess.run(
        #     ["gcloud", "storage", "cp", zip_path, gcs_path],
        #     check=True,
        #     capture_output=True,
        #     text=True,
        # )

        # print(f"✅ Done {asset_dir}")

        # 可选：上传成功后删除本地 zip，节省空间
        # os.remove(zip_path)

    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed {asset_dir}:\n{e.stderr}")

    except Exception as e:
        print(f"❌ Error processing {asset_dir}: {e}")


def main():
    asset_dirs = sorted([
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ])
    adapt_tasks = ['00345', '00360', '00028', '00614', '00103', '00553', '00731', '00015', '00648', '00506']
    asset_dirs = [f"asset_{task}" for task in adapt_tasks]
    # import pdb;pdb.set_trace()
    print(f"Found {len(asset_dirs)} assets")
    # asset_dirs=['asset_00598']
    nproc = min(NUM_PROCESSES, cpu_count())
    print(f"Using {nproc} processes")

    with Pool(processes=nproc) as pool:
        pool.map(zip_and_upload, asset_dirs)


if __name__ == "__main__":
    main()
