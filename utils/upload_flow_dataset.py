import os
import subprocess
import zipfile
from multiprocessing import Pool, cpu_count

# =======================
# CONFIG (按你当前路径写)
# =======================
SOURCE_DIR = "/home/ubuntu/automate/ego_flow_forward_0321"
ZIP_DIR = "/home/ubuntu/automate/ego_flow_forward_0321_zips"
BUCKET_PATH = "gs://cmu-gpucloud-yinongh/ego_flow_forward_0321"
NUM_PROCESSES = 16

os.makedirs(ZIP_DIR, exist_ok=True)


def gcs_file_exists(gcs_path: str) -> bool:
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def zip_task(task_id: str):
    task_dir = os.path.join(SOURCE_DIR, task_id)
    zip_path = os.path.join(ZIP_DIR, f"{task_id}.zip")
    gcs_path = f"{BUCKET_PATH}/{task_id}.zip"

    if gcs_file_exists(gcs_path):
        print(f"⏩ Skip (exists): {task_id}.zip")
        return

    try:
        print(f"📦 Zipping task {task_id}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(task_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, SOURCE_DIR)
                    zipf.write(full_path, rel_path)

        print(f"☁️ Uploading {task_id}.zip")
        subprocess.run(
            ["gcloud", "storage", "cp", zip_path, gcs_path],
            check=True,
            capture_output=True,
            text=True,
        )

        print(f"✅ Done {task_id}")

        # 可选：上传成功后删除本地 zip
        # os.remove(zip_path)

    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed {task_id}: {e.stderr}")

    except Exception as e:
        print(f"❌ Error {task_id}: {e}")


def main():

    task_ids = [
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

    print(f"Found {len(task_ids)} tasks")

    nproc = min(NUM_PROCESSES, cpu_count())
    print(f"Using {nproc} processes")

    with Pool(processes=nproc) as pool:
        pool.map(zip_task, task_ids)


if __name__ == "__main__":
    main()
