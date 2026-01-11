#!/bin/bash
set -euo pipefail

BASE_SRC="/home/ubuntu/automate/alltracker/flow_mask_data"
BASE_DEST="orchard-login-001:/project/flame/yinongh/flow/flow_mask/asset_00681"
ZONE="us-central1-c"
PROJECT="cmu-gpu-cloud"

CHECK_INTERVAL=5
STABLE_TIME=10

wait_for_file_ready() {
    local file="$1"
    local last_size=-1
    local stable_for=0

    while true; do
        if [[ ! -f "$file" ]]; then
            sleep "$CHECK_INTERVAL"
            continue
        fi

        local curr_size
        curr_size=$(stat -c%s "$file" 2>/dev/null || echo 0)

        if [[ "$curr_size" -eq "$last_size" ]]; then
            stable_for=$((stable_for + CHECK_INTERVAL))
        else
            stable_for=0
        fi

        last_size="$curr_size"

        if (( stable_for >= STABLE_TIME )); then
            return 0
        fi

        sleep "$CHECK_INTERVAL"
    done
}

# 你要处理的 asset 列表（这里只有一个 00681，可扩展）
ASSET="00681"

# index 范围
START=0
END=50

echo "=== Handling flow_asset_${ASSET}_{${START}..${END}} ==="

# 创建远程文件夹
gcloud compute ssh --zone "$ZONE" --project "$PROJECT" --tunnel-through-iap orchard-login-001 -- \
    "mkdir -p ${BASE_DEST#orchard-login-001:}"

for i in $(seq "$START" "$END"); do
    file_i="${BASE_SRC}/flow_asset_${ASSET}_${i}.h5"
    next_i=$((i + 1))
    next_file="${BASE_SRC}/flow_asset_${ASSET}_${next_i}.h5"

    echo "Waiting for $next_file to stabilize (means $file_i finished)..."
    wait_for_file_ready "$next_file"

    if [[ -f "$file_i" ]]; then
        echo "Uploading $file_i → $BASE_DEST ..."
        gcloud compute scp \
            --zone "$ZONE" \
            --project "$PROJECT" \
            --tunnel-through-iap \
            "$file_i" "$BASE_DEST"
        echo "Uploaded $file_i"
    else
        echo "WARNING: $file_i not found"
    fi
done

echo "===== All flow files uploaded successfully. ====="
