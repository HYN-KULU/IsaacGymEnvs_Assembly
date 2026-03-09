#!/bin/bash
set -e  # stop if any command fails

MESH_DIR="/home/ubuntu/automate/IsaacGymEnvs_Assembly/assets/automate/mesh"
source ~/automate/miniconda3/bin/activate
# iterate through every task_id folder
for task_id in $(ls "$MESH_DIR"); do
    echo "========================================="
    echo "Processing task_id: $task_id"
    echo "========================================="

    ###############################################
    # 1. Run Train (IsaacGym)
    ###############################################
    conda activate rlgpu
    cd ~/automate/IsaacGymEnvs_Assembly
    source prepare.sh
    cd isaacgymenvs
    python train.py \
        task=AutoMateTaskDisassemble \
        task.env.overwrite_subassemblies=True \
        "task.env.desired_subassemblies=['asset_${task_id}']" \
        headless=True \
        seed=100

    conda deactivate

    ###############################################
    # 2. Run AllTracker (flow extraction)
    ###############################################
    conda activate alltracker
    cd ~/automate/alltracker

    python process_flow.py \
        --mp4_path "./demo_video/${task_id}.mp4" \
        --query_frame 32 \
        --conf_thr 0.01 \
        --bkg_opacity 0.0 \
        --rate 2 \
        --hstack \
        --query_frame 16 \
        --task_id "${task_id}"

    conda deactivate

    ###############################################
    # 3. Run Preprocessing (flow + depth tensor)
    ###############################################
    conda activate rlgpu
    cd ~/automate/IsaacGymEnvs_Assembly

    python preprocess_data_flow_mask.py \
        --task_id "${task_id}"

    conda deactivate

    echo "✔ Finished task_id: $task_id"
    echo
done

echo "🎉 All tasks completed!"
