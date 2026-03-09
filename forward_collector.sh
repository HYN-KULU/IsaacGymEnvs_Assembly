#!/bin/bash

# Activate conda environment
source ~/automate/miniconda3/bin/activate
conda activate rlgpu
cd ~/automate/IsaacGymEnvs_Assembly
source prepare.sh
# Go into isaacgymenvs
cd isaacgymenvs

## 00345 throw

train_tasks=(
    # 10001
    00648
    # 00004 00015 00016 00021 00028 00030 00042
    # 00074 00077 00078 00081 00103 00110 00117
    # 00133 00138 00141 00163 00175 00186 00187
    # 00192 00211 00213 00255 00256 00271 00293
    # 00301 00318 00319 00320 00329 00345 00346
    # 00360 00388 00410 00417 00422 00426 00437
    # 00444 00446 00471 00480 00499 00506 00514
    # 00537 00553 00559 00581 00597 00614 00615
    # 00638 00648 00649 
    # 00659 00681 00686 00700
    # 00703 
    # 00731 00768 00783 00855 00860
    # 01041 01079 01092 01102
    # 01129 01132 01136
)

# train_tasks=(
#     00004 00015 00016 00021 00028 00030 00042
#     00074 00077 00078 00081 00103 00110 00117
#     00133 00138 00141 00163 00175 00186 00187
#     00192 00211 00213 00255 00256 00271 00293
#     00301 00318 00319 00320 00329 00345 00346
#     00360 00388 00410 00417 00422 00426 00437
#     00444 00446 00471 00480 00499 00506 00514
#     00537 00553 00559 00581 00597 00614 00615
#     00638 00648 00649 00659 00681 00686 00700
#     00703 00726 00731 00768 00783 00855 00860
#     01026 01029 01036 01041 01079 01092 01102
#     01129 01132 01136
# )

for task_id in "${train_tasks[@]}"; do
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
    for seed in $(seq 0 0); do
        python forward_data_collector.py \
            task=AutoMateTaskDisassemble \
            task.env.overwrite_subassemblies=True \
            "task.env.desired_subassemblies=['asset_${task_id}']" \
            headless=True \
            seed=$seed
    done
done

