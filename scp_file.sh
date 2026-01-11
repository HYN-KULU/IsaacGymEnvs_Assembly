#!/bin/bash
# Upload disassembly trajectory files to remote GCP instance

# scp -P 6024 fengying@58.34.245.178:/home/fengying/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/depth_relative_1030_multitask/disassembly_traj*.h5 /home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/automate_1030_multitask/
# loop from 0 to 24
# for id in $(seq 0 10); do
#     # echo "Uploading file disassembly_traj_${id}.h5 ..."
#     gcloud compute scp \
#         --zone "us-central1-c" \
#         --project "cmu-gpu-cloud" \
#         --tunnel-through-iap \
#         /home/ubuntu/automate/alltracker/flow_mask_data/00030/flow_asset_00030_${id}.h5 \
#         orchard-login-001:/project/flame/yinongh/flow/flow_mask/asset_00030
# done
# for id in $(seq 0 50); do
#     # echo "Uploading file disassembly_traj_${id}.h5 ..."
#     gcloud compute scp \
#         --zone "us-central1-c" \
#         --project "cmu-gpu-cloud" \
#         --tunnel-through-iap \
#         /home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow/asset_00110/disassembly_traj_${id}.h5 \
#         orchard-login-001:/project/flame/yinongh/flow/asset_00110
# done

# echo "Uploading file disassembly_traj_${id}.h5 ..."
# gcloud compute scp \
#     --zone "us-central1-c" \
#     --project "cmu-gpu-cloud" \
#     --tunnel-through-iap \
#     /home/ubuntu/automate/IsaacGymEnvs_Assembly/flow_diffusion_policy_00681_0108.h5\
#     orchard-login-001:/project/flame/yinongh
    

gcloud compute scp --zone "us-central1-c" --project "cmu-gpu-cloud" --tunnel-through-iap orchard-login-001:/home/yinongh/automate/logs/automate/diffusion_policy_flow_cond_00681/policy_epoch_600.ckpt /home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/diffusion_policy_flow_cond/
# echo "✅ All files uploaded successfully!"
# (base) yinongh@orchard-login-001:~/automate$ ls /project/flame/yinongh/flow
# asset_00681  flow_mask
# (base) yinongh@orchard-login-001:~/automate$ ls /project/flame/yinongh/flow/flow_mask
# asset_00681