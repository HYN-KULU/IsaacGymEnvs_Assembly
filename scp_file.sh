#!/bin/bash
# Upload disassembly trajectory files to remote GCP instance

# scp -P 6024  /home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/3dgt_flow_policy/epoch_0.pt fengying@58.34.245.178:/home/fengying/automate/logs
scp -r -P 6024  /home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/tasks/automate/data/flow_0125_gt_zips/asset_00559.zip fengying@58.34.245.178:/home/fengying/automate/data/
# scp -P 6024 fengying@58.34.245.178:/home/fengying/automate/logs/metaflow_nofreeze/meta_flow_checkpoint_15000.pt /home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/metaflow_nofreeze/
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
#     /home/ubuntu/automate/flow_diffusion_policy_multitask_0206.h5\
#     orchard-login-001:/project/flame/yinongh
    

# gcloud compute scp --zone "us-central1-c" --project "cmu-gpu-cloud" --tunnel-through-iap orchard-login-001:/project/flame/yinongh/processed_dataset_depth_relative_insertion_net_action.h5 /home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/insertion_net_multitask_ckpt/
# gcloud compute scp --zone "us-central1-c" --project "cmu-gpu-cloud" --tunnel-through-iap orchard-login-001:~/automate/train_masked_3dgt_flow_mask_input.log /home/ubuntu/automate/IsaacGymEnvs_Assembly/
# gcloud compute scp --zone "us-central1-c" --project "cmu-gpu-cloud" --tunnel-through-iap orchard-login-001:~/automate/logs/automate/masked_3dgt_flow_net_mask_input_ckpt/epoch_195.pt /home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/3dgt_flow_policy
# echo "✅ All files uploaded successfully!"
# (base) yinongh@orchard-login-001:~/automate$ ls /project/flame/yinongh/flow
# asset_00681  flow_mask
# (base) yinongh@orchard-login-001:~/automate$ ls /project/flame/yinongh/flow/flow_mask
# asset_00681

