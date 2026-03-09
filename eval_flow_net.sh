python eval_flow_net_vid.py \
    --traj 2 \
    --ckpt /home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/mean_centered_flow_net_multi_ckpt/epoch_65.pt \
    --out traj_00028_meantest_ours2.mp4 \
    --task_id 00028
    # --ckpt /home/ubuntu/automate/IsaacGymEnvs_Assembly/logs/automate/flow_net_multitask_ckpt/multitask_last.pt \
# python eval_flow_net_vid.py \
#     --traj 2 \
#     --ckpt logs/automate/flow_net_multitask_ckpt/epoch_00731_15.pt \
#     --out traj00731_pre_pred_flow.mp4 \
#     --task_id 00731_pre