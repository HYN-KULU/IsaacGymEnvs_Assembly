python eval_flow_net_vid.py \
    --traj 2 \
    --ckpt logs/automate/flow_net_multitask_ckpt/multi_2_epoch_100.pt \
    --out traj00681_pre_pred_flow_beforefinetune.mp4 \
    --task_id 00681_pre
# python eval_flow_net_vid.py \
#     --traj 2 \
#     --ckpt logs/automate/flow_net_multitask_ckpt/epoch_00731_15.pt \
#     --out traj00731_pre_pred_flow.mp4 \
#     --task_id 00731_pre