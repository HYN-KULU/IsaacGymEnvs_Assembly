# cd /home/ubuntu/automate/IsaacGymEnvs_Assembly/utils
# python download_data_3dgt.py
# python download_flow_data_3dgt.py
# python launcher_3dgt_flow_maskinput.py
# python index_preprocessing_numpy.py
# cd /home/ubuntu/automate/IsaacGymEnvs_Assembly
torchrun --nproc_per_node=8 train_meta_flow_multigpu.py
# 