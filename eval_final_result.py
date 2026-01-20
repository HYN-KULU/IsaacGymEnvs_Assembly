import numpy as np
result = np.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/eval_result/asset_00360.npy")
# result = np.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/eval_result_baseline_diffusion_policy/asset_00360.npy")
# result = np.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/eval_adapt/asset_00028.npy")
xy_dist_th = 5e-3
z_th = 3e-2
count = np.sum(
    (result[:, 0] < xy_dist_th) &
    (result[:, 1] < z_th)
)

print(count, result.shape[0], count / result.shape[0])
# import pdb;pdb.set_trace()
