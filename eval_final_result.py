import numpy as np
import matplotlib.pyplot as plt
# result = np.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/eval_result/asset_00731.npy")
# result = np.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/eval_result_baseline_diffusion_policy/asset_00015.npy")
# result = np.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/eval_dp_finetune/asset_00015.npy")
result = np.load("/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/eval_result_3dgt_flow/asset_00028.npy")
xy_dist_th = 5e-4
z_th = 1e-1
# print(result)
# result[:30,0] = result[:30,0] / 20  # mm to m
# result[15:25,0] = result[15:25,0]   # mm to m
# result[30:,0] = result[30:,0] / 2  # mm to m
count = np.sum(
    (result[:, 0] < xy_dist_th) &
    (result[:, 1] < z_th)
)


print(count, result.shape[0], count / result.shape[0]) 
print("Average Error: ", np.mean(result, axis=0))

num_target = 100
idx = np.random.choice(result.shape[0], size=num_target, replace=True)
result = result[idx]

x = result[:, 0]

plt.figure()
plt.hist(x, bins=100)
plt.xlabel("XY Distance Error (m)")
plt.ylabel("Count")
plt.title("Task 00553: Diffusion Policy Conditioned on Finetuned Flow")
plt.grid(True)

save_path = "/home/ubuntu/automate/IsaacGymEnvs_Assembly/isaacgymenvs/eval_result_hist_x.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved histogram to {save_path}")
# import pdb;pdb.set_trace()
