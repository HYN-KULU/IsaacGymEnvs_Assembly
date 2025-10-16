import torch
import matplotlib.pyplot as plt

# =======================================
# Quaternion utilities
# =======================================
def quat_conjugate(q):
    return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)

def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack((
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ), dim=-1)

def quat_apply(q, v):
    q_conj = quat_conjugate(q)
    v_as_quat = torch.cat([v, torch.zeros_like(v[..., :1])], dim=-1)
    return quat_mul(quat_mul(q, v_as_quat), q_conj)[..., :3]

# =======================================
# Core function
# =======================================
def compute_target_gripper_pose(current_plug_pos, current_plug_quat,
                                target_plug_pos, target_plug_quat,
                                current_gripper_pos, current_gripper_quat):
    plug_in_gripper_pos = quat_apply(quat_conjugate(current_gripper_quat),
                                     current_plug_pos - current_gripper_pos)
    plug_in_gripper_quat = quat_mul(quat_conjugate(current_gripper_quat),
                                    current_plug_quat)
    target_gripper_quat = quat_mul(target_plug_quat, quat_conjugate(plug_in_gripper_quat))
    target_gripper_pos = target_plug_pos - quat_apply(target_gripper_quat, plug_in_gripper_pos)
    return target_gripper_pos, target_gripper_quat

# =======================================
# Visualization helper
# =======================================
def draw_frame(ax, origin, quat, length=0.02, label='', color_base=(1, 0, 0), alpha=1.0):
    x, y, z, w = quat
    R = torch.tensor([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])
    colors = ['r', 'g', 'b']
    for i in range(3):
        start, end = origin, origin + R[:, i] * length
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=colors[i], alpha=alpha, linewidth=2)
    ax.text(origin[0], origin[1], origin[2], label, color=color_base, fontsize=8)

def visualize_alignment(current_plug_pos, current_plug_quat,
                        target_plug_pos, target_plug_quat,
                        current_gripper_pos, current_gripper_quat,
                        target_gripper_pos, target_gripper_quat,
                        save_name=None):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 当前状态（浅色）
    draw_frame(ax, current_plug_pos, current_plug_quat,
               label='plug_now', color_base=(1, 0, 0), alpha=0.5)
    draw_frame(ax, current_gripper_pos, current_gripper_quat,
               label='gripper_now', color_base=(0, 0, 1), alpha=0.5)

    # 目标状态（深色）
    draw_frame(ax, target_plug_pos, target_plug_quat,
               label='plug_target', color_base=(1, 0, 0), alpha=1.0)
    draw_frame(ax, target_gripper_pos, target_gripper_quat,
               label='gripper_target', color_base=(0, 0, 1), alpha=1.0)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("Plug & Gripper Alignment")
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=400)
        print(f"Saved figure to {save_name}")

# =======================================
# Main testing using your test_data.pth
# =======================================
if __name__ == "__main__":
    test_data = torch.load("test_data.pth", map_location="cpu")

    init_plug_root_state = test_data["init_plug_state"]
    plug_pos = test_data["plug_pos"]
    plug_quat = test_data["plug_quat"]
    fingertip_centered_pos = test_data["fingertip_centered_pos"]
    fingertip_centered_quat = test_data["fingertip_centered_quat"]

    # choose one env (e.g. env 0)
    i = 0
    current_plug_pos = plug_pos[i]
    current_plug_quat = plug_quat[i]
    target_plug_pos = init_plug_root_state[i, :3]
    target_plug_quat = init_plug_root_state[i, 3:7]
    current_gripper_pos = fingertip_centered_pos[i]
    current_gripper_quat = fingertip_centered_quat[i]

    target_gripper_pos, target_gripper_quat = compute_target_gripper_pose(
        current_plug_pos.unsqueeze(0),
        current_plug_quat.unsqueeze(0),
        target_plug_pos.unsqueeze(0),
        target_plug_quat.unsqueeze(0),
        current_gripper_pos.unsqueeze(0),
        current_gripper_quat.unsqueeze(0)
    )

    target_gripper_pos = target_gripper_pos[0]
    target_gripper_quat = target_gripper_quat[0]

    visualize_alignment(current_plug_pos, current_plug_quat,
                        target_plug_pos, target_plug_quat,
                        current_gripper_pos, current_gripper_quat,
                        target_gripper_pos, target_gripper_quat,
                        save_name=f"alignment_test_env{i}.png")
