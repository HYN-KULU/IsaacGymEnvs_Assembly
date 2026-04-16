import torch
import math


def quat_normalize(q, eps=1e-8):
    return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=eps)


def quat_conjugate(q):
    qc = q.clone()
    qc[:, :3] = -qc[:, :3]
    return qc


def quat_mul(q1, q2):
    # quaternion format: (x, y, z, w)
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ], dim=-1)

def quat_to_axis_angle(q, eps=1e-8):
    """
    q: [N, 4], normalized quaternion (x, y, z, w)
    return:
        axis: [N, 3]
        angle: [N]
    """
    q = quat_normalize(q)

    xyz = q[:, :3]
    w = torch.clamp(q[:, 3], -1.0, 1.0)

    sin_half = torch.norm(xyz, dim=-1)
    angle = 2.0 * torch.atan2(sin_half, w)

    # wrap to [-pi, pi]
    angle = (angle + math.pi) % (2 * math.pi) - math.pi

    axis = xyz / torch.clamp(sin_half.unsqueeze(-1), min=eps)

    default_axis = torch.zeros_like(axis)
    default_axis[:, 2] = 1.0
    small_mask = sin_half < eps
    axis[small_mask] = default_axis[small_mask]

    return axis, angle

def axis_angle_to_quat(axis, angle):
    """
    axis: [N, 3]
    angle: [N]
    return [N, 4] in (x, y, z, w)
    """
    axis = axis / torch.clamp(torch.norm(axis, dim=-1, keepdim=True), min=1e-8)

    half = 0.5 * angle
    s = torch.sin(half).unsqueeze(-1)   # [N, 1]
    c = torch.cos(half).unsqueeze(-1)   # [N, 1]

    q = torch.cat([axis * s, c], dim=-1)   # [N, 4]
    return quat_normalize(q)

def quat_step_toward(current_q, target_q, max_angle_step_deg=0.5):
    """
    返回一个小步旋转 q_step，使 current_q 朝 target_q 逼近一点点
    """
    current_q = quat_normalize(current_q)
    target_q = quat_normalize(target_q)

    # 当前 -> 目标 的误差旋转
    q_err = quat_mul(target_q, quat_conjugate(current_q))
    q_err = quat_normalize(q_err)

    axis, angle = quat_to_axis_angle(q_err)

    max_angle_step = torch.deg2rad(
        torch.tensor(max_angle_step_deg, device=current_q.device, dtype=current_q.dtype)
    )

    step_angle = torch.clamp(angle, min=-max_angle_step, max=max_angle_step)

    q_step = axis_angle_to_quat(axis, step_angle)
    return q_step, angle

def yaw_quat_from_angle(yaw):
    """
    yaw: [N]
    return quaternion [N, 4] in (x, y, z, w)
    """
    q = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
    q[:, 2] = torch.sin(0.5 * yaw)   # z
    q[:, 3] = torch.cos(0.5 * yaw)   # w
    return quat_normalize(q)


def disturb_quat_in_yaw(target_q, deg_min=5.0, deg_max=20.0, both_directions=True, world_frame=True):
    """
    对 target_q 加随机 yaw 扰动

    Args:
        target_q: [N, 4], quaternion in (x, y, z, w)
        deg_min: 最小扰动角度（度）
        deg_max: 最大扰动角度（度）
        both_directions: 是否允许正负两个方向扰动
        world_frame: 
            True  -> delta * target_q   （世界坐标系 z 轴）
            False -> target_q * delta   （局部坐标系 z 轴）

    Returns:
        disturbed_q: [N, 4]
        yaw_deg: [N]，实际加的扰动角度（度）
    """
    target_q = quat_normalize(target_q)

    N = target_q.shape[0]
    device = target_q.device
    dtype = target_q.dtype

    yaw_deg = deg_min + (deg_max - deg_min) * torch.rand((N,), device=device, dtype=dtype)

    if both_directions:
        sign = torch.where(
            torch.rand((N,), device=device, dtype=dtype) > 0.5,
            torch.tensor(1.0, device=device, dtype=dtype),
            torch.tensor(-1.0, device=device, dtype=dtype),
        )
        yaw_deg = yaw_deg * sign

    yaw_rad = torch.deg2rad(yaw_deg)
    delta_q = yaw_quat_from_angle(yaw_rad)

    if world_frame:
        disturbed_q = quat_mul(delta_q, target_q)
    else:
        disturbed_q = quat_mul(target_q, delta_q)

    disturbed_q = quat_normalize(disturbed_q)
    return disturbed_q, yaw_deg