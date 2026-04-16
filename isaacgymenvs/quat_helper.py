import torch
import math

import torch


def quat_normalize(q, eps=1e-8):
    return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=eps)


def quat_conjugate(q):
    qc = q.clone()
    qc[:, :3] = -qc[:, :3]
    return qc


def quat_mul(q1, q2):
    # q = (x, y, z, w)
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ], dim=-1)


def yaw_from_quat(q):
    """
    Extract yaw from quaternion (x, y, z, w).
    Returns angle in radians in [-pi, pi].
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def yaw_quat(yaw):
    q = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
    q[:, 2] = torch.sin(0.5 * yaw)
    q[:, 3] = torch.cos(0.5 * yaw)
    return q