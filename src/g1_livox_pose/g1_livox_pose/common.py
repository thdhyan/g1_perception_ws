"""Shared constants and geometry helpers for g1_livox_pose."""

from __future__ import annotations

import math
import numpy as np


# Waymo 14-joint convention (matches LPFormer / DAPT / VoxelKP).
JOINT_NAMES = (
    "nose",              # 0
    "left_shoulder",     # 1
    "left_elbow",        # 2
    "left_wrist",        # 3
    "left_hip",          # 4
    "left_knee",         # 5
    "left_ankle",        # 6
    "right_shoulder",    # 7
    "right_elbow",       # 8
    "right_wrist",       # 9
    "right_hip",         # 10
    "right_knee",        # 11
    "right_ankle",       # 12
    "head",              # 13
)
NUM_JOINTS = len(JOINT_NAMES)

BONES = (
    (0, 13), (13, 1), (13, 7),   # head/neck
    (1, 7), (1, 4), (7, 10), (4, 10),  # torso
    (1, 2), (2, 3),              # left arm
    (7, 8), (8, 9),              # right arm
    (4, 5), (5, 6),              # left leg
    (10, 11), (11, 12),          # right leg
)


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw from a unit quaternion."""
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix."""
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def detection_to_box7(det) -> np.ndarray:
    """vision_msgs Detection3D -> [x, y, z, dx, dy, dz, yaw]."""
    c = det.bbox.center
    s = det.bbox.size
    return np.array([
        c.position.x, c.position.y, c.position.z,
        s.x, s.y, s.z,
        quaternion_to_yaw(c.orientation.x, c.orientation.y, c.orientation.z, c.orientation.w),
    ], dtype=np.float64)


def crop_points_in_box(points: np.ndarray, box7: np.ndarray, margin: float = 0.3) -> np.ndarray:
    """Select points inside the yaw-oriented box expanded by margin (meters)."""
    if points.shape[0] == 0:
        return points
    cx, cy, cz, dx_, dy_, dz_, yaw = box7
    dxp = points[:, 0] - cx
    dyp = points[:, 1] - cy
    dzp = points[:, 2] - cz
    c, s = math.cos(yaw), math.sin(yaw)
    xr = c * dxp + s * dyp
    yr = -s * dxp + c * dyp
    mask = (
        (np.abs(xr) <= dx_ / 2.0 + margin)
        & (np.abs(yr) <= dy_ / 2.0 + margin)
        & (np.abs(dzp) <= dz_ / 2.0 + margin)
    )
    return points[mask]
