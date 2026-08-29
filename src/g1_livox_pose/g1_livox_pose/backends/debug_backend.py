"""Debug pose backend: projects a deterministic standing figure into the box.

Zero-ML placeholder used to validate plumbing (crop -> infer -> TF ->
tracking -> sequences) end-to-end. The figure is axis-aligned with the
detection box yaw and scaled to the box height; it carries no motion
information beyond what the detector provides.
"""

from __future__ import annotations

import math
import numpy as np

from ..common import NUM_JOINTS


# Offsets as fractions of box height h, relative to box center (which sits
# at mid-height). x = forward, y = left within the box frame.
_REL = np.array([
    [0.000,  0.000,  0.44],   # nose
    [0.000,  0.120,  0.31],   # left_shoulder
    [0.000,  0.160,  0.10],   # left_elbow
    [0.000,  0.180, -0.10],   # left_wrist
    [0.000,  0.070, -0.02],   # left_hip
    [0.000,  0.080, -0.26],   # left_knee
    [0.000,  0.080, -0.48],   # left_ankle
    [0.000, -0.120,  0.31],   # right_shoulder
    [0.000, -0.160,  0.10],   # right_elbow
    [0.000, -0.180, -0.10],   # right_wrist
    [0.000, -0.070, -0.02],   # right_hip
    [0.000, -0.080, -0.26],   # right_knee
    [0.000, -0.080, -0.48],   # right_ankle
    [0.000,  0.000,  0.40],   # head
])


class DebugBackend:
    num_joints = NUM_JOINTS

    def load(self) -> None:
        pass

    def infer(self, points: np.ndarray, box7: np.ndarray):
        x, y, z = float(box7[0]), float(box7[1]), float(box7[2])
        h = max(float(box7[5]), 0.5)
        yaw = float(box7[6])

        rel = _REL.copy() * h
        c, s = math.cos(yaw), math.sin(yaw)
        kx = c * rel[:, 0] - s * rel[:, 1]
        ky = s * rel[:, 0] + c * rel[:, 1]

        keypoints = np.empty((NUM_JOINTS, 3), dtype=np.float32)
        keypoints[:, 0] = kx + x
        keypoints[:, 1] = ky + y
        keypoints[:, 2] = rel[:, 2] + z

        valid = np.ones(NUM_JOINTS, dtype=np.uint8)
        score = 1.0
        return keypoints, valid, score
