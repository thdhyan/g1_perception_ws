#!/usr/bin/env python3
"""CenterPoint 3D Detection Model & Backend for Livox Mid-360 LiDAR.

Uses the official OpenPCDet / Livox LD_base architecture and pretrained checkpoint.
Applies exact ground offset preprocessing and CUDA 3D IoU NMS.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# Add OpenPCDet / livox_detection project directory to sys.path
LIVOX_PROJ_DIR = "/home/thakk100/Projects/Thesis/livox_detection"
if LIVOX_PROJ_DIR not in sys.path:
    sys.path.insert(0, LIVOX_PROJ_DIR)

try:
    from livoxdetection.models.ld_base_v1 import LD_base
except ImportError:
    LD_base = None

CLASS_NAMES = ("car", "pedestrian", "cyclist")
CLASS_COLORS = {
    "car": (0.0, 1.0, 1.0),
    "pedestrian": (1.0, 1.0, 0.0),
    "cyclist": (0.0, 1.0, 0.0),
}


def mask_points_out_of_range(pc: np.ndarray, pc_range: list[float]) -> np.ndarray:
    pc_range_arr = np.array(pc_range, dtype=np.float32)
    pc_range_arr[3:6] -= 0.01
    mask_x = (pc[:, 0] > pc_range_arr[0]) & (pc[:, 0] < pc_range_arr[3])
    mask_y = (pc[:, 1] > pc_range_arr[1]) & (pc[:, 1] < pc_range_arr[4])
    mask_z = (pc[:, 2] > pc_range_arr[2]) & (pc[:, 2] < pc_range_arr[5])
    return pc[mask_x & mask_y & mask_z]


class CenterPointBackend:
    """Official LD_base CenterPoint backend for Livox Mid-360 LiDAR."""

    def __init__(
        self,
        checkpoint: str = "/home/thakk100/Projects/Thesis/livox_detection/pt/livox_model_1.pt",
        device: str = "cuda",
        score_threshold: float = 0.30,
        offset_ground: float = 1.33,
        offset_angle: float = 0.0,
    ):
        self.checkpoint = checkpoint
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.model = None
        self.logger = logging.getLogger("CenterPointBackend")

        # Preprocessing parameters (1.33m aligns G1 Mid-360 ground plane with model voxel grid)
        self.offset_angle = offset_angle
        self.offset_ground = offset_ground
        self.point_cloud_range = [0, -44.8, -2, 224, 44.8, 4]

    def load(self) -> None:
        if LD_base is None:
            raise ImportError(f"Could not import LD_base from {LIVOX_PROJ_DIR}")

        ckpt_path = Path(self.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self.model = LD_base()
        checkpoint_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model_state = checkpoint_data.get("model_state_dict", checkpoint_data)
        cleaned_state = {k.replace("module.", ""): v for k, v in model_state.items()}
        self.model.load_state_dict(cleaned_state)
        # Set head score threshold lower so we get candidates down to self.score_threshold
        t = max(0.05, float(self.score_threshold))
        self.model.head.POST_PROCESSING['SCORE_THRESH'] = [t, t, t]
        self.model.to(self.device).eval()
        self.logger.info(f"Loaded CenterPoint LD_base checkpoint from {ckpt_path} (Head Threshold: {t:.2f})")


    def infer(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run CenterPoint inference on (N, 4) point cloud [x, y, z, intensity]."""
        if self.model is None or points.shape[0] == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        # 1. Preprocess input points
        pts = points.copy()
        if pts.shape[1] < 4:
            pts = np.hstack([pts[:, :3], np.zeros((pts.shape[0], 1), dtype=np.float32)])

        pts_raw_min, pts_raw_max = pts[:, :3].min(axis=0), pts[:, :3].max(axis=0)

        pts[:, 2] += pts[:, 0] * np.tan(self.offset_angle / 180.0 * np.pi) + self.offset_ground
        pts_masked = mask_points_out_of_range(pts, self.point_cloud_range)

        if pts_masked.shape[0] == 0:
            self.logger.warning(
                f"[PointCloud Masked to 0!] Raw bounds: X=[{pts_raw_min[0]:.2f}, {pts_raw_max[0]:.2f}], "
                f"Y=[{pts_raw_min[1]:.2f}, {pts_raw_max[1]:.2f}], Z=[{pts_raw_min[2]:.2f}, {pts_raw_max[2]:.2f}]"
            )
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )


        # 2. Pad batch index 0 and prepare CUDA dict
        coor_pad = np.pad(pts_masked, ((0, 0), (1, 0)), mode="constant", constant_values=0)
        data_infer = {
            "points": torch.from_numpy(coor_pad).float().to(self.device),
            "batch_size": 1,
        }

        # 3. Model forward pass
        with torch.no_grad():
            pred_dicts = self.model(data_infer)

        if not pred_dicts or len(pred_dicts) == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        boxes = pred_dicts[0]["pred_boxes"].cpu().numpy()
        scores = pred_dicts[0]["pred_scores"].cpu().numpy()
        labels = pred_dicts[0]["pred_labels"].cpu().numpy()

        if boxes.shape[0] == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        # 4. Filter by score threshold
        keep = scores >= self.score_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        if boxes.shape[0] == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        # 5. Undo ground offset on z center to return boxes in raw sensor coordinates
        boxes[:, 2] -= self.offset_ground

        # Convert 1-indexed labels (1=Vehicle, 2=Pedestrian, 3=Cyclist) to 0-indexed
        labels = labels - 1

        return boxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int64)
