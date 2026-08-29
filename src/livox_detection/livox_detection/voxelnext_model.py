#!/usr/bin/env python3
"""VoxelNeXt 3D Detection Backend for Livox Mid-360 LiDAR.

VoxelNeXt (Chen et al., CVPR 2023) — fully sparse, anchor-free 3D object
detector. Predicts objects directly on sparse voxel features without any
sparse-to-dense BEV conversion, anchors, or center proxies.

This module wraps the OpenPCDet-based VoxelNeXt model:

    backend = VoxelNeXtBackend(...)
    backend.load()
    boxes, scores, labels = backend.infer(points)

Requires:
    - pcdet (built from VoxelNeXt repo via `python setup.py develop`)
    - spconv-cu121
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Class mapping: nuScenes 10-class → G1 3-class convention
# --------------------------------------------------------------------------- #
# nuScenes class order (0-indexed after label-1):
#   0=car, 1=truck, 2=construction_vehicle, 3=bus, 4=trailer,
#   5=barrier, 6=motorcycle, 7=bicycle, 8=pedestrian, 9=traffic_cone
#
# G1 convention:
#   0=car, 1=pedestrian, 2=cyclist

NUSCENES_CLASS_NAMES = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)

G1_CLASS_NAMES = ("car", "pedestrian", "cyclist")
G1_CLASS_COLORS = {
    "car": (0.0, 1.0, 1.0),
    "pedestrian": (1.0, 1.0, 0.0),
    "cyclist": (0.0, 1.0, 0.0),
}

# Map nuScenes label (1-indexed from model output) → G1 0-indexed label
# Vehicles (car, truck, construction_vehicle, bus, trailer) → car (0)
# motorcycle → cyclist (2)
# bicycle → cyclist (2)
# pedestrian → pedestrian (1)
# barrier, traffic_cone → -1 (discard)
_NUSCENES_TO_G1 = {
    1: 0,   # car → car
    2: 0,   # truck → car
    3: 0,   # construction_vehicle → car
    4: 0,   # bus → car
    5: 0,   # trailer → car
    6: -1,  # barrier → discard
    7: 2,   # motorcycle → cyclist
    8: 2,   # bicycle → cyclist
    9: 1,   # pedestrian → pedestrian
    10: -1, # traffic_cone → discard
}


def mask_points_out_of_range(pc: np.ndarray, pc_range: list[float]) -> np.ndarray:
    """Filter points outside the detection range."""
    pc_range_arr = np.array(pc_range, dtype=np.float32)
    pc_range_arr[3:6] -= 0.01
    mask_x = (pc[:, 0] > pc_range_arr[0]) & (pc[:, 0] < pc_range_arr[3])
    mask_y = (pc[:, 1] > pc_range_arr[1]) & (pc[:, 1] < pc_range_arr[4])
    mask_z = (pc[:, 2] > pc_range_arr[2]) & (pc[:, 2] < pc_range_arr[5])
    return pc[mask_x & mask_y & mask_z]


# Shared aliases used across the livox_detection package
CLASS_NAMES = G1_CLASS_NAMES
CLASS_COLORS = G1_CLASS_COLORS


class VoxelNeXtBackend:
    """VoxelNeXt 3D detection backend using OpenPCDet framework.

    Args:
        checkpoint: Path to VoxelNeXt .pth checkpoint file.
        device: 'cuda' or 'cpu'.
        score_threshold: Minimum score to keep detections.
        offset_ground: Z-offset for Livox Mid-360 ground alignment (1.33m default).
        cfg_file: Path to VoxelNeXt YAML config (e.g. cbgs_voxel0075_voxelnext.yaml).
        voxelnext_dir: Path to the cloned VoxelNeXt directory.
    """

    def __init__(
        self,
        checkpoint: str = "",
        device: str = "cuda",
        score_threshold: float = 0.25,
        offset_ground: float = 1.33,
        cfg_file: str = "",
        voxelnext_dir: str = "",
    ):
        self.checkpoint = checkpoint
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.offset_ground = offset_ground
        self.cfg_file = cfg_file
        self.voxelnext_dir = voxelnext_dir
        self.model = None
        self.cfg = None
        self.logger = logging.getLogger("VoxelNeXtBackend")

        # VoxelNeXt uses 360° symmetric range by default (better for Mid-360)
        self.point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

    def load(self) -> None:
        """Load VoxelNeXt model from checkpoint."""
        # Ensure VoxelNeXt's pcdet is importable
        if self.voxelnext_dir:
            vn_dir = str(Path(self.voxelnext_dir).resolve())
            if vn_dir not in sys.path:
                sys.path.insert(0, vn_dir)

        try:
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.datasets import DatasetTemplate
            from pcdet.models import build_network, load_data_to_gpu
            from pcdet.utils import common_utils
        except ImportError as e:
            raise ImportError(
                f"Cannot import pcdet. Build VoxelNeXt first: "
                f"cd {self.voxelnext_dir} && python setup.py develop. Error: {e}"
            )

        # Load config (switch to tools/ directory so relative _BASE_CONFIG_ resolves)
        cfg_path = Path(self.cfg_file).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"VoxelNeXt config not found: {cfg_path}")

        import os
        orig_cwd = os.getcwd()
        tools_dir = Path(self.voxelnext_dir) / "tools"
        if tools_dir.exists():
            os.chdir(str(tools_dir))

        try:
            cfg_from_yaml_file(str(cfg_path), cfg)
            self.cfg = cfg
        finally:
            os.chdir(orig_cwd)

        # Build Demo Dataset wrapper for model initialization and data collation
        class _LivoxDemoDataset(DatasetTemplate):
            def __init__(self, dataset_cfg, class_names, training=False):
                super().__init__(
                    dataset_cfg=dataset_cfg, class_names=class_names, training=training
                )

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {}

        self.dataset = _LivoxDemoDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
        )

        # Build model
        self.model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=self.dataset,
        )

        # Load checkpoint
        ckpt_path = Path(self.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"VoxelNeXt checkpoint not found: {ckpt_path}")

        self.model.load_params_from_file(filename=str(ckpt_path), logger=self.logger, to_cpu=True)
        self.model.to(self.device).eval()

        self.logger.info(
            f"VoxelNeXt loaded from {ckpt_path} "
            f"(config: {cfg_path.name}, device: {self.device}, "
            f"threshold: {self.score_threshold})"
        )

    def infer(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run VoxelNeXt inference on (N, 4) point cloud [x, y, z, intensity].

        Returns:
            boxes: (M, 7) float32 — [cx, cy, cz, dx, dy, dz, yaw]
            scores: (M,) float32 — confidence scores
            labels: (M,) int64 — G1 class indices (0=car, 1=pedestrian, 2=cyclist)
        """
        empty = (
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

        if self.model is None or points.shape[0] == 0:
            return empty

        # 1. Preprocess: ensure 5 columns [x, y, z, intensity, timestamp], apply ground offset
        pts = points.copy()
        if pts.shape[1] < 4:
            pts = np.hstack([pts[:, :3], np.zeros((pts.shape[0], 1), dtype=np.float32)])
        if pts.shape[1] == 4:
            # Add 5th feature (timestamp=0.0) expected by nuScenes point_feature_encoder
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)])

        pts[:, 2] += self.offset_ground

        # 2. Mask to point cloud range
        pts_masked = mask_points_out_of_range(pts, self.point_cloud_range)
        if pts_masked.shape[0] == 0:
            self.logger.warning("[VoxelNeXt] All points masked out of range.")
            return empty

        # 3. Format input_dict and prepare via OpenPCDet pipeline
        input_dict = {
            "points": pts_masked,
            "frame_id": 0,
        }

        try:
            from pcdet.models import load_data_to_gpu
            data_dict = self.dataset.prepare_data(data_dict=input_dict)
            data_dict = self.dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            # 4. Forward pass
            with torch.no_grad():
                pred_dicts, _ = self.model(data_dict)
        except Exception as e:
            self.logger.error(f"VoxelNeXt forward pass failed: {e}")
            return empty

        if not pred_dicts or len(pred_dicts) == 0:
            return empty

        # 5. Extract predictions (boxes can be [K, 7] or [K, 9] for nuScenes with velocity)
        pred = pred_dicts[0]
        boxes_pred = pred["pred_boxes"].cpu().numpy()    # (K, 7 or 9)
        scores_pred = pred["pred_scores"].cpu().numpy()  # (K,)
        labels_pred = pred["pred_labels"].cpu().numpy()  # (K,) 1-indexed

        if boxes_pred.shape[0] == 0:
            return empty

        # Slice to standard 7D box: [x, y, z, dx, dy, dz, yaw]
        if boxes_pred.shape[1] > 7:
            boxes_pred = boxes_pred[:, :7]

        # 6. Filter by score threshold
        keep = scores_pred >= self.score_threshold
        boxes_pred = boxes_pred[keep]
        scores_pred = scores_pred[keep]
        labels_pred = labels_pred[keep]

        if boxes_pred.shape[0] == 0:
            return empty

        # 7. Map nuScenes labels to G1 convention, discard unmapped classes
        g1_labels = np.array(
            [_NUSCENES_TO_G1.get(int(lbl), -1) for lbl in labels_pred],
            dtype=np.int64,
        )
        valid = g1_labels >= 0
        boxes_pred = boxes_pred[valid]
        scores_pred = scores_pred[valid]
        g1_labels = g1_labels[valid]

        if boxes_pred.shape[0] == 0:
            return empty

        # 8. Undo ground offset on z center → return boxes in raw sensor coords
        boxes_pred[:, 2] -= self.offset_ground

        return (
            boxes_pred.astype(np.float32),
            scores_pred.astype(np.float32),
            g1_labels,
        )
