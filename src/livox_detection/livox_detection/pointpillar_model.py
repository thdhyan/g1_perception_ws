#!/usr/bin/env python3
"""PointPillar 3D Detection Model & Backend for Livox Mid-360 LiDAR.

Ported and self-contained for the Unitree G1 humanoid robot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ("car", "pedestrian", "cyclist")
CLASS_COLORS = {
    "car": (0.0, 1.0, 1.0),
    "pedestrian": (1.0, 1.0, 0.0),
    "cyclist": (0.0, 1.0, 0.0),
}

VOXEL_SIZE = (0.2, 0.2, 0.2)
POINT_CLOUD_RANGE = (0.0, -44.8, -2.0, 224.0, 44.8, 4.0)


def voxelize_points(
    points: np.ndarray,
    voxel_size: tuple[float, float, float] = VOXEL_SIZE,
    point_cloud_range: tuple[float, ...] = POINT_CLOUD_RANGE,
    max_points_per_pillar: int = 32,
    max_pillars: int = 12000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize point cloud into pillars."""
    if points.shape[0] == 0:
        return (
            np.zeros((0, max_points_per_pillar, 4), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    mask = (
        (points[:, 0] >= point_cloud_range[0])
        & (points[:, 0] < point_cloud_range[3])
        & (points[:, 1] >= point_cloud_range[1])
        & (points[:, 1] < point_cloud_range[4])
        & (points[:, 2] >= point_cloud_range[2])
        & (points[:, 2] < point_cloud_range[5])
    )
    points = points[mask]

    if points.shape[0] == 0:
        return (
            np.zeros((0, max_points_per_pillar, 4), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    voxel_indices = (
        (points[:, :3] - np.array(point_cloud_range[:3])) / np.array(voxel_size)
    ).astype(np.int64)

    grid_size = (
        int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
        int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
    )

    hash_dict = {}
    pillar_list = []
    pillar_indices_list = []
    num_voxels_list = []

    for i, (x_idx, y_idx, _) in enumerate(voxel_indices):
        key = (int(x_idx), int(y_idx))
        if 0 <= key[0] < grid_size[0] and 0 <= key[1] < grid_size[1]:
            if key not in hash_dict:
                if len(hash_dict) >= max_pillars:
                    break
                hash_dict[key] = len(pillar_list)
                pillar_list.append([])
                pillar_indices_list.append(np.array([key[0], key[1], 0]))
            pillar_idx = hash_dict[key]
            if len(pillar_list[pillar_idx]) < max_points_per_pillar:
                pillar_list[pillar_idx].append(points[i])

    if not pillar_list:
        return (
            np.zeros((0, max_points_per_pillar, 4), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    pillars_array = np.zeros((len(pillar_list), max_points_per_pillar, 4), dtype=np.float32)
    for i, pillar in enumerate(pillar_list):
        for j, point in enumerate(pillar):
            pillars_array[i, j] = point
        num_voxels_list.append(len(pillar))

    return (
        pillars_array,
        np.array(pillar_indices_list, dtype=np.int64),
        np.array(num_voxels_list, dtype=np.int64),
    )


class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels: int = 4, feat_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, feat_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(feat_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pillars: torch.Tensor, num_voxels: torch.Tensor) -> torch.Tensor:
        # pillars: (P, N, C) -> permute to (P, C, N)
        x = pillars.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # Max pool across points in pillar
        x, _ = torch.max(x, dim=2)  # (P, feat_channels)
        return x


class PointPillarScatter(nn.Module):
    def __init__(self, feat_channels: int = 64):
        super().__init__()
        self.feat_channels = feat_channels

    def forward(self, pillar_feats: torch.Tensor, pillar_indices: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        h, w = grid_size
        batch_canvas = torch.zeros((1, self.feat_channels, h, w), dtype=pillar_feats.dtype, device=pillar_feats.device)
        if pillar_feats.shape[0] == 0:
            return batch_canvas
        x_idx = pillar_indices[:, 0]
        y_idx = pillar_indices[:, 1]
        valid = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
        batch_canvas[0, :, y_idx[valid], x_idx[valid]] = pillar_feats[valid].t()
        return batch_canvas


class PointPillarBackbone(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 128):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=4, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.block1(x)
        b2 = self.block2(b1)
        u1 = self.deconv1(b1)
        u2 = self.deconv2(b2)
        return torch.cat([u1, u2], dim=1)  # 128 channels


class PointPillarHead(nn.Module):
    def __init__(self, in_channels: int = 128, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.conv_cls = nn.Conv2d(in_channels, num_classes, 1)
        self.conv_box = nn.Conv2d(in_channels, 7, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.conv_cls(x), self.conv_box(x)


class PointPillarNet(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.pfn = PillarFeatureNet(4, 64)
        self.scatter = PointPillarScatter(64)
        self.backbone = PointPillarBackbone(64, 128)
        self.head = PointPillarHead(128, num_classes)

    def forward(self, pillars: torch.Tensor, indices: torch.Tensor, num_voxels: torch.Tensor, grid_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        pfeats = self.pfn(pillars, num_voxels)
        canvas = self.scatter(pfeats, indices, grid_size)
        feat = self.backbone(canvas)
        cls_preds, box_preds = self.head(feat)
        return cls_preds, box_preds


class PointPillarBackend:
    """Runs PointPillars inference with automatic Euclidean clustering fallback."""

    def __init__(
        self,
        checkpoint: str = "",
        device: str = "cuda",
        score_threshold: float = 0.4,
    ):
        self.checkpoint = checkpoint
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.model: PointPillarNet | None = None
        self.logger = logging.getLogger("PointPillarBackend")

    def load(self) -> None:
        ckpt_path = Path(self.checkpoint) if self.checkpoint else None
        if ckpt_path and ckpt_path.exists():
            try:
                state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                state = state.get("model_state_dict", state)
                self.model = PointPillarNet()
                self.model.load_state_dict(state, strict=False)
                self.model.to(self.device).eval()
                self.logger.info(f"PointPillar checkpoint loaded from {ckpt_path}")
                return
            except Exception as e:
                self.logger.warning(f"Could not load PointPillar weights ({e}); using clustering fallback.")
        self.logger.info("Using Euclidean Clustering for PointPillar detection.")
        self.model = None

    def infer(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            return self._clustering_fallback(points)

        try:
            return self._infer_with_model(points)
        except Exception as e:
            self.logger.error(f"PointPillar inference error: {e}, falling back to clustering.")
            return self._clustering_fallback(points)

    def _infer_with_model(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pillars, pillar_indices, num_voxels = voxelize_points(points)
        if pillars.shape[0] == 0:
            return self._clustering_fallback(points)

        h = int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1])
        w = int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0])

        with torch.no_grad():
            pillars_t = torch.from_numpy(pillars).to(self.device)
            indices_t = torch.from_numpy(pillar_indices).to(self.device)
            num_voxels_t = torch.from_numpy(num_voxels).to(self.device)

            cls_preds, box_preds = self.model(pillars_t, indices_t, num_voxels_t, (h, w))
            scores = torch.sigmoid(cls_preds[0])
            scores_np = scores.cpu().numpy()
            boxes_np = box_preds[0].cpu().numpy()

        boxes, out_scores, labels = [], [], []
        num_classes = len(CLASS_NAMES)
        for c in range(num_classes):
            score_map = scores_np[c]
            peaks = score_map > self.score_threshold
            if not peaks.any():
                continue
            for y, x in zip(*np.where(peaks)):
                score = float(score_map[y, x])
                box = boxes_np[:, y, x]
                cx = x * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0]
                cy = y * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1]
                boxes.append([cx, cy, float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[6])])
                out_scores.append(score)
                labels.append(c)

        if not boxes:
            return self._clustering_fallback(points)

        return (
            np.array(boxes, dtype=np.float32),
            np.array(out_scores, dtype=np.float32),
            np.array(labels, dtype=np.int64),
        )

    def _clustering_fallback(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Euclidean clustering fallback to detect standing pedestrians.

        Uses a coarse XY grid + connected-component flood-fill to cluster
        nearby points, then applies human-body geometric filters.
        """
        if points.shape[0] == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        xyz = points[:, :3]
        radial = np.linalg.norm(xyz[:, :2], axis=1)
        # Filter: keep points within a reasonable human detection range
        # Height: -1.6 to 2.0 m (relative to sensor), Distance: 0.3 to 20 m
        keep = (
            (xyz[:, 2] > -1.6)
            & (xyz[:, 2] < 2.0)
            & (radial < 20.0)
            & (radial > 0.3)
        )
        xyz = xyz[keep]
        if xyz.shape[0] < 5:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        # Coarse XY grid (0.5m cells) for spatial grouping
        cell_size = 0.5
        grid_xy = np.floor(xyz[:, :2] / cell_size).astype(np.int64)
        # Offset to avoid negative indices
        grid_xy -= grid_xy.min(axis=0)

        # Build occupied cell map: cell_key -> list of point indices
        cell_map = {}
        for i in range(xyz.shape[0]):
            key = (int(grid_xy[i, 0]), int(grid_xy[i, 1]))
            cell_map.setdefault(key, []).append(i)

        # Connected-component flood-fill over occupied cells
        visited = set()
        clusters = []

        for seed_key in cell_map:
            if seed_key in visited:
                continue
            # BFS flood-fill to neighboring cells (8-connected)
            cluster_indices = []
            queue = [seed_key]
            visited.add(seed_key)
            while queue:
                curr = queue.pop()
                cluster_indices.extend(cell_map[curr])
                cx, cy = curr
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nb = (cx + dx, cy + dy)
                        if nb in cell_map and nb not in visited:
                            visited.add(nb)
                            queue.append(nb)

            if len(cluster_indices) >= 5:
                clusters.append(np.array(cluster_indices))

        # Evaluate each cluster for human-body geometry
        boxes, scores, labels = [], [], []
        if len(clusters) > 0:
            self.logger.debug(f"[Clustering] Found {len(clusters)} raw clusters from flood-fill.")

        for idx_arr in clusters:
            cluster_pts = xyz[idx_arr]
            lo = cluster_pts.min(axis=0)
            hi = cluster_pts.max(axis=0)
            extent = hi - lo
            height = float(extent[2])
            width_x = float(extent[0])
            width_y = float(extent[1])
            footprint = max(width_x, width_y)
            num_pts = len(idx_arr)

            # Human-body geometric filter:
            if not (0.4 < height < 2.3 and footprint < 2.0 and num_pts >= 4):
                continue

            centre = (lo + hi) / 2.0
            height_ratio = min(height / 1.7, 1.0)
            shape_score = height_ratio * min(1.0, height / max(footprint, 0.1))
            conf = float(np.clip(0.5 + 0.4 * shape_score, 0.4, 0.95))

            boxes.append([
                centre[0], centre[1], centre[2],
                max(width_x, 0.3), max(width_y, 0.3), height, 0.0
            ])
            scores.append(conf)
            labels.append(CLASS_NAMES.index("pedestrian"))


        if not boxes:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        return (
            np.array(boxes, dtype=np.float32),
            np.array(scores, dtype=np.float32),
            np.array(labels, dtype=np.int64),
        )

