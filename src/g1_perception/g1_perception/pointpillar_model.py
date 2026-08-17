"""PointPillar 3D object detector (Lang et al., CVPR 2019).

Inference-only implementation. Architecture: pillar voxelization + feature
encoding + 2D CNN backbone + SSD-style detection head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PillarFeatureNet(nn.Module):
    """Encode points in each pillar into a feature vector."""

    def __init__(
        self,
        in_channels: int = 4,  # x, y, z, intensity
        feat_channels: int = 64,
        voxel_size: tuple[float, float, float] = (0.2, 0.2, 0.2),
        point_cloud_range: tuple[float, ...] = (0.0, -44.8, -2.0, 224.0, 44.8, 4.0),
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)

        # Input: x, y, z, intensity, dx, dy, dz (offsets from pillar center)
        num_input_features = in_channels + 3
        self.linear1 = nn.Linear(num_input_features, feat_channels)
        self.norm = nn.BatchNorm1d(feat_channels)

    def forward(
        self, pillars: torch.Tensor, pillar_indices: torch.Tensor, num_voxels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pillars: (N, max_points, 4) of x, y, z, intensity
            pillar_indices: (N, 3) of pillar voxel indices [x_idx, y_idx, z_idx]
            num_voxels: (N,) number of valid points in each pillar

        Returns:
            pillar_features: (N, feat_channels)
            pillar_indices: (N, 3)
        """
        # Get pillar centers in world coordinates
        pillar_centers = pillar_indices.float() * torch.tensor(
            self.voxel_size, device=pillars.device, dtype=torch.float32
        ) + torch.tensor(
            self.point_cloud_range[:3], device=pillars.device, dtype=torch.float32
        )  # (N, 3)

        # Compute offsets from pillar center
        points_xyz = pillars[..., :3]  # (N, max_points, 3)
        center_offsets = points_xyz - pillar_centers.unsqueeze(1)  # (N, max_points, 3)

        # Concatenate with intensity
        feats = torch.cat([pillars, center_offsets], dim=-1)  # (N, max_points, 7)

        # Flatten and encode
        n_pillars, n_points, n_features = feats.shape
        feats_flat = feats.view(-1, n_features)  # (N*max_points, 7)

        # Create mask for valid points
        mask = torch.arange(n_points, device=pillars.device).unsqueeze(0) < num_voxels.unsqueeze(1)
        mask_flat = mask.view(-1)  # (N*max_points,)

        # Encode only valid points
        feats_encoded = torch.zeros(
            n_pillars * n_points, self.feat_channels, dtype=pillars.dtype, device=pillars.device
        )
        feats_encoded[mask_flat] = self.norm(self.linear1(feats_flat[mask_flat]))
        feats_encoded = feats_encoded.view(n_pillars, n_points, self.feat_channels)

        # Max pool over points in each pillar
        feats_encoded[~mask] = -1e6
        pillar_features = feats_encoded.max(dim=1)[0]  # (N, feat_channels)

        return pillar_features, pillar_indices.long()


class Backbone2D(nn.Module):
    """2D CNN backbone: stride-2 blocks upsample, then concatenate and fuse."""

    def __init__(self, in_channels: int = 64, layer_nums: list[int] | None = None):
        super().__init__()
        if layer_nums is None:
            layer_nums = [3, 5, 5]
        self.in_channels = in_channels

        # Downsample path: 5 stride-2 blocks
        self.blocks = nn.ModuleList()
        channels = [in_channels, 64, 128, 256]
        for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(
                            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(inplace=True),
                        )
                        for _ in range(layer_nums[i])
                    ],
                )
            )

        # Upsample and fuse
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample
        b0 = self.blocks[0](x)
        b1 = self.blocks[1](b0)
        b2 = self.blocks[2](b1)

        # Upsample and concatenate
        u0 = self.deconv1(b2)
        u1 = self.deconv2(u0 + b1)
        u2 = self.deconv3(u1 + b0)
        u3 = self.deconv4(u2 + x)

        return u3


class SSDHead(nn.Module):
    """SSD-style detection head: shared conv -> multi-task heads."""

    def __init__(self, in_channels: int = 64, num_anchors: int = 2, num_classes: int = 3):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Shared
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Task heads
        self.hm_head = nn.Conv2d(128, num_anchors * num_classes, 1)
        self.reg_head = nn.Conv2d(128, num_anchors * 7, 1)  # x, y, z, dx, dy, dz, yaw

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        hm = self.hm_head(shared)  # (B, K*C, H, W)
        reg = self.reg_head(shared)  # (B, K*7, H, W)
        return hm, reg


class PointPillar(nn.Module):
    """PointPillar detector: end-to-end inference."""

    def __init__(
        self,
        voxel_size: tuple[float, float, float] = (0.2, 0.2, 0.2),
        point_cloud_range: tuple[float, ...] = (0.0, -44.8, -2.0, 224.0, 44.8, 4.0),
        num_classes: int = 3,
        num_anchors: int = 2,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.pillar_feature_net = PillarFeatureNet(
            in_channels=4,
            feat_channels=64,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )
        self.backbone = Backbone2D(in_channels=64)
        self.head = SSDHead(in_channels=64, num_anchors=num_anchors, num_classes=num_classes)

    def forward(
        self,
        pillars: torch.Tensor,
        pillar_indices: torch.Tensor,
        num_voxels: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pillars: (N, max_points, 4) point features per pillar
            pillar_indices: (N, 3) voxel grid indices
            num_voxels: (N,) number of valid points per pillar
            spatial_shape: (height, width) of pseudo-image

        Returns:
            heatmap: (1, K*C, H, W)
            regression: (1, K*7, H, W)
        """
        # Encode pillars into feature vectors
        pillar_features, pillar_indices = self.pillar_feature_net(pillars, pillar_indices, num_voxels)

        # Scatter features into pseudo-image
        h, w = spatial_shape
        pseudo_image = torch.zeros(
            (1, 64, h, w), dtype=pillar_features.dtype, device=pillar_features.device
        )
        for i, (x_idx, y_idx, _) in enumerate(pillar_indices):
            x_idx, y_idx = int(x_idx), int(y_idx)
            if 0 <= x_idx < w and 0 <= y_idx < h:
                pseudo_image[0, :, y_idx, x_idx] = pillar_features[i]

        # Backbone + head
        backbone_feat = self.backbone(pseudo_image)
        hm, reg = self.head(backbone_feat)

        return hm, reg
