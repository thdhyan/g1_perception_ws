#!/usr/bin/env python3
"""ROS2 node: LiDAR point cloud -> PointPillar 3D detections + RViz markers.

PointPillar is a pillar-based 3D object detector (Lang et al., CVPR 2019) that
voxelizes points into pillars, encodes them, scatters into a pseudo-image, and
runs a 2D CNN + SSD head. Faster than CenterPoint, suitable for edge inference.

Example:
    ros2 run g1_perception pointpillar_node --checkpoint pt/pointpillar_model.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

from .pointpillar_model import PointPillar

# Same class ordering as G1_sim backend
CLASS_NAMES = ("car", "pedestrian", "cyclist")
CLASS_COLORS = {
    "car": (0.0, 1.0, 1.0),
    "pedestrian": (1.0, 1.0, 0.0),
    "cyclist": (0.0, 1.0, 0.0),
}

# PointPillar config
VOXEL_SIZE = (0.2, 0.2, 0.2)
POINT_CLOUD_RANGE = (0.0, -44.8, -2.0, 224.0, 44.8, 4.0)


def voxelize_points(
    points: np.ndarray,
    voxel_size: tuple[float, float, float],
    point_cloud_range: tuple[float, ...],
    max_points_per_pillar: int = 32,
    max_pillars: int = 12000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize point cloud into pillars.

    Args:
        points: (N, 4) of x, y, z, intensity
        voxel_size: (vx, vy, vz)
        point_cloud_range: (xmin, ymin, zmin, xmax, ymax, zmax)
        max_points_per_pillar: max points to keep per pillar
        max_pillars: max pillars to keep

    Returns:
        pillars: (num_pillars, max_points, 4)
        pillar_indices: (num_pillars, 3) of voxel grid indices
        num_voxels: (num_pillars,) count of valid points
    """
    if points.shape[0] == 0:
        return (
            np.zeros((0, max_points_per_pillar, 4), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    # Filter to point cloud range
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

    # Compute voxel indices
    voxel_indices = (
        (points[:, :3] - np.array(point_cloud_range[:3])) / np.array(voxel_size)
    ).astype(np.int64)

    # Grid dimensions
    grid_size = (
        int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
        int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
        int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
    )

    # Hash to pillar index
    hash_dict = {}
    pillar_list = []
    pillar_indices_list = []
    num_voxels_list = []

    for i, (x_idx, y_idx, z_idx) in enumerate(voxel_indices):
        # Only care about x, y for pillars; z is the feature dimension
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

    # Convert to arrays
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


class PointPillarBackend:
    """PointPillar inference backend."""

    def __init__(self, checkpoint: str = "", device: str = "cuda", score_threshold: float = 0.4):
        self.checkpoint = checkpoint
        self.device = device
        self.score_threshold = score_threshold
        self.model = None
        self.logger = logging.getLogger("pointpillar_backend")

    def load(self) -> None:
        """Load model checkpoint or fall back to clustering."""
        checkpoint_path = Path(self.checkpoint)

        if checkpoint_path.exists():
            self.logger.info(f"Loading PointPillar checkpoint: {checkpoint_path}")
            self.model = PointPillar(num_classes=len(CLASS_NAMES), num_anchors=2)
            try:
                state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state)
                self.model.to(self.device)
                self.model.eval()
                self.logger.info("PointPillar model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {e}")
                self.logger.warning("Falling back to clustering backend")
                self.model = None
        else:
            self.logger.warning(
                f"Checkpoint not found: {checkpoint_path}. "
                "Falling back to simple Euclidean clustering."
            )
            self.model = None

    def infer(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run detection on points.

        Args:
            points: (N, 4) of x, y, z, intensity

        Returns:
            boxes: (M, 7) of x, y, z, dx, dy, dz, yaw
            scores: (M,) confidences
            labels: (M,) class indices
        """
        if self.model is None:
            return self._clustering_fallback(points)

        try:
            return self._infer_with_model(points)
        except Exception as e:
            self.logger.error(f"Inference failed: {e}, falling back to clustering")
            return self._clustering_fallback(points)

    def _infer_with_model(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run PointPillar model inference."""
        if points.shape[0] == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        # Voxelize
        pillars, pillar_indices, num_voxels = voxelize_points(
            points, VOXEL_SIZE, POINT_CLOUD_RANGE
        )

        if pillars.shape[0] == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        # Inference
        with torch.no_grad():
            pillars_t = torch.from_numpy(pillars).to(self.device)
            indices_t = torch.from_numpy(pillar_indices).to(self.device)
            num_voxels_t = torch.from_numpy(num_voxels).to(self.device)

            h = int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1])
            w = int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0])

            hm, reg = self.model(pillars_t, indices_t, num_voxels_t, (h, w))

            # Simple post-processing: NMS over heatmap peaks
            boxes, scores, labels = self._decode_predictions(hm, reg, h, w)

        return boxes, scores, labels

    def _decode_predictions(
        self, hm: torch.Tensor, reg: torch.Tensor, h: int, w: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode heatmap and regression to boxes.

        Minimal decoding: find peaks in heatmap, gather regression targets.
        """
        # Reshape: (1, K*C, H, W) -> (1, H, W, K, C)
        num_anchors = 2
        num_classes = len(CLASS_NAMES)

        hm = hm.view(1, h, w, num_anchors, num_classes)
        reg = reg.view(1, h, w, num_anchors, 7)

        hm = torch.sigmoid(hm)
        hm_np = hm[0].cpu().numpy()
        reg_np = reg[0].cpu().numpy()

        boxes, scores, labels = [], [], []

        # Iterate each class
        for cls_idx in range(num_classes):
            # Simple peak detection: local maxima
            hm_cls = hm_np[..., cls_idx]  # (H, W, num_anchors)
            for anchor_idx in range(num_anchors):
                score_map = hm_cls[:, :, anchor_idx]
                # Use threshold
                peaks = score_map > self.score_threshold
                if not peaks.any():
                    continue

                for y, x in zip(*np.where(peaks)):
                    score = float(score_map[y, x])
                    box = reg_np[y, x, anchor_idx]  # (7,)

                    # Convert grid coordinates to world
                    center_x = x * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0]
                    center_y = y * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1]

                    box_world = np.array(
                        [center_x, center_y, box[2], box[3], box[4], box[5], box[6]],
                        dtype=np.float32,
                    )
                    boxes.append(box_world)
                    scores.append(score)
                    labels.append(cls_idx)

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

    def _clustering_fallback(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple Euclidean clustering fallback (same as G1_sim backend)."""
        if points.shape[0] == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        xyz = points[:, :3]
        radial = np.linalg.norm(xyz[:, :2], axis=1)
        keep = (
            (xyz[:, 2] > -1.6)
            & (xyz[:, 2] < 1.2)
            & (radial < 25.0)
            & (radial > 0.5)
        )
        xyz = xyz[keep]
        if xyz.shape[0] < 12:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        grid = np.floor(xyz[:, :2] / 0.2).astype(np.int64)
        _, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)

        boxes, scores, labels = [], [], []
        for cell in np.flatnonzero(counts >= 12):
            member = xyz[inverse == cell]
            lo, hi = member.min(axis=0), member.max(axis=0)
            extent = hi - lo
            height = float(extent[2])
            footprint = float(max(extent[0], extent[1]))

            if not (0.8 < height < 2.2 and footprint < 1.2 and height > footprint):
                continue

            centre = (lo + hi) / 2.0
            boxes.append([centre[0], centre[1], centre[2], max(extent[0], 0.3), max(extent[1], 0.3), height, 0.0])
            scores.append(0.5)
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


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Yaw-only rotation as (x, y, z, w)."""
    return (0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0)))


class PointPillarDetectionNode(Node):
    def __init__(self, backend: PointPillarBackend, input_topic: str, frame_override: str | None, max_hz: float):
        super().__init__("g1_pointpillar_detection")
        self.backend = backend
        self.frame_override = frame_override
        self.max_hz = max_hz
        self.last_inference_time = 0.0

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.detection_pub = self.create_publisher(Detection3DArray, "/g1/detections/pointpillar", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/g1/detection_markers/pointpillar", 10)
        self.create_subscription(PointCloud2, input_topic, self.on_cloud, sensor_qos)

        self._busy = False
        self.get_logger().info(f"backend={backend.__class__.__name__} listening on {input_topic} @ {max_hz} Hz")

    def on_cloud(self, msg: PointCloud2) -> None:
        if self._busy:
            return

        # Rate limiting
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_inference_time < 1.0 / self.max_hz:
            return

        self._busy = True
        try:
            points = self.cloud_to_array(msg)
            boxes, scores, labels = self.backend.infer(points)
            frame = self.frame_override or msg.header.frame_id
            self.detection_pub.publish(self.to_detection_array(boxes, scores, labels, msg.header.stamp, frame))
            self.marker_pub.publish(self.to_markers(boxes, labels, msg.header.stamp, frame))
            self.last_inference_time = now
        except Exception as exc:
            self.get_logger().error(f"detection failed: {exc}")
        finally:
            self._busy = False

    @staticmethod
    def cloud_to_array(msg: PointCloud2) -> np.ndarray:
        available = {f.name for f in msg.fields}
        fields = ["x", "y", "z"] + (["intensity"] if "intensity" in available else [])
        raw = point_cloud2.read_points(msg, field_names=fields, skip_nans=True)
        if raw.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32)

        points = np.stack([raw[name] for name in fields], axis=-1).astype(np.float32)
        if points.shape[1] == 3:
            points = np.hstack([points, np.zeros((points.shape[0], 1), dtype=np.float32)])
        return points

    def to_detection_array(self, boxes, scores, labels, stamp, frame: str) -> Detection3DArray:
        array = Detection3DArray()
        array.header.stamp = stamp
        array.header.frame_id = frame

        for box, score, label in zip(boxes, scores, labels):
            detection = Detection3D()
            detection.header = array.header

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = CLASS_NAMES[int(label)]
            hypothesis.hypothesis.score = float(score)
            detection.results.append(hypothesis)

            bbox = BoundingBox3D()
            bbox.center.position.x = float(box[0])
            bbox.center.position.y = float(box[1])
            bbox.center.position.z = float(box[2])
            qx, qy, qz, qw = yaw_to_quaternion(float(box[6]))
            bbox.center.orientation.x = qx
            bbox.center.orientation.y = qy
            bbox.center.orientation.z = qz
            bbox.center.orientation.w = qw
            bbox.size.x = float(box[3])
            bbox.size.y = float(box[4])
            bbox.size.z = float(box[5])
            detection.bbox = bbox

            array.detections.append(detection)
        return array

    def to_markers(self, boxes, labels, stamp, frame: str) -> MarkerArray:
        markers = MarkerArray()

        clear = Marker()
        clear.header.frame_id = frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for i, (box, label) in enumerate(zip(boxes, labels)):
            name = CLASS_NAMES[int(label)]
            red, green, blue = CLASS_COLORS[name]

            marker = Marker()
            marker.header.frame_id = frame
            marker.header.stamp = stamp
            marker.ns = "detections"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(box[0])
            marker.pose.position.y = float(box[1])
            marker.pose.position.z = float(box[2])
            qx, qy, qz, qw = yaw_to_quaternion(float(box[6]))
            marker.pose.orientation.x = qx
            marker.pose.orientation.y = qy
            marker.pose.orientation.z = qz
            marker.pose.orientation.w = qw
            marker.scale.x = float(box[3])
            marker.scale.y = float(box[4])
            marker.scale.z = float(box[5])
            marker.color.r, marker.color.g, marker.color.b = red, green, blue
            marker.color.a = 0.4
            markers.markers.append(marker)

        return markers


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 PointPillar 3D detection node.")
    parser.add_argument("--checkpoint", default="G1_sim/detection/pt/pointpillar_model.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--input-topic", default="/livox/mid360/points")
    parser.add_argument("--max-hz", type=float, default=10.0)
    parser.add_argument("--frame", default=None, help="Override the cloud's frame_id.")
    args = parser.parse_args()

    backend = PointPillarBackend(
        checkpoint=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold,
    )
    try:
        backend.load()
    except Exception as exc:
        print(f"[pointpillar] failed to load backend: {exc}", file=sys.stderr)
        raise SystemExit(1)

    rclpy.init()
    node = PointPillarDetectionNode(backend, args.input_topic, args.frame, args.max_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
