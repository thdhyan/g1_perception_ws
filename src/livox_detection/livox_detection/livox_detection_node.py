#!/usr/bin/env python3
"""ROS 2 Node: Livox LiDAR -> 3D Object & Human Detection in Pelvis Frame.

VoxelNeXt backend (OpenPCDet architecture).
Subscribes to `/livox/lidar` (supports PointCloud2 and Livox CustomMsg).
Transforms detected 3D bounding boxes and coordinates into the robot's `pelvis` frame using TF2.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Ensure .venv site-packages is available for torch/torchvision/onnxruntime
_here = Path(__file__).resolve()
for _ancestor in _here.parents:
    _candidate = _ancestor / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break

import numpy as np
import rclpy

from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformListener, TransformException
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

from .voxelnext_model import CLASS_COLORS, CLASS_NAMES, VoxelNeXtBackend


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Yaw rotation as (x, y, z, w)."""
    return (0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0)))


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix."""
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


class LivoxDetectionNode(Node):
    """ROS 2 Node for 3D human and object detection transformed to robot pelvis frame."""

    def __init__(self):
        super().__init__("g1_livox_detection")

        # Declare parameters
        self.declare_parameter("algorithm", "voxelnext")  # "voxelnext" — only supported backend
        # VoxelNeXt-specific parameters
        _ws_root_str = str(Path(__file__).resolve().parents[3])
        self.declare_parameter(
            "voxelnext_cfg",
            str(Path(_ws_root_str) / "VoxelNeXt" / "tools" / "cfgs" / "nuscenes_models" / "cbgs_voxel0075_voxelnext.yaml"),
        )
        self.declare_parameter("voxelnext_dir", str(Path(_ws_root_str) / "VoxelNeXt"))
        self.declare_parameter(
            "checkpoint_path",
            str(Path(_ws_root_str) / "pt" / "voxelnext_nuscenes.pth"),
        )
        self.declare_parameter("max_hz", 10.0)
        self.declare_parameter("score_threshold", 0.10)
        self.declare_parameter("accumulate_frames", 4)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("input_topic", "/livox/lidar")
        self.declare_parameter("target_frame", "pelvis")
        self.declare_parameter("frame_override", "")
        self.declare_parameter("max_distance", 25.0)
        self.declare_parameter("offset_ground", 1.33)
        # Comma-separated class names to keep (must match CLASS_NAMES); empty = no filter
        self.declare_parameter("class_filter", "pedestrian")

        self.algorithm = self.get_parameter("algorithm").value.lower()
        if self.algorithm != "voxelnext":
            self.get_logger().warning(
                f"algorithm '{self.algorithm}' is no longer supported; using 'voxelnext'."
            )
            self.algorithm = "voxelnext"
        self.checkpoint_path = self.get_parameter("checkpoint_path").value
        self.max_hz = float(self.get_parameter("max_hz").value)
        self.score_threshold = float(self.get_parameter("score_threshold").value)
        self.accumulate_frames = int(self.get_parameter("accumulate_frames").value)
        self.device = self.get_parameter("device").value
        self.input_topic = self.get_parameter("input_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.frame_override = self.get_parameter("frame_override").value or None
        self.max_distance = float(self.get_parameter("max_distance").value)
        self.offset_ground = float(self.get_parameter("offset_ground").value)
        class_filter_str = self.get_parameter("class_filter").value.strip()
        self.class_filter = (
            {c.strip() for c in class_filter_str.split(",") if c.strip()}
            if class_filter_str else None
        )
        self.voxelnext_cfg = self.get_parameter("voxelnext_cfg").value
        self.voxelnext_dir = self.get_parameter("voxelnext_dir").value

        from collections import deque
        self._cloud_buffer = deque(maxlen=max(1, self.accumulate_frames))

        # Rate limiting
        self.last_inference_time = 0.0
        self.min_interval = 1.0 / self.max_hz if self.max_hz > 0 else 0.0
        self._busy = False
        self._frame_count = 0
        self._diag_interval = 10  # Log diagnostics every N frames

        # TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize detection backend
        self.backend = None
        self._init_backend()

        # BEST_EFFORT matches lidar_bridge and livox_ros_driver2 publishers
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )


        # Publishers
        self.detection_pub = self.create_publisher(
            Detection3DArray, "/g1/detections/livox", 10
        )
        self.detection_pub_alias = self.create_publisher(
            Detection3DArray, f"/g1/detections/{self.algorithm}", 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, "/g1/detection_markers/livox", 10
        )
        self.marker_pub_alias = self.create_publisher(
            MarkerArray, f"/g1/detection_markers/{self.algorithm}", 10
        )
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/livox/mid360/points", 10
        )


        # Subscribers: Try PointCloud2 first (standard for xfer_format=0)
        self.cloud_sub = self.create_subscription(
            PointCloud2, self.input_topic, self.on_pointcloud2, sensor_qos
        )

        # Optionally support Livox CustomMsg if driver is present
        self.custom_msg_sub = None
        try:
            from livox_ros_driver2.msg import CustomMsg
            self.custom_msg_sub = self.create_subscription(
                CustomMsg, self.input_topic, self.on_custom_msg, sensor_qos
            )
            self.get_logger().info("Livox CustomMsg subscriber registered.")
        except ImportError:
            self.get_logger().info("livox_ros_driver2 CustomMsg not found; using PointCloud2 interface.")

        self.get_logger().info(
            f"LivoxDetectionNode started [Algorithm: {self.algorithm}, Target Frame: {self.target_frame}, Topic: {self.input_topic}]"
        )

    def _init_backend(self) -> None:
        """Initialize the VoxelNeXt detection backend (OpenPCDet)."""
        try:
            self.backend = VoxelNeXtBackend(
                checkpoint=self.checkpoint_path,
                device=self.device,
                score_threshold=self.score_threshold,
                offset_ground=self.offset_ground,
                cfg_file=self.voxelnext_cfg,
                voxelnext_dir=self.voxelnext_dir,
            )
            self.backend.load()
            self.get_logger().info("Loaded backend: voxelnext")
        except Exception as e:
            self.backend = None
            self.get_logger().error(
                f"VoxelNeXt backend unavailable ({e}). Detection disabled — "
                f"build pcdet from VoxelNeXt/ (python setup.py develop) and install spconv-cu121."
            )


    def on_pointcloud2(self, msg: PointCloud2) -> None:
        """Process incoming PointCloud2 message."""
        if self._busy:
            return
        now = time.time()
        if now - self.last_inference_time < self.min_interval:
            return
        self.last_inference_time = now

        self._busy = True
        try:
            # Sanitize frame_id (map livox_frame -> mid360_link)
            frame_id = msg.header.frame_id
            if frame_id == "livox_frame" or not frame_id:
                frame_id = "mid360_link"
                msg.header.frame_id = frame_id

            points = self.cloud_to_array(msg)
            # Republish cloud for downstream consumers
            self.cloud_pub.publish(msg)
            self._process_and_publish(points, msg.header.stamp, frame_id)
        except Exception as e:
            self.get_logger().error(f"Detection on PointCloud2 failed: {e}")
        finally:
            self._busy = False

    def on_custom_msg(self, msg) -> None:
        """Process incoming Livox CustomMsg message."""
        if self._busy:
            return
        now = time.time()
        if now - self.last_inference_time < self.min_interval:
            return
        self.last_inference_time = now

        self._busy = True
        try:
            frame_id = msg.header.frame_id
            if frame_id == "livox_frame" or not frame_id:
                frame_id = "mid360_link"
                msg.header.frame_id = frame_id

            points = self.custom_msg_to_array(msg)
            # Convert and publish PointCloud2
            cloud_msg = self.custom_msg_to_cloud(msg, points)
            self.cloud_pub.publish(cloud_msg)
            self._process_and_publish(points, msg.header.stamp, frame_id)
        except Exception as e:
            self.get_logger().error(f"Detection on CustomMsg failed: {e}")
        finally:
            self._busy = False

    def _process_and_publish(self, points: np.ndarray, stamp, source_frame: str) -> None:
        """Run 3D detection and transform detections to target pelvis frame."""
        source_frame = self.frame_override or source_frame
        self._frame_count += 1

        if points.shape[0] > 0:
            self._cloud_buffer.append(points)

        if len(self._cloud_buffer) > 0:
            infer_points = np.vstack(self._cloud_buffer) if len(self._cloud_buffer) > 1 else self._cloud_buffer[0]
        else:
            infer_points = points

        if self.backend is None or infer_points.shape[0] == 0:
            boxes, scores, labels = np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        else:
            boxes, scores, labels = self.backend.infer(infer_points)
            if boxes.shape[0] > 0 and self.max_distance > 0:
                dist2d = np.hypot(boxes[:, 0], boxes[:, 1])
                keep = dist2d <= self.max_distance
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if boxes.shape[0] > 0 and self.class_filter is not None:
                keep = np.array([
                    (CLASS_NAMES[lbl] if 0 <= lbl < len(CLASS_NAMES) else None) in self.class_filter
                    for lbl in labels
                ], dtype=bool)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # Periodic diagnostic logging
        if self._frame_count % self._diag_interval == 1:
            max_score = float(scores.max()) if scores.size > 0 else 0.0
            class_dist = {}
            for lbl in labels:
                name = CLASS_NAMES[lbl] if 0 <= lbl < len(CLASS_NAMES) else f"cls{lbl}"
                class_dist[name] = class_dist.get(name, 0) + 1
            self.get_logger().info(
                f"[Frame {self._frame_count}] Points: {points.shape[0]:,d} | "
                f"Detections: {len(scores)} (max_score={max_score:.3f}) | "
                f"Classes: {class_dist} | Source: {source_frame}"
            )

        # Lookup transform from sensor frame to target frame (pelvis)
        transformed_boxes, output_frame = self._transform_boxes_to_target(boxes, source_frame, stamp)

        # Publish Detection3DArray in target_frame (pelvis)
        det_array = self.to_detection_array(transformed_boxes, scores, labels, stamp, output_frame)
        self.detection_pub.publish(det_array)
        self.detection_pub_alias.publish(det_array)

        # Publish visual markers in target_frame (pelvis)
        markers = self.to_markers(transformed_boxes, scores, labels, stamp, output_frame)
        self.marker_pub.publish(markers)
        self.marker_pub_alias.publish(markers)


    def _transform_boxes_to_target(
        self, boxes: np.ndarray, source_frame: str, stamp
    ) -> Tuple[np.ndarray, str]:
        """Transform (M, 7) boxes from source_frame to self.target_frame using TF2."""
        if boxes.shape[0] == 0:
            return boxes, (self.target_frame if self.target_frame else source_frame)

        if not self.target_frame or self.target_frame == source_frame:
            return boxes, source_frame

        try:
            # Lookup transform with 0.1s timeout
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
            t = tf.transform.translation
            q = tf.transform.rotation

            trans = np.array([t.x, t.y, t.z], dtype=np.float64)
            rot_mat = quaternion_to_matrix(q.x, q.y, q.z, q.w)

            # Compute yaw change from rotation matrix
            delta_yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])

            transformed = np.zeros_like(boxes)
            for i in range(boxes.shape[0]):
                center_src = boxes[i, :3].astype(np.float64)
                center_target = rot_mat @ center_src + trans
                transformed[i, :3] = center_target
                transformed[i, 3:6] = boxes[i, 3:6]  # Box dimensions (dx, dy, dz) invariant
                transformed[i, 6] = boxes[i, 6] + delta_yaw  # Transformed yaw

            return transformed, self.target_frame
        except TransformException as ex:
            self.get_logger().debug(
                f"Could not transform {source_frame} to {self.target_frame}: {ex}; publishing in {source_frame}"
            )
            return boxes, source_frame

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

    @staticmethod
    def custom_msg_to_array(msg) -> np.ndarray:
        if msg.point_num == 0:
            return np.zeros((0, 4), dtype=np.float32)
        points = np.zeros((msg.point_num, 4), dtype=np.float32)
        for i, p in enumerate(msg.points):
            points[i, 0] = p.x
            points[i, 1] = p.y
            points[i, 2] = p.z
            points[i, 3] = p.reflectivity / 255.0
        return points

    @staticmethod
    def custom_msg_to_cloud(msg, points: np.ndarray) -> PointCloud2:
        cloud = PointCloud2()
        cloud.header = msg.header
        cloud.height = 1
        cloud.width = points.shape[0]
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = points.astype(np.float32).tobytes()
        return cloud

    def to_detection_array(
        self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, stamp, frame: str
    ) -> Detection3DArray:
        array = Detection3DArray()
        array.header.stamp = stamp
        array.header.frame_id = frame

        for box, score, label in zip(boxes, scores, labels):
            detection = Detection3D()
            detection.header = array.header

            hyp = ObjectHypothesisWithPose()
            cls_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else "unknown"
            hyp.hypothesis.class_id = cls_name
            hyp.hypothesis.score = float(score)
            detection.results.append(hyp)

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

    def to_markers(
        self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, stamp, frame: str
    ) -> MarkerArray:
        markers = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.header.frame_id = frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            cls_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else "unknown"
            r, g, b = CLASS_COLORS.get(cls_name, (1.0, 1.0, 1.0))
            qx, qy, qz, qw = yaw_to_quaternion(float(box[6]))

            # 1. 3D Bounding Box Cube
            box_marker = Marker()
            box_marker.header.frame_id = frame
            box_marker.header.stamp = stamp
            box_marker.ns = "livox_3d_boxes"
            box_marker.id = i * 2
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD
            box_marker.pose.position.x = float(box[0])
            box_marker.pose.position.y = float(box[1])
            box_marker.pose.position.z = float(box[2])
            box_marker.pose.orientation.x = qx
            box_marker.pose.orientation.y = qy
            box_marker.pose.orientation.z = qz
            box_marker.pose.orientation.w = qw
            box_marker.scale.x = float(box[3])
            box_marker.scale.y = float(box[4])
            box_marker.scale.z = float(box[5])
            box_marker.color.r = r
            box_marker.color.g = g
            box_marker.color.b = b
            box_marker.color.a = 0.45
            markers.markers.append(box_marker)

            # 2. Text Label Marker above box
            text_marker = Marker()
            text_marker.header.frame_id = frame
            text_marker.header.stamp = stamp
            text_marker.ns = "livox_labels"
            text_marker.id = i * 2 + 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(box[0])
            text_marker.pose.position.y = float(box[1])
            text_marker.pose.position.z = float(box[2] + box[5] / 2.0 + 0.25)
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.25
            dist = np.linalg.norm(box[:3])
            text_marker.text = f"{cls_name} ({score:.2f}) [{dist:.2f}m in {frame}]"
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.95
            markers.markers.append(text_marker)

        return markers


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LivoxDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
