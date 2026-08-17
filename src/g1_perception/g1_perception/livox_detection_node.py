#!/usr/bin/env python3
"""ROS2 node: Livox native CustomMsg -> CenterPoint 3D detections + RViz markers.

Reads native livox_ros_driver2 CustomMsg directly (no conversion needed).
Converts CustomMsg points to numpy array and runs the CenterPoint backend.
Also converts to PointCloud2 and publishes for other nodes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray


def _find_g1_sim() -> Path:
    """Locate G1_sim near this file.

    Walks up from this file's location looking for a ``G1_sim`` sibling of any
    ancestor directory. Works whether this package is run from the ``src``
    tree (``colcon build --symlink-install``) or a copied ``install`` tree.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor.parent / "G1_sim"
        if candidate.is_dir():
            return candidate
    # Fall back to documented default layout
    return here.parents[3] / "G1_sim"


G1_SIM_PATH = _find_g1_sim()


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Yaw-only rotation as ``(x, y, z, w)``; boxes are never rolled or pitched."""
    return (0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0)))


class LivoxDetectionNode(Node):
    def __init__(self):
        super().__init__("g1_livox_detection")

        # Declare parameters
        self.declare_parameter("checkpoint_path", "pt/livox_model_1.pt")
        self.declare_parameter("max_hz", 5.0)
        self.declare_parameter("score_threshold", 0.4)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("input_topic", "/livox/lidar")
        self.declare_parameter("frame_override", "")

        checkpoint_path = self.get_parameter("checkpoint_path").value
        self.max_hz = self.get_parameter("max_hz").value
        self.score_threshold = self.get_parameter("score_threshold").value
        device = self.get_parameter("device").value
        input_topic = self.get_parameter("input_topic").value
        self.frame_override = self.get_parameter("frame_override").value
        self.frame_override = self.frame_override if self.frame_override else None

        # Add G1_sim to path so we can import its modules
        if str(G1_SIM_PATH / "detection") not in sys.path:
            sys.path.insert(0, str(G1_SIM_PATH / "detection"))

        # Lazy import of CenterPoint backend; only loaded if checkpoint exists
        self.backend = None
        self._load_backend(checkpoint_path, device)

        # Rate limiting: track last inference time
        self.last_inference_time = 0.0
        self.min_inference_interval = 1.0 / self.max_hz if self.max_hz > 0 else 0.0

        # Try to import CustomMsg; fail gracefully if livox_ros_driver2 not available
        try:
            from livox_ros_driver2.msg import CustomMsg
            self.CustomMsg = CustomMsg
        except ImportError:
            self.get_logger().error(
                "Failed to import livox_ros_driver2.msg.CustomMsg; "
                "ensure livox_ros_driver2 is installed and built"
            )
            raise RuntimeError(
                "livox_ros_driver2 not available; cannot continue"
            )

        # Sensor data is best-effort: dropping a stale cloud beats queueing it.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.detection_pub = self.create_publisher(
            Detection3DArray, "/g1/detections/livox", 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, "/g1/detection_markers/livox", 10
        )
        # Also publish converted PointCloud2 for other nodes
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/livox/mid360/points", 10
        )
        self.create_subscription(self.CustomMsg, input_topic, self.on_custom_msg, sensor_qos)

        self._busy = False
        if self.backend:
            self.get_logger().info(
                f"Livox detection listening on {input_topic} "
                f"(max_hz={self.max_hz}, checkpoint={checkpoint_path})"
            )
        else:
            self.get_logger().warning(
                f"Livox detection listening on {input_topic} but checkpoint not loaded; "
                "publishing empty detections. Set checkpoint_path to enable inference."
            )

    def _load_backend(self, checkpoint_path: str, device: str) -> None:
        """Load the CenterPoint backend, or log a warning if checkpoint is missing."""
        try:
            from livox_centerpoint import LivoxCenterPointBackend

            # Resolve checkpoint path: if relative, interpret as relative to G1_sim/detection
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.is_absolute():
                ckpt_path = G1_SIM_PATH / "detection" / checkpoint_path

            if not ckpt_path.exists():
                self.get_logger().warning(
                    f"Checkpoint not found at {ckpt_path}; inference disabled"
                )
                return

            self.backend = LivoxCenterPointBackend(
                checkpoint=str(ckpt_path),
                device=device,
                score_threshold=self.score_threshold,
            )
            self.backend.load()
            self.get_logger().info(f"CenterPoint checkpoint loaded from {ckpt_path}")

        except FileNotFoundError as e:
            self.get_logger().warning(f"Checkpoint load failed: {e}")
        except Exception as e:
            self.get_logger().warning(f"Backend initialization failed: {e}")

    def on_custom_msg(self, msg) -> None:
        """Process incoming Livox CustomMsg."""
        # Skip if already processing or if rate limit not met
        if self._busy:
            return
        if self.backend is None:
            # Still publish empty arrays so subscribers don't block waiting for data
            frame = self.frame_override or msg.header.frame_id
            self.detection_pub.publish(
                self.to_detection_array(
                    np.zeros((0, 7), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    msg.header.stamp,
                    frame,
                )
            )
            self.marker_pub.publish(
                self.to_markers(
                    np.zeros((0, 7), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    msg.header.stamp,
                    frame,
                )
            )
            # Also publish empty cloud
            self.cloud_pub.publish(
                self.custom_msg_to_cloud(msg, np.zeros((0, 4), dtype=np.float32))
            )
            return

        # Rate limiting: drop frames that arrive too fast
        now = time()
        if now - self.last_inference_time < self.min_inference_interval:
            return
        self.last_inference_time = now

        self._busy = True
        try:
            points = self.custom_msg_to_array(msg)
            frame = self.frame_override or msg.header.frame_id

            # Publish converted PointCloud2 for other nodes
            self.cloud_pub.publish(self.custom_msg_to_cloud(msg, points))

            # Run inference
            boxes, scores, labels = self.backend.infer(points)
            self.detection_pub.publish(self.to_detection_array(boxes, scores, labels, msg.header.stamp, frame))
            self.marker_pub.publish(self.to_markers(boxes, labels, msg.header.stamp, frame))
        except Exception as exc:
            # Keep the node alive across a bad frame
            self.get_logger().error(f"detection failed: {exc}")
        finally:
            self._busy = False

    @staticmethod
    def custom_msg_to_array(msg) -> np.ndarray:
        """Convert Livox CustomMsg to (N, 4) array of [x, y, z, reflectivity_normalized]."""
        if msg.point_num == 0:
            return np.zeros((0, 4), dtype=np.float32)

        points = np.zeros((msg.point_num, 4), dtype=np.float32)
        for i, point in enumerate(msg.points):
            points[i, 0] = point.x
            points[i, 1] = point.y
            points[i, 2] = point.z
            # Normalize reflectivity to [0, 1]
            points[i, 3] = point.reflectivity / 255.0

        return points

    @staticmethod
    def custom_msg_to_cloud(msg, points: np.ndarray) -> PointCloud2:
        """Convert CustomMsg to PointCloud2 for other nodes."""
        cloud = PointCloud2()
        cloud.header = msg.header
        cloud.height = 1
        cloud.width = points.shape[0]

        # Define fields: x, y, z, intensity
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 16  # 4 floats * 4 bytes
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True

        # Pack points into byte buffer
        cloud.data = points.astype(np.float32).tobytes()

        return cloud

    def to_detection_array(
        self, boxes, scores, labels, stamp, frame: str
    ) -> Detection3DArray:
        """Convert detections to Detection3DArray message."""
        array = Detection3DArray()
        array.header.stamp = stamp
        array.header.frame_id = frame

        for box, score, label in zip(boxes, scores, labels):
            detection = Detection3D()
            detection.header = array.header

            hypothesis = ObjectHypothesisWithPose()
            # Import CLASS_NAMES from backend module
            from backend import CLASS_NAMES

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
        """Convert detections to MarkerArray (3D bounding box visualization)."""
        from backend import CLASS_COLORS, CLASS_NAMES

        markers = MarkerArray()

        # Clear previous frame's boxes with a DELETEALL marker
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
            marker.ns = "livox_detections"
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
            marker.color.a = 0.4  # translucent so point cloud stays visible
            markers.markers.append(marker)

        return markers


def main() -> None:
    rclpy.init()
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
