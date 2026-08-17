#!/usr/bin/env python3
"""
Livox Front 15m Point Cloud Filter & Snapshot Service Node:
1. Subscribes to `/livox/lidar` (PointCloud2).
2. Filters point cloud in real time to front face (X >= 0m) within 15 meters.
3. Publishes continuous filtered live stream on `/livox/live_front_15m` and `/g1/livox/front_roi`.
4. Provides ROS 2 service `/g1/publish_front_snapshot` (std_srvs/srv/Trigger) to capture and publish
   a dense front 15m point cloud snapshot on `/livox/collected_points_front_15m` (latched).
"""

import threading
import time
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_srvs.srv import Trigger


class LivoxFrontFilterNode(Node):
    def __init__(self):
        super().__init__("livox_front_filter_node")

        # Parameters
        self.declare_parameter("input_topic", "/livox/lidar")
        self.declare_parameter("min_x", 0.0)             # Forward hemisphere
        self.declare_parameter("max_range", 15.0)        # Max distance in meters
        self.declare_parameter("collect_frames", 10)     # Frames to accumulate for snapshot
        self.declare_parameter("collect_duration_sec", 1.5)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.min_x = float(self.get_parameter("min_x").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.collect_frames = int(self.get_parameter("collect_frames").value)
        self.collect_duration_sec = float(self.get_parameter("collect_duration_sec").value)

        self.lock = threading.Lock()
        self.is_collecting_snapshot = False
        self.snapshot_frames: List[np.ndarray] = []
        self.snapshot_start_time = 0.0
        self.last_header = None
        self.frame_count = 0

        # QoS Profiles
        sensor_sub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        live_pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publishers
        self.pub_live_front = self.create_publisher(
            PointCloud2, "/livox/live_front_15m", live_pub_qos
        )
        self.pub_live_roi = self.create_publisher(
            PointCloud2, "/g1/livox/front_roi", live_pub_qos
        )
        self.pub_snapshot_front = self.create_publisher(
            PointCloud2, "/livox/collected_points_front_15m", latched_qos
        )

        # Subscriber
        self.sub_lidar = self.create_subscription(
            PointCloud2, self.input_topic, self.on_pointcloud, sensor_sub_qos
        )

        # Services
        self.srv_snapshot = self.create_service(
            Trigger, "/g1/publish_front_snapshot", self.handle_trigger_snapshot
        )
        self.srv_snapshot_alias = self.create_service(
            Trigger, "/g1/get_front_snapshot", self.handle_trigger_snapshot
        )

        self.get_logger().info(
            f"LivoxFrontFilterNode active:\n"
            f"  - Input Topic: '{self.input_topic}' (RELIABLE QoS)\n"
            f"  - ROI Filter: X >= {self.min_x:.1f}m, Distance <= {self.max_range:.1f}m\n"
            f"  - Live Topics: '/livox/live_front_15m', '/g1/livox/front_roi'\n"
            f"  - Snapshot Topic: '/livox/collected_points_front_15m' (Latched)\n"
            f"  - Snapshot Service: '/g1/publish_front_snapshot'"
        )

    def filter_front_points(self, points: np.ndarray) -> np.ndarray:
        """Filter points array to front face and range <= max_range."""
        if points.shape[0] == 0:
            return points
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        dist = np.sqrt(x**2 + y**2 + z**2)
        mask = (x >= self.min_x) & (dist <= self.max_range)
        return points[mask]

    def on_pointcloud(self, msg: PointCloud2) -> None:
        """Process incoming live LiDAR frame."""
        points = self.cloud_to_array(msg)
        if points.shape[0] == 0:
            return

        frame_id = msg.header.frame_id
        if frame_id == "livox_frame" or not frame_id:
            frame_id = "mid360_link"
            msg.header.frame_id = frame_id

        self.frame_count += 1
        front_pts = self.filter_front_points(points)

        # 1. Publish real-time live front point cloud (10Hz)
        if front_pts.shape[0] > 0:
            front_msg = self.array_to_cloud(msg.header, front_pts)
            self.pub_live_front.publish(front_msg)
            self.pub_live_roi.publish(front_msg)

        # 2. Accumulate for snapshot if service was triggered
        with self.lock:
            if not self.is_collecting_snapshot:
                return
            self.snapshot_frames.append(front_pts if front_pts.shape[0] > 0 else points)
            self.last_header = msg.header
            num_frames = len(self.snapshot_frames)
            elapsed = time.time() - self.snapshot_start_time

        if num_frames >= self.collect_frames or elapsed >= self.collect_duration_sec:
            threading.Thread(target=self._publish_snapshot_cloud, daemon=True).start()

    def handle_trigger_snapshot(self, request, response):
        """Service callback to trigger front snapshot collection."""
        with self.lock:
            self.is_collecting_snapshot = True
            self.snapshot_frames.clear()
            self.snapshot_start_time = time.time()

        self.get_logger().info(
            f"[*] Collecting {self.collect_frames} frames of front 15m points for snapshot..."
        )
        response.success = True
        response.message = f"Front snapshot collection started for {self.collect_frames} frames / {self.collect_duration_sec:.1f}s."
        return response

    def _publish_snapshot_cloud(self) -> None:
        """Merge accumulated front frames and publish latched point cloud."""
        with self.lock:
            self.is_collecting_snapshot = False
            frames = list(self.snapshot_frames)
            header = self.last_header

        if not frames or header is None:
            return

        merged = np.vstack(frames)
        front_merged = self.filter_front_points(merged)

        msg = self.array_to_cloud(header, front_merged if front_merged.shape[0] > 0 else merged)
        self.pub_snapshot_front.publish(msg)
        self.get_logger().info(
            f"[✓] Published dense front snapshot ({msg.width:,d} points across {len(frames)} frames) "
            f"to '/livox/collected_points_front_15m' (Latched)"
        )

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
    def array_to_cloud(header, points: np.ndarray) -> PointCloud2:
        cloud = PointCloud2()
        cloud.header = header
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


def main(args=None):
    rclpy.init(args=args)
    node = LivoxFrontFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
