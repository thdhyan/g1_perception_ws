#!/usr/bin/env python3
"""ROS 2 Node: Streams Livox LiDAR point clouds from CSV recordings to /livox/lidar.

Emulates the real Livox Mid-360 sensor topic in PointCloud2 format at 10Hz,
supporting looping, speed scaling, zero-point filtering, and optional static TF publication.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


def create_pointcloud2(
    points_xyz_i: np.ndarray,
    frame_id: str,
    stamp,
) -> PointCloud2:
    """Create a ROS2 sensor_msgs/PointCloud2 from an Nx4 numpy array [x, y, z, intensity]."""
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_xyz_i.shape[0]

    # Define fields: x, y, z, intensity (all float32, 16 bytes per point)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True

    if points_xyz_i.dtype != np.float32:
        points_xyz_i = points_xyz_i.astype(np.float32)

    msg.data = points_xyz_i.tobytes()
    return msg


class LivoxCSVReader:
    """Memory-efficient streaming reader for Livox Viewer CSV export files."""

    def __init__(
        self,
        file_path: str,
        time_window_ns: int = 100_000_000,  # 100ms default (10Hz)
        filter_zeros: bool = True,
    ):
        self.file_path = file_path
        self.time_window_ns = time_window_ns
        self.filter_zeros = filter_zeros
        self.file_handle = None
        self._open_file()

    def _open_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        self.file_handle = open(self.file_path, "r", buffering=1024 * 1024)
        # Skip header rows
        # Line 1: Column names
        # Line 2: Device serial / metadata
        self.file_handle.readline()
        self.file_handle.readline()

    def reset(self):
        """Reset reader to beginning of data."""
        if self.file_handle:
            self.file_handle.close()
        self._open_file()

    def close(self):
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.close()

    def read_frames(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Yields (points_array_Nx4, start_timestamp_ns) for each time window."""
        current_frame_points = []
        frame_start_ts = None

        for line in self.file_handle:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split(",")
            if len(parts) < 11:
                continue

            try:
                ts = int(parts[6])
                x = float(parts[7])
                y = float(parts[8])
                z = float(parts[9])
                reflectivity = float(parts[10])
            except (ValueError, IndexError):
                continue

            if self.filter_zeros and (x == 0.0 and y == 0.0 and z == 0.0):
                continue

            if frame_start_ts is None:
                frame_start_ts = ts

            if ts - frame_start_ts >= self.time_window_ns:
                if current_frame_points:
                    yield np.array(current_frame_points, dtype=np.float32), frame_start_ts
                current_frame_points = []
                frame_start_ts = ts

            current_frame_points.append((x, y, z, reflectivity))

        if current_frame_points:
            yield np.array(current_frame_points, dtype=np.float32), frame_start_ts


class LivoxCSVPlayerNode(Node):
    """ROS 2 Node to stream recorded Livox CSV files as sensor_msgs/PointCloud2."""

    def __init__(self):
        super().__init__("livox_csv_player_node")

        # Declare ROS 2 parameters
        default_csv = "/home/thakk100/Projects/thesis/g1_perception_ws/l;ong_test.csv.Csv"
        self.declare_parameter("csv_path", default_csv)
        self.declare_parameter("topic", "/livox/lidar")
        self.declare_parameter("frame_id", "mid360_link")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("time_window_ms", 100.0)
        self.declare_parameter("filter_zeros", True)
        self.declare_parameter("loop", True)
        self.declare_parameter("playback_speed", 1.0)
        self.declare_parameter("publish_tf", True)

        self.csv_path = self.get_parameter("csv_path").value
        self.topic_name = self.get_parameter("topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.time_window_ms = float(self.get_parameter("time_window_ms").value)
        self.filter_zeros = bool(self.get_parameter("filter_zeros").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.playback_speed = float(self.get_parameter("playback_speed").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        # Fallback check for CSV path
        if not os.path.exists(self.csv_path):
            alt_path = os.path.join(os.getcwd(), "l;ong_test.csv.Csv")
            if os.path.exists(alt_path):
                self.csv_path = alt_path
            else:
                self.get_logger().error(f"CSV file does not exist at: {self.csv_path}")

        # QoS matching robot livox driver
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.pub_cloud = self.create_publisher(PointCloud2, self.topic_name, sensor_qos)
        self.get_logger().info(
            f"Livox CSV Player initialized.\n"
            f"  File: {self.csv_path}\n"
            f"  Publishing to: {self.topic_name}\n"
            f"  Frame ID: {self.frame_id}\n"
            f"  Target Rate: {self.rate_hz:.1f} Hz (window: {self.time_window_ms}ms)\n"
            f"  Loop: {self.loop} | Speed: {self.playback_speed}x"
        )

        # Optional static TF broadcaster (base_link -> pelvis -> mid360_link -> livox_frame)
        if self.publish_tf:
            self._publish_static_tfs()

        time_window_ns = int(self.time_window_ms * 1_000_000)
        self.reader = LivoxCSVReader(
            file_path=self.csv_path,
            time_window_ns=time_window_ns,
            filter_zeros=self.filter_zeros,
        )
        self.frame_generator = self.reader.read_frames()

        self.frame_count = 0
        timer_period = (1.0 / self.rate_hz) / max(0.01, self.playback_speed)
        self.timer = self.create_timer(timer_period, self._publish_next_frame)

    def _publish_static_tfs(self):
        """Publishes static transforms so downstream nodes (detection, RViz) work without TF conflicts."""
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        tfs = []
        now = self.get_clock().now().to_msg()

        # 1. base_link -> pelvis (z=0.76m places robot feet flat on ground plane z=0)
        tf_base = TransformStamped()
        tf_base.header.stamp = now
        tf_base.header.frame_id = "base_link"
        tf_base.child_frame_id = "pelvis"
        tf_base.transform.translation.x = 0.0
        tf_base.transform.translation.y = 0.0
        tf_base.transform.translation.z = 0.76
        tf_base.transform.rotation.w = 1.0
        tfs.append(tf_base)

        # 2. odom -> base_link (identity)
        tf_odom = TransformStamped()
        tf_odom.header.stamp = now
        tf_odom.header.frame_id = "odom"
        tf_odom.child_frame_id = "base_link"
        tf_odom.transform.rotation.w = 1.0
        tfs.append(tf_odom)

        self.tf_broadcaster.sendTransform(tfs)
        self.get_logger().info("Published clean static TFs (odom -> base_link -> pelvis [z=0.76]).")

    def _publish_next_frame(self):
        try:
            points, ts_ns = next(self.frame_generator)
        except StopIteration:
            if self.loop:
                self.get_logger().info("Reached end of CSV. Looping playback...")
                self.reader.reset()
                self.frame_generator = self.reader.read_frames()
                try:
                    points, ts_ns = next(self.frame_generator)
                except StopIteration:
                    self.get_logger().error("Failed to read frames after reset.")
                    return
            else:
                self.get_logger().info("Playback completed. Exiting player timer.")
                self.timer.cancel()
                return

        stamp = self.get_clock().now().to_msg()
        cloud_msg = create_pointcloud2(points, self.frame_id, stamp)
        self.pub_cloud.publish(cloud_msg)

        self.frame_count += 1
        if self.frame_count % 20 == 1:
            self.get_logger().info(
                f"[Frame {self.frame_count}] Published {points.shape[0]:,d} points on {self.topic_name} (frame: {self.frame_id})"
            )

    def destroy_node(self):
        if hasattr(self, "reader"):
            self.reader.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LivoxCSVPlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
