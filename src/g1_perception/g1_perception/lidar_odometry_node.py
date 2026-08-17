#!/usr/bin/env python3
"""LiDAR odometry via ICP scan matching on /livox/mid360/points.

Publishes /odom (nav_msgs/Odometry) and TF odom→base_link.
No open3d — uses scipy KDTree + numpy SVD for point-to-point ICP.

Dependencies: scipy, numpy (both standard in ROS2 Jazzy environment).

Parameters:
  input_topic             (default: /livox/mid360/points)
  odom_frame              (default: odom)
  base_frame              (default: base_link)
  publish_tf              (default: true)
  voxel_size              (default: 0.15)  — grid downsample leaf size (m)
  max_correspondence_dist (default: 0.5)   — ICP max point pair distance (m)
  max_iterations          (default: 30)    — ICP iterations per frame
  min_points              (default: 100)   — skip frame if fewer points
  fitness_threshold       (default: 0.3)   — skip update if mean dist > this
"""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial import KDTree
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformBroadcaster


def _voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Grid voxel downsample: one point per voxel cell (centroid)."""
    if len(pts) == 0:
        return pts
    voxel_idx = np.floor(pts / voxel_size).astype(np.int32)
    # Use structured key for unique voxel lookup
    keys = voxel_idx[:, 0] * 1_000_003 + voxel_idx[:, 1] * 1_009 + voxel_idx[:, 2]
    _, inv = np.unique(keys, return_inverse=True)
    out = np.zeros((inv.max() + 1, 3), dtype=np.float32)
    np.add.at(out, inv, pts)
    counts = np.bincount(inv)
    return out / counts[:, None]


def _icp(src: np.ndarray, dst: np.ndarray,
         max_dist: float, max_iter: int) -> tuple[np.ndarray, float]:
    """Point-to-point ICP. Returns 4x4 transform and mean correspondence dist."""
    T = np.eye(4, dtype=np.float64)
    src_h = src.copy().astype(np.float64)

    for _ in range(max_iter):
        tree = KDTree(dst)
        dists, idx = tree.query(src_h, workers=-1)

        # Filter by max distance
        mask = dists < max_dist
        if mask.sum() < 10:
            break

        s = src_h[mask]
        d = dst[idx[mask]].astype(np.float64)

        # Compute centroids
        cs = s.mean(axis=0)
        cd = d.mean(axis=0)

        # SVD for optimal rotation
        H = (s - cs).T @ (d - cd)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cd - R @ cs

        # Accumulate
        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = t
        T = dT @ T
        src_h = (R @ src_h.T).T + t

        # Convergence check
        if np.linalg.norm(t) < 1e-4 and abs(np.trace(R) - 3) < 1e-5:
            break

    # Final fitness: mean dist after registration
    tree = KDTree(dst)
    dists, _ = tree.query(src_h, workers=-1)
    fitness = float(dists[dists < max_dist].mean()) if (dists < max_dist).any() else 999.0

    return T, fitness


def _yaw_from_matrix(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def _quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class LidarOdometryNode(Node):
    def __init__(self) -> None:
        super().__init__("g1_lidar_odometry")

        self.declare_parameter("input_topic", "/livox/lidar")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "livox_frame")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("voxel_size", 0.15)
        self.declare_parameter("max_correspondence_dist", 0.5)
        self.declare_parameter("max_iterations", 30)
        self.declare_parameter("min_points", 100)
        self.declare_parameter("fitness_threshold", 0.3)

        self._input_topic = self.get_parameter("input_topic").value
        self._odom_frame = self.get_parameter("odom_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._publish_tf = self.get_parameter("publish_tf").value
        self._voxel_size = self.get_parameter("voxel_size").value
        self._max_dist = self.get_parameter("max_correspondence_dist").value
        self._max_iter = int(self.get_parameter("max_iterations").value)
        self._min_pts = int(self.get_parameter("min_points").value)
        self._fitness_thresh = self.get_parameter("fitness_threshold").value

        # World pose accumulator
        self._T_world = np.eye(4, dtype=np.float64)
        self._prev_pts: np.ndarray | None = None
        self._prev_stamp: float | None = None

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self._path_pub = self.create_publisher(Path, "/odom_path", 10)
        self._tf_br = TransformBroadcaster(self)

        self._path = Path()
        self._path.header.frame_id = self._odom_frame

        self.create_subscription(PointCloud2, self._input_topic, self._on_cloud, sensor_qos)
        self.get_logger().info(
            f"LiDAR odometry: {self._input_topic} → /odom  "
            f"(voxel={self._voxel_size}m, max_dist={self._max_dist}m, "
            f"max_iter={self._max_iter})"
        )

    def _on_cloud(self, msg: PointCloud2) -> None:
        # Use laptop clock, not robot clock (robot clock skewed ~8h behind)
        now = self.get_clock().now()
        stamp_sec = now.nanoseconds * 1e-9
        msg.header.stamp = now.to_msg()

        # Convert to (N,3) float32
        raw = point_cloud2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)
        if raw.shape[0] < self._min_pts:
            return
        pts = np.stack([raw["x"], raw["y"], raw["z"]], axis=-1).astype(np.float32)

        # Remove ground (z < -0.3 m relative to sensor) and far points
        pts = pts[(pts[:, 2] > -0.3) & (np.linalg.norm(pts, axis=1) < 25.0)]
        pts = _voxel_downsample(pts, self._voxel_size)

        if self._prev_pts is None:
            # First frame — identity
            self._prev_pts = pts
            self._prev_stamp = stamp_sec
            self._publish(msg.header.stamp, np.zeros(3), 0.0, np.zeros(3))
            return

        dt = stamp_sec - self._prev_stamp if self._prev_stamp else 0.1
        dt = max(dt, 1e-3)

        t0 = time.monotonic()
        T_icp, fitness = _icp(pts, self._prev_pts, self._max_dist, self._max_iter)
        elapsed = time.monotonic() - t0

        self.get_logger().debug(
            f"ICP fitness={fitness:.4f} dt={dt:.3f}s icp_time={elapsed*1000:.1f}ms"
        )

        if fitness > self._fitness_thresh:
            self.get_logger().warning(
                f"ICP poor match (fitness={fitness:.3f} > {self._fitness_thresh}), "
                "skipping pose update"
            )
            # Still update reference to avoid accumulating drift on next frame
            self._prev_pts = pts
            self._prev_stamp = stamp_sec
            return

        # Accumulate world pose
        self._T_world = self._T_world @ np.linalg.inv(T_icp)

        pos = self._T_world[:3, 3]
        yaw = _yaw_from_matrix(self._T_world[:3, :3])

        # Twist from delta T
        delta_pos = T_icp[:3, 3]
        delta_yaw = _yaw_from_matrix(T_icp[:3, :3])
        vel = delta_pos / dt
        vyaw = delta_yaw / dt

        self._publish(msg.header.stamp, pos, yaw, vel, vyaw)

        self._prev_pts = pts
        self._prev_stamp = stamp_sec

    def _publish(self, stamp, pos: np.ndarray, yaw: float,
                 vel: np.ndarray, vyaw: float = 0.0) -> None:
        qx, qy, qz, qw = _quat_from_yaw(yaw)

        # Odometry msg
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(vel[0])
        odom.twist.twist.linear.y = float(vel[1])
        odom.twist.twist.angular.z = float(vyaw)
        self._odom_pub.publish(odom)

        # TF odom → base_link
        if self._publish_tf:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self._odom_frame
            tf.child_frame_id = self._base_frame
            tf.transform.translation.x = float(pos[0])
            tf.transform.translation.y = float(pos[1])
            tf.transform.translation.z = float(pos[2])
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self._tf_br.sendTransform(tf)

        # Path for RViz
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self._odom_frame
        pose_msg.pose = odom.pose.pose
        self._path.header.stamp = stamp
        self._path.poses.append(pose_msg)
        # Cap path length to avoid memory growth
        if len(self._path.poses) > 2000:
            self._path.poses = self._path.poses[-2000:]
        self._path_pub.publish(self._path)


def main() -> None:
    rclpy.init()
    node = LidarOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
