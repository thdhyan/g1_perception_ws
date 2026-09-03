#!/usr/bin/env python3
"""Republish Unitree G1 LiDAR clouds onto ``/livox/mid360/points``.

G1_sim's detection pipeline always listens on ``/livox/mid360/points`` with
frame_id ``mid360_link``. Source modes:

* ``source:=sim``          - passthrough (Isaac Sim already publishes there).
* ``source:=real``         - **real robot with g1_sensors.launch.py running** —
                             remaps ``/livox/lidar`` (PointCloud2, frame
                             ``livox_frame``) published by livox_ros_driver2
                             (xfer_format=0). Confirmed live on
                             unitree@ubuntu.local. Output frame rewritten to
                             ``mid360_link`` (matches the static TF in
                             g1_sensors.launch.py: livox_frame → mid360_link).
* ``source:=unitree_slam`` - uses ``/unitree/slam_mapping/points``, the point
                             cloud from the robot's onboard SLAM pipeline.
* ``source:=utlidar``      - legacy ``/utlidar/cloud`` (per old Unitree docs).

Set ``input_topic`` to override source-based selection entirely.

Optional: set ``target_frame`` (e.g. ``pelvis``) to apply a TF transform so the
published cloud is in a different coordinate frame (useful when the sensor is
mounted inverted and the robot's TF tree has the correct transform).
"""

from __future__ import annotations

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2

OUTPUT_TOPIC = "/livox/mid360/points"
OUTPUT_FRAME = "mid360_link"

SIM_INPUT_TOPIC   = OUTPUT_TOPIC
REAL_INPUT_TOPIC  = "/livox/lidar"
UNITREE_SLAM_INPUT_TOPIC = "/unitree/slam_mapping/points"
UTLIDAR_INPUT_TOPIC = "/utlidar/cloud"

_SOURCE_MAP = {
    "sim":          SIM_INPUT_TOPIC,
    "real":         REAL_INPUT_TOPIC,
    "unitree_slam": UNITREE_SLAM_INPUT_TOPIC,
    "utlidar":      UTLIDAR_INPUT_TOPIC,
}


class LidarBridge(Node):
    def __init__(self) -> None:
        super().__init__("g1_lidar_bridge")

        self.declare_parameter("source",           "real")
        self.declare_parameter("input_topic",      "")
        self.declare_parameter("output_topic",     OUTPUT_TOPIC)
        self.declare_parameter("output_frame_id",  OUTPUT_FRAME)
        # If set, transform cloud to this frame via TF instead of just renaming.
        self.declare_parameter("target_frame",     "")

        source        = self.get_parameter("source").value
        input_override = self.get_parameter("input_topic").value
        self.output_topic    = self.get_parameter("output_topic").value
        self.output_frame_id = self.get_parameter("output_frame_id").value
        self.target_frame    = self.get_parameter("target_frame").value.strip()

        if input_override:
            input_topic = input_override
        elif source in _SOURCE_MAP:
            input_topic = _SOURCE_MAP[source]
        else:
            raise ValueError(
                f"source must be one of {list(_SOURCE_MAP)}, got {source!r}"
            )

        self.passthrough = (input_topic == self.output_topic and not self.target_frame)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # TF transform setup (only when target_frame is set)
        self._tf_buf = None
        if self.target_frame:
            from tf2_ros import Buffer, TransformListener
            import tf2_sensor_msgs  # noqa: F401 — registers do_transform_cloud
            self._tf_buf = Buffer()
            self._tf_listener = TransformListener(self._tf_buf, self)
            self.output_frame_id = self.target_frame

        if not self.passthrough:
            self.publisher = self.create_publisher(PointCloud2, self.output_topic, sensor_qos)
            self.create_subscription(PointCloud2, input_topic, self.on_cloud, sensor_qos)

        self.get_logger().info(
            f"source={source} input={input_topic} output={self.output_topic} "
            f"frame_id={self.output_frame_id}"
            + (f" [TF→{self.target_frame}]" if self.target_frame else "")
            + (" (passthrough, no republish needed)" if self.passthrough else "")
        )

    def on_cloud(self, msg: PointCloud2) -> None:
        if self._tf_buf is not None:
            try:
                from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
                transform = self._tf_buf.lookup_transform(
                    self.target_frame,
                    msg.header.frame_id,
                    msg.header.stamp,
                    timeout=Duration(seconds=0.05),
                )
                msg = do_transform_cloud(msg, transform)
            except Exception as e:
                # TF not yet available or timeout — drop frame silently
                self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=5.0)
                return
        else:
            msg.header.frame_id = self.output_frame_id
        self.publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = LidarBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
