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
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2

OUTPUT_TOPIC = "/livox/mid360/points"
OUTPUT_FRAME = "mid360_link"

# Sim already publishes on the output topic/frame directly.
SIM_INPUT_TOPIC = OUTPUT_TOPIC
# Real robot: livox_ros_driver2 with xfer_format=0 publishes PointCloud2 here.
# Confirmed live from g1_sensors.launch.py on unitree@ubuntu.local.
REAL_INPUT_TOPIC = "/livox/lidar"
# Robot onboard SLAM point cloud (confirmed live).
UNITREE_SLAM_INPUT_TOPIC = "/unitree/slam_mapping/points"
# Legacy topic per old Unitree G1 LiDAR docs.
UTLIDAR_INPUT_TOPIC = "/utlidar/cloud"

_SOURCE_MAP = {
    "sim": SIM_INPUT_TOPIC,
    "real": REAL_INPUT_TOPIC,
    "unitree_slam": UNITREE_SLAM_INPUT_TOPIC,
    "utlidar": UTLIDAR_INPUT_TOPIC,
}


class LidarBridge(Node):
    def __init__(self) -> None:
        super().__init__("g1_lidar_bridge")

        self.declare_parameter("source", "real")  # "real" or "sim"
        self.declare_parameter("input_topic", "")  # override; empty = derive from source
        self.declare_parameter("output_topic", OUTPUT_TOPIC)
        self.declare_parameter("output_frame_id", OUTPUT_FRAME)

        source = self.get_parameter("source").value
        input_override = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.output_frame_id = self.get_parameter("output_frame_id").value

        if input_override:
            input_topic = input_override
        elif source in _SOURCE_MAP:
            input_topic = _SOURCE_MAP[source]
        else:
            raise ValueError(
                f"source must be one of {list(_SOURCE_MAP)}, got {source!r}"
            )

        self.passthrough = input_topic == self.output_topic

        # Sensor data is best-effort: dropping a stale cloud beats queueing it.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        if not self.passthrough:
            self.publisher = self.create_publisher(PointCloud2, self.output_topic, sensor_qos)
            self.create_subscription(PointCloud2, input_topic, self.on_cloud, sensor_qos)

        self.get_logger().info(
            f"source={source} input={input_topic} output={self.output_topic} "
            f"frame_id={self.output_frame_id}"
            + (" (passthrough, no republish needed)" if self.passthrough else "")
        )

    def on_cloud(self, msg: PointCloud2) -> None:
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
