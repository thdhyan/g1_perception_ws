#!/usr/bin/env python3
"""Republish /scan with current laptop time (fixes robot clock skew).

Robot (Foxy) clock is behind laptop (Jazzy) clock — LaserScan arrives with
old timestamps that slam_toolbox's TF buffer can't match. This node replaces
the stamp with the current laptop ROS clock before forwarding.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import LaserScan


class ScanRestamper(Node):
    def __init__(self):
        super().__init__("scan_restamper")

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._pub = self.create_publisher(LaserScan, "/scan_synced", 10)
        self.create_subscription(LaserScan, "/scan", self._cb, qos)
        self.get_logger().info("scan_restamper: /scan → /scan_synced (laptop time)")

    def _cb(self, msg: LaserScan) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = ScanRestamper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
