#!/usr/bin/env python3
"""Converts the sim's 32FC1-metres depth image to 16UC1 millimetres.

cuVSLAM's RGBD path reinterprets the depth buffer's raw bytes as uint16
regardless of the ROS `encoding` field, while Gazebo's rgbd_camera sensor
publishes 32FC1 metres -- so cuVSLAM reads float32 mantissa bytes as
integers and gets noise (proven 2026-08-18 via enable_debug_mode dumps; see
HANDOFF.md "Session 3"). `depth_scale_factor` cannot compensate, it scales
*after* the wrong read. This node produces depth in the format cuVSLAM
actually expects; consumers then use depth_scale_factor=0.001.

Non-finite input (Gazebo emits +inf for no-return/sky) and out-of-range
values map to 0, the RealSense invalid-depth sentinel that cuVSLAM and
nvblox both assume.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

MM_PER_M = 1000.0
UINT16_MAX = 65535


class DepthToUint16(Node):
    def __init__(self):
        super().__init__("depth_to_uint16")
        self.declare_parameter("input_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("output_topic", "/camera/depth/image_16uc1")
        in_topic = self.get_parameter("input_topic").value
        out_topic = self.get_parameter("output_topic").value

        self._pub = self.create_publisher(Image, out_topic, qos_profile_sensor_data)
        self._sub = self.create_subscription(
            Image, in_topic, self._on_depth, qos_profile_sensor_data
        )
        self._warned_encoding = False
        self.get_logger().info(f"32FC1 m -> 16UC1 mm: {in_topic} -> {out_topic}")

    def _on_depth(self, msg: Image):
        if msg.encoding != "32FC1":
            if not self._warned_encoding:
                self.get_logger().error(
                    f"expected 32FC1 input, got '{msg.encoding}' -- not converting"
                )
                self._warned_encoding = True
            return

        metres = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        mm = metres * MM_PER_M
        # np.nan_to_num first: NaN/inf would otherwise compare False in the
        # range mask and slip through as undefined casts.
        mm = np.nan_to_num(mm, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(mm, 0, UINT16_MAX, out=mm)
        out_data = mm.astype(np.uint16)

        out = Image()
        out.header = msg.header
        out.height = msg.height
        out.width = msg.width
        out.encoding = "16UC1"
        out.is_bigendian = 0
        out.step = msg.width * 2
        out.data = out_data.tobytes()
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DepthToUint16()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
