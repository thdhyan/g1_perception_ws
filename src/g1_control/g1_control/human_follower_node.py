#!/usr/bin/env python3
"""
G1 human follower node -- ROS2 node that subscribes to /g1/selected_human
(PoseStamped) and publishes nav goals to /g1/nav_goal at fixed rate.

The robot follows behind the human at a standoff distance, facing the same
direction the human is facing. Hysteresis thresholds prevent jitter.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion


def _quat_to_yaw(q: Quaternion) -> float:
    """Convert quaternion to yaw (rotation around z-axis)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _yaw_to_quat(yaw: float) -> Quaternion:
    """Convert yaw (rotation around z-axis) to quaternion."""
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    return q


class HumanFollowerNode(Node):
    def __init__(self):
        super().__init__("g1_human_follower")

        self.declare_parameter("standoff_distance", 0.6)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("pos_threshold", 0.15)
        self.declare_parameter("yaw_threshold", 0.26)
        self.declare_parameter("publish_rate", 2.0)

        self.standoff_distance = self.get_parameter("standoff_distance").value
        self.map_frame = self.get_parameter("map_frame").value
        self.pos_threshold = self.get_parameter("pos_threshold").value
        self.yaw_threshold = self.get_parameter("yaw_threshold").value
        publish_rate = self.get_parameter("publish_rate").value

        self.nav_goal_pub = self.create_publisher(PoseStamped, "/g1/nav_goal", 10)
        self.create_subscription(
            PoseStamped, "/g1/selected_human", self.on_human_pose, 10
        )

        self.last_goal = None
        self.last_human_x = None
        self.last_human_y = None
        self.last_human_yaw = None

        period = 1.0 / publish_rate
        self.create_timer(period, self.on_publish_timer)

        self.get_logger().info(
            f"Human follower: standoff={self.standoff_distance}m, "
            f"pos_threshold={self.pos_threshold}m, "
            f"yaw_threshold={self.yaw_threshold}rad, "
            f"publish_rate={publish_rate}Hz"
        )

    def on_human_pose(self, msg: PoseStamped):
        """Process new human pose, compute standoff goal if thresholds crossed."""
        human_x = msg.pose.position.x
        human_y = msg.pose.position.y
        human_yaw = _quat_to_yaw(msg.pose.orientation)

        update_goal = False

        # Check position threshold
        if self.last_human_x is not None and self.last_human_y is not None:
            dx = human_x - self.last_human_x
            dy = human_y - self.last_human_y
            displacement = math.hypot(dx, dy)
            if displacement > self.pos_threshold:
                update_goal = True
        else:
            update_goal = True

        # Check yaw threshold
        if not update_goal and self.last_human_yaw is not None:
            yaw_diff = abs(human_yaw - self.last_human_yaw)
            # Normalize to [-pi, pi]
            if yaw_diff > math.pi:
                yaw_diff = 2.0 * math.pi - yaw_diff
            if yaw_diff > self.yaw_threshold:
                update_goal = True

        # Compute standoff point: 60cm BEHIND the direction human is FACING
        # standoff_x = human_x - standoff_distance * cos(human_yaw)
        # standoff_y = human_y - standoff_distance * sin(human_yaw)
        standoff_x = human_x - self.standoff_distance * math.cos(human_yaw)
        standoff_y = human_y - self.standoff_distance * math.sin(human_yaw)

        if update_goal:
            self.last_human_x = human_x
            self.last_human_y = human_y
            self.last_human_yaw = human_yaw

            goal = PoseStamped()
            goal.header.frame_id = self.map_frame
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = standoff_x
            goal.pose.position.y = standoff_y
            goal.pose.position.z = 0.0
            goal.pose.orientation = _yaw_to_quat(human_yaw)

            self.last_goal = goal
            self.get_logger().debug(
                f"Updated goal: ({standoff_x:.2f}, {standoff_y:.2f}), yaw={human_yaw:.2f}rad"
            )

    def on_publish_timer(self):
        """Republish last goal if one exists (Nav2 needs periodic updates)."""
        if self.last_goal is not None:
            self.last_goal.header.stamp = self.get_clock().now().to_msg()
            self.nav_goal_pub.publish(self.last_goal)


def main(args=None):
    rclpy.init(args=args)
    node = HumanFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
