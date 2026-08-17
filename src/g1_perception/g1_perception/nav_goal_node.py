#!/usr/bin/env python3
"""
nav_goal_node.py

ROS2 node that sends navigation goals to Nav2 (NavigateToPose action) for the
Unitree G1 robot to walk to a given x,y position in the lidar/map frame.

Subscribes:
    /g1/nav_goal (geometry_msgs/PoseStamped)
        Goal pose. Orientation defaults to identity (facing forward) if not specified.
        Frame should match map_frame parameter.

        Example from CLI:
        ros2 topic pub --once /g1/nav_goal geometry_msgs/PoseStamped \
          '{header: {frame_id: "map"}, pose: {position: {x: 1.0, y: 2.0, z: 0.0}, orientation: {w: 1.0}}}'

Action Client:
    /navigate_to_pose (nav2_msgs/action/NavigateToPose)
        Nav2 navigation goal action server.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose


class NavGoalNode(Node):
    """Sends navigation goals to Nav2 NavigateToPose action server."""

    def __init__(self):
        super().__init__('g1_nav_goal')

        # Parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('goal_timeout_sec', 60.0)

        self.map_frame = self.get_parameter('map_frame').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.goal_timeout_sec = self.get_parameter('goal_timeout_sec').value

        # Action client for Nav2
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._goal_handle = None
        self._pending_goal = None

        # Subscription to nav goal topic
        self.create_subscription(
            PoseStamped, '/g1/nav_goal', self._nav_goal_callback, qos_profile=10)

        self.get_logger().info(
            f"NavGoalNode started. "
            f"Listening on /g1/nav_goal, sending to /navigate_to_pose. "
            f"Timeout: {self.goal_timeout_sec}s"
        )

    # ------------------------------------------------------------------
    def _nav_goal_callback(self, msg: PoseStamped):
        """Receive goal from topic; cancel current goal if one is in progress."""
        goal_pose = msg

        # Validate frame
        if goal_pose.header.frame_id != self.map_frame:
            self.get_logger().warn(
                f"Goal received in frame '{goal_pose.header.frame_id}', "
                f"expected '{self.map_frame}'. Proceeding anyway."
            )

        # Ensure orientation is normalized (default to identity if not set)
        if (goal_pose.pose.orientation.x == 0 and
            goal_pose.pose.orientation.y == 0 and
            goal_pose.pose.orientation.z == 0 and
            goal_pose.pose.orientation.w == 0):
            # Zero quaternion; default to identity (facing forward)
            goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # Cancel any in-flight goal
        if self._goal_handle is not None:
            self.get_logger().info("Canceling current goal; new goal received.")
            self._cancel_goal_async(self._goal_handle)
            self._goal_handle = None

        # Send new goal
        self._send_goal(goal_pose)

    # ------------------------------------------------------------------
    def _send_goal(self, goal_pose: PoseStamped):
        """Send goal to Nav2 action server."""
        # Wait for action server (with timeout)
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn(
                "Nav2 NavigateToPose action server not available. "
                "Ensure nav2_bringup is running."
            )
            self._pending_goal = goal_pose
            return

        # Create action goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info(
            f"Sending goal: x={goal_pose.pose.position.x:.2f}, "
            f"y={goal_pose.pose.position.y:.2f}, "
            f"frame={goal_pose.header.frame_id}"
        )

        # Send goal asynchronously
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg, feedback_callback=self._feedback_callback)
        send_goal_future.add_done_callback(self._goal_response_callback)

    # ------------------------------------------------------------------
    def _goal_response_callback(self, future):
        """Handle Nav2 goal acceptance/rejection."""
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().error("Goal rejected by Nav2 action server.")
            return

        self.get_logger().info("Goal accepted by Nav2.")
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    # ------------------------------------------------------------------
    def _feedback_callback(self, feedback):
        """Log feedback from Nav2 during goal execution."""
        fb = feedback.feedback
        if hasattr(fb, 'estimated_time_remaining'):
            eta = fb.estimated_time_remaining.sec if hasattr(fb.estimated_time_remaining, 'sec') else 0
            self.get_logger().info(
                f"Navigation feedback: "
                f"distance_remaining={getattr(fb, 'distance_remaining', 'N/A'):.2f}m, "
                f"estimated_time_remaining={eta}s"
            )
        else:
            self.get_logger().debug(f"Feedback: {fb}")

    # ------------------------------------------------------------------
    def _result_callback(self, future):
        """Handle Nav2 goal result (success/failure)."""
        result = future.result()
        if result.result is None:
            self.get_logger().error("Goal execution failed or was canceled.")
        else:
            self.get_logger().info("Goal execution succeeded.")
        self._goal_handle = None

    # ------------------------------------------------------------------
    def _cancel_goal_async(self, goal_handle):
        """Cancel an in-flight goal."""
        if goal_handle is None:
            return
        cancel_future = goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(lambda f: self.get_logger().info("Goal canceled."))


def main(args=None):
    rclpy.init(args=args)
    node = NavGoalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
