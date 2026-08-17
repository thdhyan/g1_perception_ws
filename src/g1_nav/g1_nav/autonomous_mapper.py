#!/usr/bin/env python3
"""Autonomous mapping & exploration commander for Unitree G1 humanoid.

Sends smooth velocity commands (Twist on /cmd_vel) and pose deltas to WBC
to systematically explore and map the environment while monitoring SLAM
and maintaining balance.
"""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import Empty, String


class AutonomousMapperNode(Node):
    def __init__(self):
        super().__init__("g1_autonomous_mapper")

        self.declare_parameter("linear_speed", 0.20)
        self.declare_parameter("turn_speed", 0.35)
        self.declare_parameter("step_duration", 4.0)
        self.declare_parameter("pause_duration", 2.0)
        self.turn_duration = 3.2
        self.declare_parameter("startup_delay", 4.0)
        self.declare_parameter("auto_start", True)

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.turn_speed = float(self.get_parameter("turn_speed").value)
        self.step_duration = float(self.get_parameter("step_duration").value)
        self.pause_duration = float(self.get_parameter("pause_duration").value)
        self.startup_delay = float(self.get_parameter("startup_delay").value)
        self.auto_start = bool(self.get_parameter("auto_start").value)

        # Velocity publisher for WBC
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status_sub = self.create_subscription(
            String, "/g1/wbc_status", self._wbc_status_cb, 10
        )

        # Waypoint schedule for warehouse loop mapping:
        # (vx, vy, wz, duration_sec, description)
        self.patrol_plan: List[Tuple[float, float, float, float, str]] = [
            (0.0, 0.0, 0.0, self.startup_delay, "Initial Stand & Balance Stabilization"),
            (self.linear_speed, 0.0, 0.0, self.step_duration, "Forward Exploration 1"),
            (0.0, 0.0, 0.0, self.pause_duration, "Settle & Scan 1"),
            (0.0, 0.0, self.turn_speed, self.turn_duration, "Rotate Left 90 deg"),
            (0.0, 0.0, 0.0, self.pause_duration, "Settle & Scan 2"),
            (self.linear_speed, 0.0, 0.0, self.step_duration, "Forward Exploration 2"),
            (0.0, 0.0, 0.0, self.pause_duration, "Settle & Scan 3"),
            (0.0, 0.0, self.turn_speed, self.turn_duration, "Rotate Left 90 deg"),
            (0.0, 0.0, 0.0, self.pause_duration, "Settle & Scan 4"),
            (self.linear_speed, 0.0, 0.0, self.step_duration, "Forward Exploration 3"),
            (0.0, 0.0, 0.0, self.pause_duration, "Settle & Scan 5"),
            (0.0, 0.0, self.turn_speed, self.turn_duration, "Rotate Left 90 deg"),
            (0.0, 0.0, 0.0, self.pause_duration, "Settle & Scan 6"),
            (self.linear_speed, 0.0, 0.0, self.step_duration, "Forward Return Leg"),
            (0.0, 0.0, 0.0, self.pause_duration, "Settle & Scan Final"),
            (0.0, 0.0, self.turn_speed, self.turn_duration, "Rotate Left to Initial Heading"),
        ]

        self.current_leg_idx = 0
        self.leg_start_time = self.get_clock().now().nanoseconds / 1e9
        self.wbc_ready = True
        self.active = self.auto_start

        # Continuous control timer (10 Hz)
        self.timer = self.create_timer(0.1, self._control_loop)
        self.get_logger().info(
            f"Autonomous Mapper initialized. Legs: {len(self.patrol_plan)}, auto_start={self.auto_start}"
        )

    def _wbc_status_cb(self, msg: String):
        # Can monitor status e.g. "walk", "balance"
        pass

    def _control_loop(self):
        if not self.active:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.current_leg_idx >= len(self.patrol_plan):
            # Loop the exploration pattern or hold stationary
            self.current_leg_idx = 0
            self.leg_start_time = now
            self.get_logger().info("Restarting warehouse exploration loop.")

        vx, vy, wz, duration, desc = self.patrol_plan[self.current_leg_idx]

        if now - self.leg_start_time > duration:
            # Advance to next leg
            self.current_leg_idx += 1
            self.leg_start_time = now
            if self.current_leg_idx < len(self.patrol_plan):
                _, _, _, _, next_desc = self.patrol_plan[self.current_leg_idx]
                self.get_logger().info(f"[AutoMapper] Next Phase ({self.current_leg_idx+1}/{len(self.patrol_plan)}): {next_desc}")
            return

        # Publish smooth velocity command
        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.linear.y = float(vy)
        cmd.angular.z = float(wz)
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
