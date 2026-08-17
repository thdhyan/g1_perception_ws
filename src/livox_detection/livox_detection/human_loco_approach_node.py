#!/usr/bin/env python3
"""Node 3: Human Locomotion Approach Controller Node.

Subscribes to `/g1/selected_human` (PoseStamped in pelvis frame).
Calculates the trajectory to bring the robot to exactly 60 cm (0.6 m) in front
of the target human, facing the human.

Commands movement via:
    - `/g1/cmd_pose` (geometry_msgs/msg/Twist for cmd_pose_bridge)
    - Socket RPC to `/tmp/g1_robot_bridge.sock` (if available)

Publishes:
    - `/g1/cmd_pose` (geometry_msgs/msg/Twist)
    - `/g1/approach_goal` (geometry_msgs/msg/PoseStamped)
    - `/g1/approach_markers` (visualization_msgs/msg/MarkerArray)
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

DEFAULT_SOCKET_PATH = "/tmp/g1_robot_bridge.sock"


class HumanLocoApproachNode(Node):
    def __init__(self):
        super().__init__("g1_human_loco_approach")

        # Declare parameters
        self.declare_parameter("standoff_distance", 0.60)  # 60 cm in front of human
        self.declare_parameter("socket_path", DEFAULT_SOCKET_PATH)
        self.declare_parameter("auto_execute", True)
        self.declare_parameter("linear_speed", 0.20)
        self.declare_parameter("min_approach_threshold", 0.10)  # 10 cm deadband

        self.standoff_dist = float(self.get_parameter("standoff_distance").value)
        self.socket_path = self.get_parameter("socket_path").value
        self.auto_execute = bool(self.get_parameter("auto_execute").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.min_thresh = float(self.get_parameter("min_approach_threshold").value)

        self.lock = threading.Lock()
        self.active_goal: Optional[PoseStamped] = None
        self.is_executing = False

        # Publishers
        self.pub_cmd_pose = self.create_publisher(Twist, "/g1/cmd_pose", 10)
        self.pub_goal_pose = self.create_publisher(PoseStamped, "/g1/approach_goal", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/g1/approach_markers", 10)

        # Subscriber
        self.sub_selected = self.create_subscription(
            PoseStamped, "/g1/selected_human", self.on_selected_human, 10
        )

        self.get_logger().info(
            f"HumanLocoApproachNode active [Standoff: {self.standoff_dist * 100:.0f} cm, Auto-Execute: {self.auto_execute}]"
        )

    def on_selected_human(self, msg: PoseStamped) -> None:
        """Handle new selected human target."""
        pos = msg.pose.position
        dist = math.sqrt(pos.x**2 + pos.y**2)

        if dist < 0.05:
            self.get_logger().warning("Selected human position is too close to pelvis origin; ignoring.")
            return

        with self.lock:
            if self.is_executing:
                self.get_logger().info("Approach already in progress; queuing updated target.")
            self.active_goal = msg

        # Calculate waypoint 60 cm in front of human (along vector from robot to human)
        # Vector from robot (0, 0) to human (pos.x, pos.y)
        unit_x = pos.x / dist
        unit_y = pos.y / dist

        # Distance the robot needs to travel so it stops at standoff_dist from human
        travel_dist = max(0.0, dist - self.standoff_dist)

        # Target stopping position in pelvis frame
        goal_x = unit_x * travel_dist
        goal_y = unit_y * travel_dist
        target_yaw_rad = math.atan2(pos.y, pos.x)
        target_yaw_deg = math.degrees(target_yaw_rad)

        # Publish target waypoint pose
        goal_msg = PoseStamped()
        goal_msg.header = msg.header
        goal_msg.pose.position.x = goal_x
        goal_msg.pose.position.y = goal_y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.z = math.sin(target_yaw_rad / 2.0)
        goal_msg.pose.orientation.w = math.cos(target_yaw_rad / 2.0)
        self.pub_goal_pose.publish(goal_msg)

        # Publish RViz approach visual trajectory markers
        self._publish_approach_markers(msg.header, pos, (goal_x, goal_y), target_yaw_rad)

        self.get_logger().info(
            f"\n[APPROACH PLAN]: Target Human at {dist:.2f}m -> Standoff: {self.standoff_dist:.2f}m\n"
            f"  Movement Required: dx={goal_x:+.2f}m, dy={goal_y:+.2f}m (Travel: {travel_dist:.2f}m), dyaw={target_yaw_deg:+.1f}°"
        )

        if self.auto_execute and travel_dist > self.min_thresh:
            threading.Thread(
                target=self._execute_motion,
                args=(goal_x, goal_y, target_yaw_deg, travel_dist),
                daemon=True,
            ).start()
        elif travel_dist <= self.min_thresh:
            self.get_logger().info("Robot is already at the target 60cm standoff distance.")

    def _execute_motion(self, dx: float, dy: float, dyaw_deg: float, travel_dist: float) -> None:
        """Execute sequential 2-stage motion: Rotate to face human FIRST, then walk straight forward."""
        with self.lock:
            if self.is_executing:
                return
            self.is_executing = True

        try:
            self.get_logger().info(
                f"Executing 2-stage approach: Target Distance={travel_dist:.2f}m, Heading Alignment={dyaw_deg:+.1f}°"
            )

            # 1. Publish to /g1/cmd_pose for ROS bridges
            cmd = Twist()
            cmd.linear.x = float(travel_dist)
            cmd.linear.y = 0.0
            cmd.angular.z = float(dyaw_deg)
            self.pub_cmd_pose.publish(cmd)

            # Stage 1: Rotate in place to face target directly FIRST
            if abs(dyaw_deg) > 3.0:
                self.get_logger().info(f"🔄 Stage 1: Rotating {dyaw_deg:+.1f}° to face human directly...")
                self._send_socket_command({"cmd": "rotate", "degrees": dyaw_deg, "yaw_rate": 0.50})
                time.sleep(0.8)

            # Stage 2: Walk straight forward (+X only) to reach standoff distance
            if travel_dist > self.min_thresh:
                self.get_logger().info(f"🚶 Stage 2: Walking straight forward {travel_dist:.2f}m at {self.linear_speed:.2f}m/s...")
                self._send_socket_command({"cmd": "move", "dx": travel_dist, "dy": 0.0, "speed": self.linear_speed})
                time.sleep(0.5)

            # Stage 3: Active Hard Stop
            self.get_logger().info(f"🛑 Stage 3: Arrived at standoff position. Fully halted.")
            self._send_socket_command({"cmd": "stop"})

            self.get_logger().info(f"[✓] Approach motion sequence completed successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to execute approach motion: {e}")
        finally:
            with self.lock:
                self.is_executing = False

    def _send_socket_command(self, req: dict, timeout: float = 15.0) -> Optional[dict]:
        """Send command directly to robot_bridge socket if available."""
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                sock.connect(self.socket_path)
                sock.sendall((json.dumps(req) + "\n").encode())
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                return json.loads(buf.decode())
        except (FileNotFoundError, ConnectionRefusedError):
            self.get_logger().debug(f"robot_bridge socket not found at {self.socket_path}; relying on /g1/cmd_pose topic.")
            return None
        except Exception as e:
            self.get_logger().debug(f"Socket send failed: {e}")
            return None

    def _publish_approach_markers(self, header, human_pos, goal_pos, yaw_rad: float) -> None:
        """Publish line trajectory and stopping disk in RViz."""
        markers = MarkerArray()

        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        # 1. Approach path line from robot to 60cm standoff goal
        line = Marker()
        line.header = header
        line.ns = "approach_path"
        line.id = 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.06  # Line width
        line.color.r = 0.0
        line.color.g = 0.9
        line.color.b = 1.0
        line.color.a = 0.9

        p_start = line.pose.position
        from geometry_msgs.msg import Point
        p0 = Point(x=0.0, y=0.0, z=0.0)
        p1 = Point(x=goal_pos[0], y=goal_pos[1], z=0.0)
        p_human = Point(x=human_pos.x, y=human_pos.y, z=0.0)
        line.points = [p0, p1]
        markers.markers.append(line)

        # 2. Dotted gap line representing the 60cm standoff
        gap = Marker()
        gap.header = header
        gap.ns = "standoff_gap"
        gap.id = 2
        gap.type = Marker.LINE_STRIP
        gap.action = Marker.ADD
        gap.scale.x = 0.04
        gap.color.r = 1.0
        gap.color.g = 0.2
        gap.color.b = 0.2
        gap.color.a = 0.7
        gap.points = [p1, p_human]
        markers.markers.append(gap)

        # 3. 60cm Standoff Target Stopping Pad
        pad = Marker()
        pad.header = header
        pad.ns = "stopping_pad"
        pad.id = 3
        pad.type = Marker.CYLINDER
        pad.action = Marker.ADD
        pad.pose.position.x = goal_pos[0]
        pad.pose.position.y = goal_pos[1]
        pad.pose.position.z = 0.02
        pad.pose.orientation.z = math.sin(yaw_rad / 2.0)
        pad.pose.orientation.w = math.cos(yaw_rad / 2.0)
        pad.scale.x = 0.5
        pad.scale.y = 0.5
        pad.scale.z = 0.04
        pad.color.r = 0.0
        pad.color.g = 1.0
        pad.color.b = 0.4
        pad.color.a = 0.8
        markers.markers.append(pad)

        # 4. Text Label at stopping pad
        text = Marker()
        text.header = header
        text.ns = "stopping_label"
        text.id = 4
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = goal_pos[0]
        text.pose.position.y = goal_pos[1]
        text.pose.position.z = 0.4
        text.pose.orientation.w = 1.0
        text.scale.z = 0.28
        text.text = f"Target Stop Point (60cm from human)"
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        markers.markers.append(text)

        self.pub_markers.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = HumanLocoApproachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
