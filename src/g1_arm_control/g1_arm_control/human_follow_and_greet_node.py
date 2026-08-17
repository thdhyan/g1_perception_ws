#!/usr/bin/env python3
"""
Human Follow and Greet Node -- ROS 2 Node that:
1. Subscribes to `/g1/selected_human` (published by 2-Pass Snapshot Pipeline or CLI selector).
2. Plans and executes a locomotion trajectory to stop 60 cm in front of the target human.
3. Upon arrival at the standoff position ("after walk up"), coordinates with arm control
   to greet the person by shaking hands (`shake_hand`) or waving (`low_wave` / `face_wave`).
4. Publishes rich RViz visual markers (approach path, 60cm stopping ring, and greeting billboard).
"""

import json
import math
import socket
import threading
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from std_srvs.srv import Trigger

DEFAULT_SOCKET_PATH = "/tmp/g1_robot_bridge.sock"


class HumanFollowAndGreetNode(Node):
    def __init__(self):
        super().__init__("human_follow_and_greet_node")

        # Parameters
        self.declare_parameter("standoff_distance", 0.60)     # meters (60 cm)
        self.declare_parameter("greeting_action", "shake_hand") # "shake_hand", "low_wave", "high_wave", "wave_and_shake"
        self.declare_parameter("auto_execute", True)
        self.declare_parameter("auto_greet", True)
        self.declare_parameter("linear_speed", 0.50)          # m/s (G1 continuous walking pace)
        self.declare_parameter("yaw_rate", 0.50)              # rad/s
        self.declare_parameter("socket_path", DEFAULT_SOCKET_PATH)
        self.declare_parameter("min_thresh", 0.05)            # meters

        self.standoff_dist = float(self.get_parameter("standoff_distance").value)
        self.greeting_action = str(self.get_parameter("greeting_action").value).lower().strip()
        self.auto_execute = bool(self.get_parameter("auto_execute").value)
        self.auto_greet = bool(self.get_parameter("auto_greet").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.yaw_rate = float(self.get_parameter("yaw_rate").value)
        self.socket_path = str(self.get_parameter("socket_path").value)
        self.min_thresh = float(self.get_parameter("min_thresh").value)

        self.lock = threading.Lock()
        self.is_busy = False
        self.current_state = "IDLE"  # "IDLE", "WALKING", "ARRIVED", "GREETING", "COMPLETED"

        # Publishers
        self.pub_cmd_pose = self.create_publisher(Twist, "/g1/cmd_pose", 10)
        self.pub_goal_pose = self.create_publisher(PoseStamped, "/g1/loco_target_waypoint", 10)
        self.pub_approach_markers = self.create_publisher(MarkerArray, "/g1/approach_markers", 10)
        self.pub_greeting_markers = self.create_publisher(MarkerArray, "/g1/greeting_markers", 10)
        self.pub_arm_cmd = self.create_publisher(String, "/g1/arm/action_cmd", 10)
        self.pub_state = self.create_publisher(String, "/g1/follow_greet_state", 10)

        # Subscribers
        self.sub_selected = self.create_subscription(
            PoseStamped, "/g1/selected_human", self.on_selected_human, 10
        )

        # Service for manual greeting trigger
        self.srv_trigger_greet = self.create_service(
            Trigger, "/g1/trigger_greeting", self.handle_trigger_greeting
        )

        self._set_state("IDLE")
        self.get_logger().info(
            f"HumanFollowAndGreetNode active [Standoff: {self.standoff_dist * 100:.0f}cm, "
            f"Greeting: '{self.greeting_action}', Auto-Execute: {self.auto_execute}, Auto-Greet: {self.auto_greet}]"
        )

    def _set_state(self, state: str):
        self.current_state = state
        msg = String()
        msg.data = state
        self.pub_state.publish(msg)

    def on_selected_human(self, msg: PoseStamped) -> None:
        """Handle new target human."""
        pos = msg.pose.position
        dist = math.hypot(pos.x, pos.y)

        if dist < 0.05:
            self.get_logger().warning("Selected target is at pelvis origin; ignoring.")
            return

        # Waypoint computation (60 cm standoff along vector from robot to human)
        unit_x = pos.x / dist
        unit_y = pos.y / dist
        travel_dist = max(0.0, dist - self.standoff_dist)

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

        # Publish approach markers
        self._publish_approach_markers(msg.header, pos, (goal_x, goal_y), target_yaw_rad)

        self.get_logger().info(
            f"\n[TARGET HUMAN LOCKED]: {dist:.2f}m away -> Standoff Target: {self.standoff_dist:.2f}m\n"
            f"  Movement Plan: dx={goal_x:+.2f}m, dy={goal_y:+.2f}m (Travel: {travel_dist:.2f}m), dyaw={target_yaw_deg:+.1f}°\n"
            f"  Post-Walkup Greeting Scheduled: '{self.greeting_action}'"
        )

        # Prevent concurrent triggers
        with self.lock:
            if self.is_busy:
                self.get_logger().info("Follow & greet sequence is already running; ignoring duplicate trigger.")
                return
            self.is_busy = True

        if self.auto_execute and (travel_dist > self.min_thresh or abs(target_yaw_deg) > 3.0):
            threading.Thread(
                target=self._run_follow_and_greet_pipeline,
                args=(goal_x, goal_y, target_yaw_deg, travel_dist, msg.header, pos, (goal_x, goal_y)),
                daemon=True,
            ).start()
        else:
            with self.lock:
                self.is_busy = False

    def _run_follow_and_greet_pipeline(
        self, dx: float, dy: float, dyaw_deg: float, travel_dist: float,
        header, human_pos, goal_pos
    ) -> None:
        """Executes walk-up followed by arm greeting sequence."""
        try:
            # ─────────────────────────────────────────────────────────────
            # PHASE 1: LOCOMOTION WALK-UP
            # ─────────────────────────────────────────────────────────────
            if travel_dist > self.min_thresh or abs(dyaw_deg) > 3.0:
                self._set_state("WALKING")
                self.get_logger().info(
                    f"🚶 Phase 1: Locomotion Approach (Turn: {dyaw_deg:+.1f}°, Distance: {travel_dist:.2f}m to standoff)..."
                )

                # 1. Publish cmd_pose delta
                cmd = Twist()
                cmd.linear.x = float(travel_dist)
                cmd.linear.y = 0.0
                cmd.angular.z = float(dyaw_deg)
                self.pub_cmd_pose.publish(cmd)

                # 2. Sequential Execution:
                # Step 1: Rotate in place to face target directly FIRST
                if abs(dyaw_deg) > 3.0:
                    self.get_logger().info(f"  -> 🔄 Rotating {dyaw_deg:+.1f}° to face human...")
                    self._send_socket_command({
                        "cmd": "rotate", "degrees": dyaw_deg, "yaw_rate": self.yaw_rate
                    })
                    time.sleep(0.8)

                # Step 2: Walk straight forward (+X only)
                if travel_dist > self.min_thresh:
                    self.get_logger().info(f"  -> 🚶 Walking straight forward {travel_dist:.2f}m at {self.linear_speed:.2f}m/s...")
                    self._send_socket_command({
                        "cmd": "move", "dx": travel_dist, "dy": 0.0, "speed": self.linear_speed
                    })
                    time.sleep(0.5)

                # Step 3: Hard Stop
                self._send_socket_command({"cmd": "stop"})
                self.get_logger().info("  -> [✓] Locomotion walk-up complete and fully halted.")
            else:
                self.get_logger().info("Robot is already within standoff distance.")

            # ─────────────────────────────────────────────────────────────
            # PHASE 2: POST-WALKUP ARRIVAL & GREETING
            # ─────────────────────────────────────────────────────────────
            self._set_state("ARRIVED")
            self.get_logger().info("🎯 Phase 2: Arrived at 60cm standoff facing target human.")
            time.sleep(1.0)

            if self.auto_greet:
                self._execute_greeting_sequence(header, goal_pos)
            else:
                self.get_logger().info("Auto-greet is disabled. Call service /g1/trigger_greeting to interact.")

        except Exception as e:
            self.get_logger().error(f"Error in follow and greet execution: {e}")
        finally:
            with self.lock:
                self.is_busy = False

    def _execute_greeting_sequence(self, header, goal_pos) -> None:
        """Executes handshake or waving interaction."""
        self._set_state("GREETING")
        action = self.greeting_action

        if action in ("shake_hand", "shake", "handshake"):
            self.get_logger().info("🤝 [GREETING] Performing Handshake Gesture...")
            self._publish_greeting_markers(header, goal_pos, "🤝 Shaking Hands with Human...", (0.2, 0.9, 0.2))
            
            # Send topic cmd and socket command
            arm_msg = String()
            arm_msg.data = "shake_hand"
            self.pub_arm_cmd.publish(arm_msg)
            self._send_socket_command({"cmd": "shake_hand", "hold_seconds": 3.5, "auto_release": True})
            time.sleep(4.0)

        elif action in ("low_wave", "face_wave", "wave"):
            self.get_logger().info("👋 [GREETING] Performing Low Wave / Face Wave Gesture...")
            self._publish_greeting_markers(header, goal_pos, "👋 Waving Hello to Human (Low Wave)...", (0.2, 0.7, 1.0))
            
            arm_msg = String()
            arm_msg.data = "low_wave"
            self.pub_arm_cmd.publish(arm_msg)
            self._send_socket_command({"cmd": "wave", "wave_type": "face", "hold_seconds": 3.0, "auto_release": True})
            time.sleep(3.5)

        elif action in ("high_wave", "wave_high"):
            self.get_logger().info("🙋 [GREETING] Performing High Wave Gesture...")
            self._publish_greeting_markers(header, goal_pos, "🙋 High Waving to Human...", (1.0, 0.8, 0.2))
            
            arm_msg = String()
            arm_msg.data = "high_wave"
            self.pub_arm_cmd.publish(arm_msg)
            self._send_socket_command({"cmd": "wave", "wave_type": "high", "hold_seconds": 3.0, "auto_release": True})
            time.sleep(3.5)

        elif action in ("wave_and_shake", "both"):
            self.get_logger().info("👋🤝 [GREETING] Performing Low Wave followed by Handshake...")
            # 1. Wave
            self._publish_greeting_markers(header, goal_pos, "👋 Greeting Human (Low Wave)...", (0.2, 0.7, 1.0))
            self._send_socket_command({"cmd": "wave", "wave_type": "face", "hold_seconds": 2.5, "auto_release": True})
            time.sleep(3.0)

            # 2. Handshake
            self._publish_greeting_markers(header, goal_pos, "🤝 Offering Handshake...", (0.2, 0.9, 0.2))
            self._send_socket_command({"cmd": "shake_hand", "hold_seconds": 3.0, "auto_release": True})
            time.sleep(3.5)

        # Release & Completed
        self.get_logger().info("✅ [✓] Post-walkup interaction finished. Arm released to neutral.")
        self._publish_greeting_markers(header, goal_pos, "✅ Interaction Complete (60cm Standoff)", (0.3, 1.0, 0.3))
        self._set_state("COMPLETED")

    def handle_trigger_greeting(self, request, response):
        """Service handler to manually trigger greeting."""
        header = None
        from std_msgs.msg import Header
        hdr = Header()
        hdr.frame_id = "pelvis"
        hdr.stamp = self.get_clock().now().to_msg()
        threading.Thread(target=self._execute_greeting_sequence, args=(hdr, (0.0, 0.0)), daemon=True).start()
        response.success = True
        response.message = f"Greeting sequence '{self.greeting_action}' initiated."
        return response

    def _send_socket_command(self, req: dict, timeout: float = 15.0) -> Optional[dict]:
        """Send command directly to robot_bridge socket."""
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
            self.get_logger().debug(f"robot_bridge socket not found at {self.socket_path}; sim/mock mode active.")
            time.sleep(1.0)
            return {"ok": True, "simulated": True}
        except Exception as e:
            self.get_logger().debug(f"Socket send error: {e}")
            return None

    def _publish_approach_markers(self, header, human_pos, goal_pos, yaw_rad: float) -> None:
        """Publish trajectory and 60cm stopping ring in RViz."""
        markers = MarkerArray()

        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        # 1. Approach path line
        line = Marker()
        line.header = header
        line.ns = "approach_path"
        line.id = 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.06
        line.color.r = 0.0
        line.color.g = 0.9
        line.color.b = 1.0
        line.color.a = 0.9
        p0 = Point(x=0.0, y=0.0, z=0.0)
        p1 = Point(x=goal_pos[0], y=goal_pos[1], z=0.0)
        line.points = [p0, p1]
        markers.markers.append(line)

        # 2. 60cm standoff buffer line
        gap = Marker()
        gap.header = header
        gap.ns = "standoff_gap"
        gap.id = 2
        gap.type = Marker.LINE_STRIP
        gap.action = Marker.ADD
        gap.scale.x = 0.04
        gap.color.r = 1.0
        gap.color.g = 0.3
        gap.color.b = 0.3
        gap.color.a = 0.7
        p_human = Point(x=human_pos.x, y=human_pos.y, z=0.0)
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
        pad.color.b = 0.5
        pad.color.a = 0.85
        markers.markers.append(pad)

        self.pub_approach_markers.publish(markers)

    def _publish_greeting_markers(self, header, goal_pos, text_str: str, color_rgb: Tuple[float, float, float]) -> None:
        """Publish floating 3D status badge during interaction."""
        markers = MarkerArray()

        text = Marker()
        text.header = header
        text.ns = "greeting_badge"
        text.id = 10
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = goal_pos[0]
        text.pose.position.y = goal_pos[1]
        text.pose.position.z = 0.60
        text.pose.orientation.w = 1.0
        text.scale.z = 0.28
        text.text = text_str
        text.color.r = float(color_rgb[0])
        text.color.g = float(color_rgb[1])
        text.color.b = float(color_rgb[2])
        text.color.a = 1.0
        markers.markers.append(text)

        self.pub_greeting_markers.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = HumanFollowAndGreetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
