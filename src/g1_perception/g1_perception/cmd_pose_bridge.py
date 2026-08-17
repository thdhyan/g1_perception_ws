#!/usr/bin/env python3
"""
G1 cmd_pose bridge -- ROS2 node, subscribes to a Twist topic (used as a
relative pose delta, not a velocity), forwards each command to
robot_bridge.py (the standalone process that owns the actual
unitree_sdk2py LocoClient connection).

Pure rclpy here -- does NOT import unitree_sdk2py. rclpy's Node() and
unitree_sdk2py's LocoClient() cannot both live in one process: creating a
LocoClient (its own DDS reader/writer pair) while an rclpy Node's DDS
participant is already active reliably segfaults. Confirmed by isolating
the crash to the `LocoClient()` constructor call specifically -- see
../../unitree_sdk2_python/docs/SDK_TROUBLESHOOTING.md.

Any node in the ROS graph (slam_toolbox, a joystick teleop node, a planner)
can drive the robot by publishing geometry_msgs/Twist to /g1/cmd_pose:
    linear.x  -> dx (m,  forward)
    linear.y  -> dy (m,  left)
    angular.z -> dyaw (deg, positive = counter-clockwise)

Named cmd_pose (not cmd_vel) deliberately: each message is a relative POSE
DELTA, applied once as a blocking move -- not a continuous velocity stream
published at a rate like a typical /cmd_vel. Fine for waypoint-style
commands; for continuous teleop-style driving, extend robot_bridge.py with
a streaming velocity command instead of point-to-point move/rotate.

Start robot_bridge.py first, in its own terminal, outside ROS:
    conda activate unitree_mujoco   # or any env with cyclonedds==0.10.2
    python3 robot_bridge.py enp2s0

Then:
    ros2 run g1_perception cmd_pose_bridge
    ros2 topic pub /g1/cmd_pose geometry_msgs/Twist \
      "{linear: {x: 1.0, y: 0.0}, angular: {z: 0.0}}" --once
"""
import json
import socket
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

DEFAULT_SOCKET_PATH = "/tmp/g1_robot_bridge.sock"
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds between retries


def send_command(socket_path: str, req: dict, timeout: float = 30.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(socket_path)
        sock.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode())


def send_command_with_retry(socket_path: str, req: dict, logger,
                             max_retries: int = MAX_RETRIES,
                             timeout: float = 30.0) -> dict:
    """Retry on socket-level failures (timeout, connection dropped mid-call)
    -- separate from robot_bridge.py's own internal FSM-transition retry.
    Does NOT retry ConnectionRefusedError/FileNotFoundError (bridge not
    running at all) -- that's a setup problem, not a transient one."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return send_command(socket_path, req, timeout)
        except (ConnectionRefusedError, FileNotFoundError):
            raise
        except Exception as e:
            last_error = e
            logger.warn(
                f"send_command attempt {attempt}/{max_retries} failed: {e}"
            )
            if attempt < max_retries:
                time.sleep(RETRY_DELAY)
    return {"ok": False, "error": f"gave up after {max_retries} attempts: {last_error}"}


class CmdPoseBridge(Node):
    def __init__(self):
        super().__init__("cmd_pose_bridge")

        self.declare_parameter("socket_path", DEFAULT_SOCKET_PATH)
        self.declare_parameter("topic", "/g1/cmd_pose")

        self.socket_path = self.get_parameter("socket_path").value
        topic = self.get_parameter("topic").value

        self.create_subscription(Twist, topic, self.on_cmd_pose, 10)
        self.get_logger().info(
            f"Listening on {topic}, forwarding to robot_bridge at {self.socket_path}"
        )

    def on_cmd_pose(self, msg: Twist):
        dx = msg.linear.x
        dy = msg.linear.y
        dyaw_deg = msg.angular.z  # degrees, per this node's convention

        self.get_logger().info(f"cmd_pose -> dx={dx} dy={dy} dyaw={dyaw_deg}deg")

        try:
            if dx != 0.0 or dy != 0.0:
                resp = send_command_with_retry(
                    self.socket_path, {"cmd": "move", "dx": dx, "dy": dy},
                    self.get_logger(),
                )
                if not resp.get("ok"):
                    self.get_logger().error(f"move failed: {resp.get('error')}")

            if dyaw_deg != 0.0:
                resp = send_command_with_retry(
                    self.socket_path, {"cmd": "rotate", "degrees": dyaw_deg},
                    self.get_logger(),
                )
                if not resp.get("ok"):
                    self.get_logger().error(f"rotate failed: {resp.get('error')}")
        except (ConnectionRefusedError, FileNotFoundError):
            self.get_logger().error(
                f"Could not reach robot_bridge.py at {self.socket_path}. "
                "Start it first: python3 robot_bridge.py <interface>"
            )
        except Exception as e:
            self.get_logger().error(f"robot_bridge request failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CmdPoseBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
