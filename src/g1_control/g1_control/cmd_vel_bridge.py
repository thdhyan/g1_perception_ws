#!/usr/bin/env python3
"""
G1 cmd_vel bridge -- ROS2 node that subscribes to /cmd_vel (Twist) from Nav2
controller and forwards velocity commands to robot_bridge.py via Unix socket.

Unlike cmd_pose_bridge (which sends relative pose deltas), this node sends
continuous velocity commands at a fixed publish_hz rate. Each message is only
valid for deadman_timeout seconds; if no new /cmd_vel is received within that
time, a stop command is sent.

Start robot_bridge.py first, in its own terminal, outside ROS:
    conda activate unitree_mujoco
    python3 robot_bridge.py enp2s0

Then:
    ros2 run g1_control cmd_vel_bridge
"""
import json
import socket
import threading
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


class CmdVelBridge(Node):
    def __init__(self):
        super().__init__("g1_cmd_vel_bridge")

        self.declare_parameter("socket_path", DEFAULT_SOCKET_PATH)
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("deadman_timeout", 0.5)
        self.declare_parameter("vx_scale", 1.0)
        self.declare_parameter("vy_scale", 1.0)
        self.declare_parameter("wz_scale", 1.0)

        self.socket_path = self.get_parameter("socket_path").value
        publish_hz = self.get_parameter("publish_hz").value
        self.deadman_timeout = self.get_parameter("deadman_timeout").value
        self.vx_scale = self.get_parameter("vx_scale").value
        self.vy_scale = self.get_parameter("vy_scale").value
        self.wz_scale = self.get_parameter("wz_scale").value

        self.latest_twist = None
        self.latest_twist_time = None
        self._lock = threading.Lock()

        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)

        period = 1.0 / publish_hz
        self.create_timer(period, self.on_publish_timer)

        self.get_logger().info(
            f"Listening on /cmd_vel, forwarding to robot_bridge at {self.socket_path} "
            f"at {publish_hz} Hz with deadman_timeout={self.deadman_timeout}s"
        )

    def on_cmd_vel(self, msg: Twist):
        """Store latest Twist message with receive timestamp."""
        with self._lock:
            self.latest_twist = msg
            self.latest_twist_time = time.time()

    def on_publish_timer(self):
        """Publish move or stop command at fixed rate."""
        with self._lock:
            now = time.time()

            # Check if we have a fresh Twist message
            if (self.latest_twist is not None and
                    self.latest_twist_time is not None and
                    (now - self.latest_twist_time) < self.deadman_timeout):
                # Send move command
                vx = self.latest_twist.linear.x * self.vx_scale
                vy = self.latest_twist.linear.y * self.vy_scale
                wz = self.latest_twist.angular.z * self.wz_scale

                self.get_logger().debug(f"cmd_vel -> vx={vx} vy={vy} wz={wz}")

                try:
                    resp = send_command_with_retry(
                        self.socket_path,
                        {"command": "move", "vx": vx, "vy": vy, "wz": wz},
                        self.get_logger(),
                    )
                    if not resp.get("ok"):
                        self.get_logger().warn(f"move command failed: {resp.get('error')}")
                except (ConnectionRefusedError, FileNotFoundError):
                    self.get_logger().warn(
                        f"Could not reach robot_bridge.py at {self.socket_path}. "
                        "Start it first: python3 robot_bridge.py <interface>"
                    )
                except Exception as e:
                    self.get_logger().warn(f"robot_bridge move request failed: {e}")
            else:
                # No fresh Twist or timeout -- send stop command
                self.get_logger().debug("cmd_vel deadman timeout, sending stop")
                try:
                    resp = send_command_with_retry(
                        self.socket_path,
                        {"command": "stop"},
                        self.get_logger(),
                    )
                    if not resp.get("ok"):
                        self.get_logger().warn(f"stop command failed: {resp.get('error')}")
                except (ConnectionRefusedError, FileNotFoundError):
                    self.get_logger().warn(
                        f"Could not reach robot_bridge.py at {self.socket_path}. "
                        "Start it first: python3 robot_bridge.py <interface>"
                    )
                except Exception as e:
                    self.get_logger().warn(f"robot_bridge stop request failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
