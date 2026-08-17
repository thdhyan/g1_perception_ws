#!/usr/bin/env python3
"""Move robot by a relative (dx, dy) offset, via robot_bridge.py.

Does NOT import unitree_sdk2py directly -- rclpy's Node() and
unitree_sdk2py's ChannelFactoryInitialize() each load their own copy of
libddsc.so and segfault if both run in one process (confirmed while
debugging this node -- see
../../unitree_sdk2_python/docs/SDK_TROUBLESHOOTING.md). Instead this talks
to a standalone robot_bridge.py process over a Unix socket.

Start the bridge first, once, in its own terminal (not via ros2 run):

    python3 robot_bridge.py enp2s0

Then:

    ros2 run g1_perception move_to_xy --ros-args -p dx:=1.0

Parameters:
  dx          (float, default 1.0)               -- meters forward (robot body frame)
  dy          (float, default 0.0)                -- meters left  (robot body frame)
  speed       (float, default 0.3)                -- m/s travel speed
  socket_path (str,   default /tmp/g1_robot_bridge.sock) -- robot_bridge.py socket
"""
import json
import socket

import rclpy
from rclpy.node import Node

DEFAULT_SOCKET_PATH = "/tmp/g1_robot_bridge.sock"


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


class MoveToXY(Node):
    def __init__(self):
        super().__init__("move_to_xy")

        self.declare_parameter("dx", 1.0)
        self.declare_parameter("dy", 0.0)
        self.declare_parameter("speed", 0.3)
        self.declare_parameter("socket_path", DEFAULT_SOCKET_PATH)

        dx = float(self.get_parameter("dx").value)
        dy = float(self.get_parameter("dy").value)
        speed = float(self.get_parameter("speed").value)
        socket_path = self.get_parameter("socket_path").value

        self.get_logger().info(f"Requesting move dx={dx} dy={dy} speed={speed}")

        try:
            resp = send_command(
                socket_path, {"cmd": "move", "dx": dx, "dy": dy, "speed": speed}
            )
        except (ConnectionRefusedError, FileNotFoundError):
            self.get_logger().error(
                f"Could not reach robot_bridge.py at {socket_path}. "
                "Start it first: python3 robot_bridge.py <interface>"
            )
            return
        except Exception as e:
            self.get_logger().error(f"robot_bridge request failed: {e}")
            return

        if resp.get("ok"):
            self.get_logger().info("Move complete.")
        else:
            self.get_logger().error(f"robot_bridge error: {resp.get('error')}")


def main(args=None):
    rclpy.init(args=args)
    node = MoveToXY()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
