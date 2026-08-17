#!/usr/bin/env python3
"""
G1 Arm Controller Node -- ROS 2 Node providing high-level arm action services
and topic interfaces for the Unitree G1 humanoid robot.

Services Exposed:
    /g1/arm/shake_hand   (std_srvs/srv/Trigger)  -> Executes handshake gesture
    /g1/arm/low_wave     (std_srvs/srv/Trigger)  -> Executes face/low wave gesture
    /g1/arm/wave         (std_srvs/srv/Trigger)  -> Alias for wave gesture
    /g1/arm/high_wave    (std_srvs/srv/Trigger)  -> Executes high wave gesture
    /g1/arm/release_arm  (std_srvs/srv/Trigger)  -> Releases arm to neutral position

Topics Subscribed:
    /g1/arm/action_cmd   (std_msgs/msg/String)   -> "shake_hand", "low_wave", "high_wave", "release_arm", "clap", "hug", etc.

Topics Published:
    /g1/arm/status       (std_msgs/msg/String)   -> Current arm action state ("IDLE", "SHAKING_HAND", "WAVING", etc.)
"""

import json
import socket
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

DEFAULT_SOCKET_PATH = "/tmp/g1_robot_bridge.sock"


class G1ArmControllerNode(Node):
    def __init__(self):
        super().__init__("g1_arm_controller_node")

        self.declare_parameter("socket_path", DEFAULT_SOCKET_PATH)
        self.declare_parameter("mock_mode", False)
        self.declare_parameter("default_hold_seconds", 3.0)

        self.socket_path = self.get_parameter("socket_path").value
        self.mock_mode = self.get_parameter("mock_mode").value
        self.default_hold_seconds = float(self.get_parameter("default_hold_seconds").value)

        self.lock = threading.Lock()
        self.current_action = "IDLE"

        # Publishers
        self.pub_status = self.create_publisher(String, "/g1/arm/status", 10)

        # Subscribers
        self.sub_cmd = self.create_subscription(
            String, "/g1/arm/action_cmd", self.on_action_cmd, 10
        )

        # Services
        self.srv_shake = self.create_service(Trigger, "/g1/arm/shake_hand", self.handle_shake_hand)
        self.srv_low_wave = self.create_service(Trigger, "/g1/arm/low_wave", self.handle_low_wave)
        self.srv_wave = self.create_service(Trigger, "/g1/arm/wave", self.handle_low_wave)
        self.srv_high_wave = self.create_service(Trigger, "/g1/arm/high_wave", self.handle_high_wave)
        self.srv_release = self.create_service(Trigger, "/g1/arm/release_arm", self.handle_release_arm)

        self._publish_status("IDLE")
        self.get_logger().info(
            f"G1ArmControllerNode active [Socket: {self.socket_path}, Mock: {self.mock_mode}]"
        )

    def _publish_status(self, status: str):
        self.current_action = status
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)

    def _send_bridge_command(self, req: dict, timeout: float = 12.0) -> dict:
        """Send JSON command to robot_bridge Unix Domain Socket."""
        if self.mock_mode:
            self.get_logger().info(f"[MOCK ARM] Simulating command: {req}")
            time.sleep(1.0)
            return {"ok": True, "mock": True}

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
            self.get_logger().warning(
                f"robot_bridge.py not reachable at {self.socket_path}. Falling back to simulation mock."
            )
            time.sleep(1.0)
            return {"ok": True, "simulated": True}
        except Exception as e:
            self.get_logger().error(f"Error communicating with robot_bridge: {e}")
            return {"ok": False, "error": str(e)}

    def execute_shake_hand(self, hold_seconds: float = 3.0) -> bool:
        with self.lock:
            self._publish_status("SHAKING_HAND")
            self.get_logger().info(f"🤝 Executing Handshake Gesture (hold={hold_seconds}s)...")

            res = self._send_bridge_command({
                "cmd": "shake_hand",
                "hold_seconds": hold_seconds,
                "auto_release": True
            })

            time.sleep(hold_seconds + 0.5)
            self._publish_status("IDLE")
            return res.get("ok", False)

    def execute_wave(self, wave_type: str = "face", hold_seconds: float = 3.0) -> bool:
        with self.lock:
            status_tag = "LOW_WAVING" if wave_type in ("face", "low") else "HIGH_WAVING"
            self._publish_status(status_tag)
            self.get_logger().info(f"👋 Executing Wave Gesture (type={wave_type}, hold={hold_seconds}s)...")

            res = self._send_bridge_command({
                "cmd": "wave",
                "wave_type": wave_type,
                "hold_seconds": hold_seconds,
                "auto_release": True
            })

            time.sleep(hold_seconds + 0.5)
            self._publish_status("IDLE")
            return res.get("ok", False)

    def execute_release_arm(self) -> bool:
        with self.lock:
            self._publish_status("RELEASING_ARM")
            res = self._send_bridge_command({"cmd": "release_arm"})
            self._publish_status("IDLE")
            return res.get("ok", False)

    # ---------------- Service Handlers ---------------- #
    def handle_shake_hand(self, request, response):
        success = self.execute_shake_hand(self.default_hold_seconds)
        response.success = success
        response.message = "Handshake executed successfully" if success else "Handshake failed"
        return response

    def handle_low_wave(self, request, response):
        success = self.execute_wave("face", self.default_hold_seconds)
        response.success = success
        response.message = "Low/face wave executed successfully" if success else "Wave failed"
        return response

    def handle_high_wave(self, request, response):
        success = self.execute_wave("high", self.default_hold_seconds)
        response.success = success
        response.message = "High wave executed successfully" if success else "High wave failed"
        return response

    def handle_release_arm(self, request, response):
        success = self.execute_release_arm()
        response.success = success
        response.message = "Arm released to neutral" if success else "Release failed"
        return response

    # ---------------- Topic Handler ---------------- #
    def on_action_cmd(self, msg: String):
        action = msg.data.lower().strip().replace(" ", "_")
        self.get_logger().info(f"Received arm action command topic: '{action}'")

        threading.Thread(target=self._dispatch_action, args=(action,), daemon=True).start()

    def _dispatch_action(self, action: str):
        if action in ("shake_hand", "shake", "handshake"):
            self.execute_shake_hand(self.default_hold_seconds)
        elif action in ("low_wave", "face_wave", "wave"):
            self.execute_wave("face", self.default_hold_seconds)
        elif action in ("high_wave", "wave_high"):
            self.execute_wave("high", self.default_hold_seconds)
        elif action in ("release", "release_arm", "neutral"):
            self.execute_release_arm()
        else:
            with self.lock:
                self._publish_status(f"EXECUTING_{action.upper()}")
                self._send_bridge_command({"cmd": "arm_action", "action": action})
                self._publish_status("IDLE")


def main(args=None):
    rclpy.init(args=args)
    node = G1ArmControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
