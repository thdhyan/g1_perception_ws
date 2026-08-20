#!/usr/bin/env python3
"""
Interactive ROS 2 keyboard console for the real G1: walking, human target
selection, and arm gestures from one terminal.

Keybindings:
    [W/A/S/D]   Walk forward / strafe left / back / strafe right
    [Q/E]       Turn left / right
    [SPACE]     Stop
    [Z/X]       Linear speed  -/+ 0.05 m/s
    [-/+]       Yaw rate      -/+ 0.10 rad/s
    [1-9]       Lock onto human target #1..#9
    [0/C]       Clear target selection
    [R]         Rescan: collect a fresh cloud and re-run detection
    [T]         Re-publish the frozen snapshot cloud, without re-detecting
    [V]         shake hand      [B]  high wave     [N]  clap
    [M]         high five       [,]  heart         [.]  hug
    [ESC/Ctrl+C] Quit

Motion is hold-to-move: a key sets the velocity and the terminal's own auto-repeat
keeps it alive, but the velocity decays to zero `key_hold_timeout` seconds after
the last keypress. That way releasing the key stops the robot instead of leaving
it walking, which matters more on a humanoid than the extra keystrokes cost.

Velocity goes straight to robot_bridge's Unix socket rather than through
/cmd_vel + cmd_vel_bridge, so this node is the only process needed alongside the
bridge itself.
"""

import json
import math
import os
import select
import socket
import sys
import termios
import threading
import time
import tty
from typing import List, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3D, Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

# The six gestures reachable from one keyboard row, chosen out of the sixteen the
# bridge exposes ({"cmd": "get_arm_actions"}) as the ones that read as a greeting
# to a person the robot has just walked up to.
ARM_ACTION_KEYS = {
    "v": ("shake_hand", "🤝 Shake hand"),
    "b": ("high_wave", "👋 High wave"),
    "n": ("clap", "👏 Clap"),
    "m": ("high_five", "🙌 High five"),
    ",": ("heart", "💗 Heart"),
    ".": ("hug", "🫂 Hug"),
}


class HumanKeyboardSelectorNode(Node):
    def __init__(self):
        super().__init__("g1_human_keyboard_selector")

        self.declare_parameter("input_topic", "/g1/sorted_humans")
        self.declare_parameter("socket_path", "/tmp/g1_robot_bridge.sock")
        self.declare_parameter("linear_speed", 0.30)
        self.declare_parameter("strafe_speed", 0.20)
        self.declare_parameter("yaw_rate", 0.50)
        self.declare_parameter("stream_hz", 10.0)
        self.declare_parameter("key_hold_timeout", 0.5)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.socket_path = str(self.get_parameter("socket_path").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.strafe_speed = float(self.get_parameter("strafe_speed").value)
        self.yaw_rate = float(self.get_parameter("yaw_rate").value)
        self.stream_hz = float(self.get_parameter("stream_hz").value)
        self.key_hold_timeout = float(self.get_parameter("key_hold_timeout").value)

        self.lock = threading.Lock()
        self.current_humans: List[Detection3D] = []
        self.current_header = None
        self.selected_target: Optional[Detection3D] = None
        self.selected_rank: Optional[int] = None
        self.status = "ready"

        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._last_move_key = 0.0
        self._was_moving = False

        # Latched QoS matching sorted humans publisher
        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_sorted = self.create_subscription(
            Detection3DArray, self.input_topic, self.on_sorted_detections, latched_qos
        )

        self.pub_select_id = self.create_publisher(Int32, "/g1/select_human_id", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/g1/selected_human", 10)
        self.pub_index = self.create_publisher(Int32, "/g1/selected_human_index", 10)
        self.pub_marker = self.create_publisher(MarkerArray, "/g1/selected_human_marker", 10)
        self.pub_arm_cmd = self.create_publisher(String, "/g1/arm/action_cmd", 10)

        self.cli_rescan = self.create_client(Trigger, "/g1/rescan")
        self.cli_snapshot = self.create_client(Trigger, "/g1/publish_front_snapshot")

        self.create_timer(1.0 / max(self.stream_hz, 1.0), self.on_velocity_timer)

    # ---------------------------------------------------------------- bridge

    def _bridge(self, req: dict, timeout: float = 2.0) -> Optional[dict]:
        """One line-delimited JSON round-trip to robot_bridge."""
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                sock.connect(self.socket_path)
                sock.sendall((json.dumps(req) + "\n").encode())
                return json.loads(sock.recv(4096).decode().strip() or "{}")
        except (ConnectionRefusedError, FileNotFoundError):
            self.status = f"robot_bridge not reachable at {self.socket_path}"
        except Exception as e:
            self.status = f"bridge error: {e}"
        return None

    # ---------------------------------------------------------------- motion

    def set_velocity(self, vx: float, vy: float, wz: float, label: str) -> None:
        with self.lock:
            self._vx, self._vy, self._wz = vx, vy, wz
            self._last_move_key = time.time()
            self.status = label

    def halt(self) -> None:
        with self.lock:
            self._vx = self._vy = self._wz = 0.0
            self.status = "stopped"
        self._bridge({"cmd": "stop"})
        self._was_moving = False

    def on_velocity_timer(self) -> None:
        """Stream the current velocity, decaying to zero once the key is released."""
        with self.lock:
            stale = (time.time() - self._last_move_key) > self.key_hold_timeout
            if stale and (self._vx or self._vy or self._wz):
                self._vx = self._vy = self._wz = 0.0
                self.status = "idle (key released)"
            vx, vy, wz = self._vx, self._vy, self._wz

        moving = bool(vx or vy or wz)
        if moving:
            self._bridge({"cmd": "set_velocity", "vx": vx, "vy": vy, "wz": wz,
                          "duration": max(0.2, 2.0 / max(self.stream_hz, 1.0))})
            self._was_moving = True
        elif self._was_moving:
            # Edge-triggered: one stop when motion ends, not a stop every tick.
            self._bridge({"cmd": "stop"})
            self._was_moving = False

    def adjust_linear(self, delta: float) -> None:
        self.linear_speed = max(0.05, min(1.0, self.linear_speed + delta))
        self.strafe_speed = max(0.05, min(0.8, self.strafe_speed + delta))
        self.status = f"linear speed {self.linear_speed:.2f} m/s"

    def adjust_yaw(self, delta: float) -> None:
        self.yaw_rate = max(0.1, min(1.5, self.yaw_rate + delta))
        self.status = f"yaw rate {self.yaw_rate:.2f} rad/s"

    # ------------------------------------------------------------ selection

    def on_sorted_detections(self, msg: Detection3DArray) -> None:
        with self.lock:
            self.current_humans = list(msg.detections)
            self.current_header = msg.header

    def select_human(self, rank: int) -> None:
        with self.lock:
            if not (1 <= rank <= len(self.current_humans)):
                self.status = f"human #{rank} not available (have {len(self.current_humans)})"
                return
            det = self.current_humans[rank - 1]
            self.selected_target = det
            self.selected_rank = rank
            pos = det.bbox.center.position
            dist = math.hypot(pos.x, pos.y)

        self.pub_select_id.publish(Int32(data=rank))
        self.pub_index.publish(Int32(data=rank))
        self._publish_beacon(det, rank)
        self.status = f"target locked: human #{rank} at {dist:.2f}m"

    def clear_selection(self) -> None:
        with self.lock:
            self.selected_target = None
            self.selected_rank = None
        self.pub_select_id.publish(Int32(data=0))
        self.pub_index.publish(Int32(data=0))
        self._clear_beacon()
        self.status = "selection cleared"

    def trigger_rescan(self) -> None:
        """/g1/rescan -- collect a fresh cloud AND re-run detection on it."""
        if self.cli_rescan.wait_for_service(timeout_sec=0.5):
            self.cli_rescan.call_async(Trigger.Request())
            self.status = "rescan requested (new cloud + detection)"
        else:
            self.status = "no /g1/rescan service (is the snapshot pipeline running?)"

    def trigger_snapshot(self) -> None:
        """/g1/publish_front_snapshot -- re-publish the frozen cloud without
        re-running detection. Separate from rescan because when a detection looks
        wrong the useful question is whether the cloud or the model is at fault."""
        if self.cli_snapshot.wait_for_service(timeout_sec=0.5):
            self.cli_snapshot.call_async(Trigger.Request())
            self.status = "snapshot re-published"
        else:
            self.status = "no /g1/publish_front_snapshot service"

    # ----------------------------------------------------------------- arms

    def trigger_arm_action(self, action: str, label: str) -> None:
        self.status = f"{label}..."
        self.pub_arm_cmd.publish(String(data=action))
        resp = self._bridge({"cmd": "arm_action", "action": action}, timeout=5.0)
        if resp is not None and not resp.get("ok", False):
            self.status = f"{label} rejected: {resp.get('error')}"
            return
        # arm_action has no auto-release, unlike the bridge's shake_hand/wave.
        time.sleep(3.0)
        self._bridge({"cmd": "release_arm"}, timeout=5.0)
        self.status = f"{label} done, arm released"

    # ----------------------------------------------------------------- view

    def render(self) -> None:
        with self.lock:
            humans = list(self.current_humans)
            rank = self.selected_rank
            vx, vy, wz = self._vx, self._vy, self._wz
            status = self.status

        os.system("clear")
        print("===================================================================")
        print("        UNITREE G1 -- TELEOP / SELECTION / GESTURE CONSOLE         ")
        print("===================================================================")
        if not humans:
            print("  no humans on the selection topic yet -- press [R] to rescan")
        else:
            print(f"  {len(humans)} candidate(s), sorted by distance:")
            for i, det in enumerate(humans, start=1):
                pos = det.bbox.center.position
                dist = math.hypot(pos.x, pos.y)
                score = det.results[0].hypothesis.score if det.results else 0.0
                tag = "  <== SELECTED" if rank == i else ""
                print(f"   [{i}] {dist:5.2f}m  (x={pos.x:+5.2f}, y={pos.y:+5.2f})  conf {score:.2f}{tag}")

        print("\n  MOVE   [W/A/S/D] walk/strafe   [Q/E] turn   [SPACE] stop")
        print(f"         [Z/X] speed {self.linear_speed:4.2f} m/s   [-/+] yaw {self.yaw_rate:4.2f} rad/s")
        print("  TARGET [1-9] select   [0/C] clear   [R] rescan   [T] re-publish snapshot")
        print("  ARMS   " + "   ".join(f"[{k.upper()}] {lbl.split(' ', 1)[1]}"
                                       for k, (_, lbl) in ARM_ACTION_KEYS.items()))
        print("  [ESC] quit")
        print("===================================================================")
        print(f"  vel: vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}   |   {status}")
        print("  key: ", end="", flush=True)

    def _publish_beacon(self, det: Detection3D, rank: int) -> None:
        pos = det.bbox.center.position
        markers = MarkerArray()
        beam = Marker()
        beam.header = det.header
        beam.ns = "selected_beacon"
        beam.id = 100
        beam.type = Marker.CYLINDER
        beam.action = Marker.ADD
        beam.pose.position.x = pos.x
        beam.pose.position.y = pos.y
        beam.pose.position.z = pos.z + 1.2
        beam.scale.x = 0.25
        beam.scale.y = 0.25
        beam.scale.z = 2.4
        beam.color.r = 1.0
        beam.color.g = 0.84
        beam.color.b = 0.0
        beam.color.a = 0.8
        markers.markers.append(beam)
        self.pub_marker.publish(markers)

    def _clear_beacon(self) -> None:
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        self.pub_marker.publish(markers)


def get_key(settings):
    """Read single non-blocking keystroke from raw stdin."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    key = sys.stdin.read(1) if rlist else ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main(args=None):
    rclpy.init(args=args)
    node = HumanKeyboardSelectorNode()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    old_settings = termios.tcgetattr(sys.stdin)
    node.render()
    last_render = 0.0

    try:
        while rclpy.ok():
            key = get_key(old_settings)

            if key:
                lower = key.lower()
                if key in ("\x03", "\x1b"):  # Ctrl+C, ESC
                    break

                elif lower == "w":
                    node.set_velocity(node.linear_speed, 0.0, 0.0, "forward")
                elif lower == "s":
                    node.set_velocity(-node.linear_speed, 0.0, 0.0, "backward")
                elif lower == "a":
                    node.set_velocity(0.0, node.strafe_speed, 0.0, "strafe left")
                elif lower == "d":
                    node.set_velocity(0.0, -node.strafe_speed, 0.0, "strafe right")
                elif lower == "q":
                    node.set_velocity(0.0, 0.0, node.yaw_rate, "turn left")
                elif lower == "e":
                    node.set_velocity(0.0, 0.0, -node.yaw_rate, "turn right")
                elif key == " ":
                    node.halt()

                elif lower == "z":
                    node.adjust_linear(-0.05)
                elif lower == "x":
                    node.adjust_linear(+0.05)
                elif key == "-":
                    node.adjust_yaw(-0.10)
                elif key in ("+", "="):
                    node.adjust_yaw(+0.10)

                elif key in "123456789":
                    node.select_human(int(key))
                elif key == "0" or lower == "c":
                    node.clear_selection()
                elif lower == "r":
                    node.trigger_rescan()
                elif lower == "t":
                    node.trigger_snapshot()

                elif lower in ARM_ACTION_KEYS:
                    action, label = ARM_ACTION_KEYS[lower]
                    node.trigger_arm_action(action, label)

            # Redraw on input, and periodically so the velocity readout and any
            # newly arrived detections do not sit stale on screen.
            now = time.time()
            if key or (now - last_render) > 0.5:
                node.render()
                last_render = now

    except Exception as e:
        print(f"\nconsole exception: {e}")
    finally:
        node.halt()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
