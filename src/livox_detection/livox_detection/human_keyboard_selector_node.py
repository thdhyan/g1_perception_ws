#!/usr/bin/env python3
"""
Interactive ROS 2 Keyboard Selector for Human Target Selection & Motion Control:
- Reads single raw keystrokes (termios non-blocking) like standard ROS teleop.
- Keybindings:
    [1-9] : Select human target #1 to #9 immediately.
    [R/SPACE] : Trigger a fresh LiDAR snapshot & CenterPoint rescan.
    [W]   : Trigger Wave greeting interaction.
    [S]   : Trigger Handshake greeting interaction.
    [C/0] : Clear current selection.
    [Q]   : Quit selector.
"""

import math
import os
import select
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


class HumanKeyboardSelectorNode(Node):
    def __init__(self):
        super().__init__("g1_human_keyboard_selector")

        self.declare_parameter("input_topic", "/g1/sorted_humans")
        self.input_topic = str(self.get_parameter("input_topic").value)

        self.lock = threading.Lock()
        self.current_humans: List[Detection3D] = []
        self.current_header = None
        self.selected_target: Optional[Detection3D] = None
        self.selected_rank: Optional[int] = None
        self.is_running = True

        # Latched QoS matching sorted humans publisher
        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.sub_sorted = self.create_subscription(
            Detection3DArray, self.input_topic, self.on_sorted_detections, latched_qos
        )

        # Publishers
        self.pub_select_id = self.create_publisher(Int32, "/g1/select_human_id", 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/g1/selected_human", 10)
        self.pub_index = self.create_publisher(Int32, "/g1/selected_human_index", 10)
        self.pub_marker = self.create_publisher(MarkerArray, "/g1/selected_human_marker", 10)
        self.pub_arm_cmd = self.create_publisher(String, "/g1/arm/action_cmd", 10)

        # Service Clients
        self.cli_rescan = self.create_client(Trigger, "/g1/rescan")
        self.cli_snapshot = self.create_client(Trigger, "/g1/publish_front_snapshot")

        self.get_logger().info(
            f"Interactive Keyboard Selector initialized. Listening to '{self.input_topic}'."
        )

    def on_sorted_detections(self, msg: Detection3DArray) -> None:
        """Handle new sorted human detections from snapshot pipeline."""
        with self.lock:
            self.current_humans = list(msg.detections)
            self.current_header = msg.header
            self._render_ui()

    def _render_ui(self) -> None:
        """Render clean terminal UI dashboard."""
        humans = self.current_humans
        os.system("clear")
        print("===================================================================")
        print("         🤖 UNITREE G1 HUMAN SELECTION & TELEOP SELECTOR            ")
        print("===================================================================")
        if not humans:
            print("  [Waiting for detection snapshot... Press 'R' or SPACE to scan]   ")
        else:
            print(f"  Detected Candidates: {len(humans)} person(s) sorted by distance:")
            print("  -----------------------------------------------------------------")
            for i, det in enumerate(humans, start=1):
                pos = det.bbox.center.position
                dist = math.hypot(pos.x, pos.y)
                score = det.results[0].hypothesis.score if det.results else 0.0
                tag = " 🌟 [SELECTED]" if self.selected_rank == i else (" (Closest)" if i == 1 else "")
                print(f"   [{i}]  Dist: {dist:5.2f}m | Pos: (x={pos.x:+5.2f}, y={pos.y:+5.2f}, z={pos.z:+5.2f}) | Conf: {score:.2f}{tag}")
            print("  -----------------------------------------------------------------")

        print("\n  KEYBINDINGS:")
        print("    [1-9]       : Lock onto Human #1..#9 & Start Locomotion Walk-Up")
        print("    [R / SPACE] : Take New LiDAR Snapshot / Rescan")
        print("    [W]         : Execute Low Wave Greeting")
        print("    [S]         : Execute Handshake Greeting")
        print("    [C / 0]     : Clear Target Selection")
        print("    [Q]         : Exit Selector")
        print("===================================================================")
        if self.selected_rank is not None:
            print(f"  >> CURRENT TARGET: Human #{self.selected_rank} <<")
        print("  Press any key: ", end="", flush=True)

    def select_human(self, rank: int) -> None:
        """Lock onto human target by rank (1-indexed)."""
        with self.lock:
            if not (1 <= rank <= len(self.current_humans)):
                print(f"\n[!] Human #{rank} is not available (have {len(self.current_humans)}).")
                return

            det = self.current_humans[rank - 1]
            self.selected_target = det
            self.selected_rank = rank
            pos = det.bbox.center.position
            dist = math.hypot(pos.x, pos.y)

            # Publish selection to snapshot pipeline which manages single target dispatch
            self.pub_select_id.publish(Int32(data=rank))
            self.pub_index.publish(Int32(data=rank))

            print(f"\n[✓] Target LOCKED on Human #{rank} (Dist: {dist:.2f}m). Approach initiated!")

    def clear_selection(self) -> None:
        with self.lock:
            self.selected_target = None
            self.selected_rank = None
            self.pub_select_id.publish(Int32(data=0))
            self.pub_index.publish(Int32(data=0))
            self._clear_beacon()
            print("\n[*] Target selection cleared.")

    def trigger_rescan(self) -> None:
        print("\n[*] Triggering new LiDAR snapshot & CenterPoint scan...")
        if self.cli_rescan.wait_for_service(timeout_sec=0.5):
            req = Trigger.Request()
            self.cli_rescan.call_async(req)
        elif self.cli_snapshot.wait_for_service(timeout_sec=0.5):
            req = Trigger.Request()
            self.cli_snapshot.call_async(req)

    def trigger_wave(self) -> None:
        print("\n[*] 👋 Sending Wave greeting command...")
        self.pub_arm_cmd.publish(String(data="low_wave"))

    def trigger_handshake(self) -> None:
        print("\n[*] 🤝 Sending Handshake greeting command...")
        self.pub_arm_cmd.publish(String(data="shake_hand"))

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
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main(args=None):
    rclpy.init(args=args)
    node = HumanKeyboardSelectorNode()

    # Spin ROS 2 in background executor thread
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Save original terminal settings for clean exit
    old_settings = termios.tcgetattr(sys.stdin)
    node._render_ui()

    try:
        while rclpy.ok():
            key = get_key(old_settings)
            if not key:
                continue

            if key in ("\x03", "q", "Q"):  # Ctrl+C or 'q'
                print("\nExiting Keyboard Selector...")
                break

            elif key in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
                node.select_human(int(key))

            elif key in ("r", "R", " "):
                node.trigger_rescan()

            elif key in ("w", "W"):
                node.trigger_wave()

            elif key in ("s", "S"):
                node.trigger_handshake()

            elif key in ("0", "c", "C"):
                node.clear_selection()

            node._render_ui()

    except Exception as e:
        print(f"\nSelector exception: {e}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
