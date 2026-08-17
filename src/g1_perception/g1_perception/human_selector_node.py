#!/usr/bin/env python3
"""
human_selector_node.py

ROS2 node that detects humans from CenterPoint detection output, presents a
numbered CLI menu for user selection, and publishes the selected human's pose.

Subscribes:
    /g1/detections/centerpoint  (vision_msgs/Detection3DArray)

Publishes:
    /g1/selected_human          (geometry_msgs/PoseStamped)
    /g1/selected_human_index    (std_msgs/Int32)
    /g1/human_markers           (visualization_msgs/MarkerArray)
"""

import threading
import time
from math import sqrt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from visualization_msgs.msg import MarkerArray, Marker
from builtin_interfaces.msg import Duration


class HumanSelectorNode(Node):
    """Detects humans, shows a CLI menu, and publishes selected human pose."""

    def __init__(self):
        super().__init__('g1_human_selector')

        # ---------- Parameters ----------
        self.declare_parameter('detection_topic', '/g1/detections/centerpoint')
        self.declare_parameter('min_score', 0.3)
        self.declare_parameter('redetect_cooldown_sec', 3.0)

        self.detection_topic = self.get_parameter('detection_topic').value
        self.min_score = self.get_parameter('min_score').value
        self.cooldown_sec = self.get_parameter('redetect_cooldown_sec').value

        # ---------- State (thread-safe) ----------
        self.lock = threading.Lock()
        self.last_prompt_time = 0.0
        self.current_humans = []  # List of (distance, detection) tuples, sorted by distance
        self.selected_idx = None

        # ---------- Subscriptions ----------
        self.sub_detections = self.create_subscription(
            Detection3DArray,
            self.detection_topic,
            self._detection_callback,
            QoSPresetProfiles.SENSOR_DATA.value
        )

        # ---------- Publishers ----------
        self.pub_pose = self.create_publisher(
            PoseStamped, '/g1/selected_human', QoSPresetProfiles.DEFAULT.value)
        self.pub_index = self.create_publisher(
            Int32, '/g1/selected_human_index', QoSPresetProfiles.DEFAULT.value)
        self.pub_markers = self.create_publisher(
            MarkerArray, '/g1/human_markers', QoSPresetProfiles.DEFAULT.value)

        # ---------- Input thread ----------
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info(
            f"human_selector_node started. "
            f"Subscribing: {self.detection_topic}. "
            f"min_score={self.min_score}, cooldown={self.cooldown_sec}s"
        )

    # ------------------------------------------------------------------
    def _detection_callback(self, msg: Detection3DArray):
        """Filter detections for humans, sort by distance, publish markers."""
        humans = []
        for det in msg.detections:
            # Extract class_id (string), check if it's a human/pedestrian
            class_id = det.class_id.lower() if det.class_id else ""
            if class_id not in ("pedestrian", "human"):
                continue

            # Filter by score if available
            if hasattr(det, 'score') and det.score > 0 and det.score < self.min_score:
                continue

            # Get bbox center position
            pos = det.bbox.center.position
            distance = sqrt(pos.x**2 + pos.y**2 + pos.z**2)
            humans.append((distance, det))

        # Sort by distance ascending (closest first)
        humans.sort(key=lambda x: x[0])

        with self.lock:
            self.current_humans = humans

        # Publish marker visualization
        self._publish_human_markers(msg.header, humans)

        # Show menu if cooldown elapsed and we have humans
        current_time = time.time()
        if current_time - self.last_prompt_time >= self.cooldown_sec and len(humans) > 0:
            with self.lock:
                self.last_prompt_time = current_time

            # Print menu to stdout (non-blocking via input thread)
            self._print_menu(humans)

    # ------------------------------------------------------------------
    def _print_menu(self, humans):
        """Print the CLI menu."""
        print("\n=== Humans Detected (sorted by distance) ===")
        for i, (dist, det) in enumerate(humans, start=1):
            pos = det.bbox.center.position
            print(f"[{i}]  {dist:.2f} m  @ (x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f})")
        print("[0]  Refresh / skip")

    # ------------------------------------------------------------------
    def _input_loop(self):
        """Non-blocking input loop in daemon thread."""
        while rclpy.ok():
            try:
                # Prompt only if we have current_humans
                with self.lock:
                    humans = self.current_humans
                    if not humans:
                        time.sleep(0.1)
                        continue

                try:
                    choice_str = input("Select human (0 to skip): ")
                except EOFError:
                    # Non-interactive terminal; just skip prompt
                    time.sleep(0.5)
                    continue

                try:
                    choice = int(choice_str)
                except ValueError:
                    print("Invalid input. Try again.")
                    continue

                with self.lock:
                    if choice == 0:
                        self.current_humans = []
                    elif 1 <= choice <= len(humans):
                        dist, det = humans[choice - 1]
                        self._publish_selected(det, choice - 1)  # 0-based index
                        # Clear humans to avoid re-prompting
                        self.current_humans = []
                    else:
                        print(f"Invalid selection. Choose 0-{len(humans)}.")
            except Exception as e:
                self.get_logger().debug(f"Input loop error: {e}")
                time.sleep(0.1)

    # ------------------------------------------------------------------
    def _publish_selected(self, det: 'Detection3D', idx_0based: int):
        """Publish the selected human's pose and index."""
        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header = det.header
        pose_msg.pose.position = det.bbox.center.position
        pose_msg.pose.orientation = det.bbox.center.orientation
        self.pub_pose.publish(pose_msg)

        # Publish index
        idx_msg = Int32(data=idx_0based)
        self.pub_index.publish(idx_msg)

        print(f"Selected human {idx_0based + 1} at distance {self._distance(det.bbox.center.position):.2f} m")

    # ------------------------------------------------------------------
    def _publish_human_markers(self, header, humans):
        """Publish marker visualization (red spheres)."""
        markers = MarkerArray()
        for i, (dist, det) in enumerate(humans):
            marker = Marker()
            marker.header = header
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = det.bbox.center.position
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.r = 1.0
            marker.color.a = 0.7
            marker.lifetime = Duration(sec=1, nanosec=0)
            markers.markers.append(marker)

        self.pub_markers.publish(markers)

    @staticmethod
    def _distance(pos):
        """Compute Euclidean distance from origin."""
        return sqrt(pos.x**2 + pos.y**2 + pos.z**2)


def main(args=None):
    rclpy.init(args=args)
    node = HumanSelectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
