#!/usr/bin/env python3
"""2-Pass Snapshot Point Cloud Collection, 3D Human Detection, and Selection Node.

Solves LiDAR frame flicker and inference desync by decoupling into 2 passes:
  - Pass 1 (Collection): Gathers N frames / T seconds of Livox LiDAR data into a dense point cloud.
  - Pass 2 (Detection & Selection): Runs 3D detection ONCE on the dense accumulated cloud,
    publishes latched `/livox/collected_points` and distance-sorted human markers for RViz,
    and provides a non-flickering CLI menu for the operator to select the target human.

Publishes:
    - `/livox/collected_points` (sensor_msgs/msg/PointCloud2, Transient Local)
    - `/g1/sorted_humans` (vision_msgs/msg/Detection3DArray, Transient Local)
    - `/g1/detection_markers/livox` (visualization_msgs/msg/MarkerArray)
    - `/g1/sorted_human_markers` (visualization_msgs/msg/MarkerArray)
    - `/g1/selected_human` (geometry_msgs/msg/PoseStamped)
    - `/g1/selected_human_index` (std_msgs/msg/Int32)
    - `/g1/selected_human_marker` (visualization_msgs/msg/MarkerArray)

Services:
    - `/g1/trigger_snapshot` (std_srvs/srv/Trigger)
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Int32
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

from .centerpoint_model import CLASS_COLORS, CLASS_NAMES, CenterPointBackend
from .pointpillar_model import PointPillarBackend
from .voxelnext_model import VoxelNeXtBackend


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Yaw rotation as (x, y, z, w)."""
    return (0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0)))


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix."""
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


class LivoxSnapshotPipelineNode(Node):
    def __init__(self):
        super().__init__("g1_livox_snapshot_pipeline")

        # Parameters
        self.declare_parameter("algorithm", "centerpoint")
        self.declare_parameter(
            "checkpoint_path",
            "/home/thakk100/Projects/Thesis/livox_detection/pt/livox_model_1.pt",
        )
        self.declare_parameter("input_topic", "/livox/lidar")
        self.declare_parameter("target_frame", "pelvis")
        self.declare_parameter("frame_override", "")
        self.declare_parameter("score_threshold", 0.10)
        self.declare_parameter("collect_frames", 10)
        self.declare_parameter("collect_duration_sec", 2.0)
        self.declare_parameter("max_distance", 25.0)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("auto_start", False)
        self.declare_parameter("front_max_range", 15.0)
        self.declare_parameter("front_min_x", 0.0)
        self.declare_parameter("offset_ground", 1.33)
        # VoxelNeXt-specific parameters
        _ws_root_str = str(Path(__file__).resolve().parents[3])
        self.declare_parameter(
            "voxelnext_cfg",
            str(Path(_ws_root_str) / "VoxelNeXt" / "tools" / "cfgs" / "nuscenes_models" / "cbgs_voxel0075_voxelnext.yaml"),
        )
        self.declare_parameter("voxelnext_dir", str(Path(_ws_root_str) / "VoxelNeXt"))

        self.algorithm = self.get_parameter("algorithm").value.lower()
        self.checkpoint_path = self.get_parameter("checkpoint_path").value
        self.input_topic = self.get_parameter("input_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.frame_override = self.get_parameter("frame_override").value or None
        self.score_threshold = float(self.get_parameter("score_threshold").value)
        self.collect_frames = int(self.get_parameter("collect_frames").value)
        self.collect_duration_sec = float(self.get_parameter("collect_duration_sec").value)
        self.max_distance = float(self.get_parameter("max_distance").value)
        self.device = self.get_parameter("device").value
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.front_max_range = float(self.get_parameter("front_max_range").value)
        self.front_min_x = float(self.get_parameter("front_min_x").value)
        self.offset_ground = float(self.get_parameter("offset_ground").value)
        self.voxelnext_cfg = self.get_parameter("voxelnext_cfg").value
        self.voxelnext_dir = self.get_parameter("voxelnext_dir").value

        # State management
        self.lock = threading.Lock()
        self.state = "IDLE"  # "IDLE", "COLLECTING", "INFERRING", "READY"
        self.collected_frames: List[np.ndarray] = []
        self.collection_start_time = 0.0
        self.last_header = None
        self.sorted_humans: List[Tuple[float, Detection3D]] = []
        self.selected_target: Optional[Detection3D] = None
        self.selected_rank: Optional[int] = None

        # TF2 listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Backend Model Init
        self.backend = None
        self._init_backend()

        # QoS Profiles
        # Latched QoS (Transient Local) for frozen snapshot visualization
        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        sensor_sub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        live_pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publishers
        self.pub_collected_points = self.create_publisher(
            PointCloud2, "/livox/collected_points", latched_qos
        )
        self.pub_snapshot_front_cloud = self.create_publisher(
            PointCloud2, "/livox/collected_points_front_15m", latched_qos
        )
        self.pub_live_front_cloud = self.create_publisher(
            PointCloud2, "/livox/live_front_15m", live_pub_qos
        )
        self.pub_sorted_humans = self.create_publisher(
            Detection3DArray, "/g1/sorted_humans", latched_qos
        )
        self.pub_detection_markers = self.create_publisher(
            MarkerArray, "/g1/detection_markers/livox", latched_qos
        )
        self.pub_sorted_markers = self.create_publisher(
            MarkerArray, "/g1/sorted_human_markers", latched_qos
        )
        self.pub_selected_pose = self.create_publisher(
            PoseStamped, "/g1/selected_human", 10
        )
        self.pub_selected_index = self.create_publisher(
            Int32, "/g1/selected_human_index", 10
        )
        self.pub_selected_marker = self.create_publisher(
            MarkerArray, "/g1/selected_human_marker", 10
        )

        # Subscribers
        self.sub_cloud = self.create_subscription(
            PointCloud2, self.input_topic, self.on_pointcloud2, sensor_sub_qos
        )
        self.sub_select_id = self.create_subscription(
            Int32, "/g1/select_human_id", self.on_select_human_id, 10
        )
        try:
            from livox_ros_driver2.msg import CustomMsg
            self.sub_custom = self.create_subscription(
                CustomMsg, self.input_topic, self.on_custom_msg, sensor_sub_qos
            )
        except ImportError:
            self.sub_custom = None

        # Services for programmatic trigger & retrigger
        self.srv_trigger = self.create_service(
            Trigger, "/g1/trigger_snapshot", self.handle_trigger_srv
        )
        self.srv_retrigger = self.create_service(
            Trigger, "/g1/retrigger_snapshot", self.handle_trigger_srv
        )
        self.srv_rescan = self.create_service(
            Trigger, "/g1/rescan", self.handle_trigger_srv
        )

        self.declare_parameter("enable_cli_input", False)
        self.enable_cli_input = bool(self.get_parameter("enable_cli_input").value)

        # Start CLI Input loop only if explicitly requested
        if self.enable_cli_input:
            self.cli_thread = threading.Thread(target=self._cli_input_loop, daemon=True)
            self.cli_thread.start()

        self.get_logger().info(
            f"LivoxSnapshotPipelineNode active.\n"
            f"  - Target Frame: '{self.target_frame}'\n"
            f"  - Input Topic: '{self.input_topic}'\n"
            f"  - Collection Target: {self.collect_frames} frames / {self.collect_duration_sec:.1f}s\n"
            f"  - Model: {self.algorithm} (threshold={self.score_threshold})\n"
            f"  - Output PointCloud: '/livox/collected_points' (Latched)\n"
        )

        if self.auto_start:
            self.trigger_collection()
        else:
            self._display_idle_prompt()

    def _display_idle_prompt(self) -> None:
        """Display prompt when waiting for user trigger."""
        print(f"\n=======================================================")
        print(f"  [G1 LiDAR Live Stream Active]")
        print(f"  Streaming topic: '{self.input_topic}' -> RViz")
        print(f"  Press [ENTER], type 's', or call /g1/trigger_snapshot to take a snapshot")
        print(f"=======================================================")
        print("Press [ENTER] to take snapshot: ", end="", flush=True)

    def _init_backend(self) -> None:
        """Initialize CenterPoint, PointPillars, or VoxelNeXt backend."""
        try:
            if self.algorithm == "pointpillar":
                self.backend = PointPillarBackend(
                    checkpoint="",
                    device=self.device,
                    score_threshold=self.score_threshold,
                )
            elif self.algorithm == "voxelnext":
                self.backend = VoxelNeXtBackend(
                    checkpoint=self.checkpoint_path,
                    device=self.device,
                    score_threshold=self.score_threshold,
                    offset_ground=self.offset_ground,
                    cfg_file=self.voxelnext_cfg,
                    voxelnext_dir=self.voxelnext_dir,
                )
            else:
                self.backend = CenterPointBackend(
                    checkpoint=self.checkpoint_path,
                    device=self.device,
                    score_threshold=self.score_threshold,
                    offset_ground=self.offset_ground,
                )
            self.backend.load()
            self.get_logger().info(f"Loaded backend: {self.algorithm}")
        except Exception as e:
            self.get_logger().warning(f"Backend init warning: {e}. Using PointPillars clustering fallback.")
            self.backend = PointPillarBackend(
                checkpoint="",
                device=self.device,
                score_threshold=self.score_threshold,
            )
            self.backend.load()

    def trigger_collection(self) -> None:
        """Trigger a new point cloud collection cycle."""
        with self.lock:
            self.state = "COLLECTING"
            self.collected_frames.clear()
            self.collection_start_time = time.time()
            self.sorted_humans.clear()

        print("\n=======================================================")
        print(f" [●] PASS 1: COLLECTING POINT CLOUDS FROM '{self.input_topic}'...")
        print(f"     Target: {self.collect_frames} frames (or {self.collect_duration_sec:.1f}s)")
        print("=======================================================")

    def handle_trigger_srv(self, request, response):
        """ROS 2 Service callback to trigger a snapshot."""
        self.trigger_collection()
        response.success = True
        response.message = f"Started point cloud collection for {self.collect_frames} frames / {self.collect_duration_sec:.1f}s."
        return response

    def _filter_front_15m(self, points: np.ndarray) -> np.ndarray:
        """Filter points to front hemisphere (X >= front_min_x) and within range (dist <= front_max_range)."""
        if points.shape[0] == 0:
            return points
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        dist = np.sqrt(x**2 + y**2 + z**2)
        mask = (x >= self.front_min_x) & (dist <= self.front_max_range)
        return points[mask]

    def on_pointcloud2(self, msg: PointCloud2) -> None:
        """Process incoming PointCloud2: publish live front 15m subset, and collect if active."""
        points = self.cloud_to_array(msg)
        if points.shape[0] == 0:
            return

        frame_id = msg.header.frame_id
        if frame_id == "livox_frame" or not frame_id:
            frame_id = "mid360_link"
            msg.header.frame_id = frame_id

        # Check if currently collecting snapshot
        with self.lock:
            if self.state != "COLLECTING":
                return
            self.collected_frames.append(points)
            self.last_header = msg.header
            num_frames = len(self.collected_frames)
            elapsed = time.time() - self.collection_start_time

        if num_frames % 2 == 0:
            print(f"  -> Captured frame #{num_frames}/{self.collect_frames} ({points.shape[0]:,d} points, {elapsed:.1f}s elapsed)")

        if num_frames >= self.collect_frames or elapsed >= self.collect_duration_sec:
            threading.Thread(target=self._run_pass2_inference, daemon=True).start()

    def on_custom_msg(self, msg) -> None:
        """Process incoming CustomMsg: collect only if active snapshot."""
        points = self.custom_msg_to_array(msg)
        if points.shape[0] == 0:
            return

        frame_id = msg.header.frame_id
        if frame_id == "livox_frame" or not frame_id:
            frame_id = "mid360_link"
            msg.header.frame_id = frame_id

        # Check if currently collecting snapshot
        with self.lock:
            if self.state != "COLLECTING":
                return
            self.collected_frames.append(points)
            self.last_header = msg.header
            num_frames = len(self.collected_frames)
            elapsed = time.time() - self.collection_start_time

        if num_frames % 2 == 0:
            print(f"  -> Captured frame #{num_frames}/{self.collect_frames} ({points.shape[0]:,d} points, {elapsed:.1f}s elapsed)")

        if num_frames >= self.collect_frames or elapsed >= self.collect_duration_sec:
            threading.Thread(target=self._run_pass2_inference, daemon=True).start()

    def _run_pass2_inference(self) -> None:
        """Pass 2: Merge collected points, run detection ONCE, sort, and publish latched topics."""
        with self.lock:
            if self.state != "COLLECTING":
                return
            self.state = "INFERRING"
            frames = list(self.collected_frames)
            header = self.last_header

        if not frames or header is None:
            self.get_logger().warning("No frames collected to run detection.")
            with self.lock:
                self.state = "IDLE"
            return

        total_pts = sum(f.shape[0] for f in frames)
        merged_points = np.vstack(frames)
        source_frame = self.frame_override or header.frame_id

        print("\n=======================================================")
        print(f" [●] PASS 2: RUNNING 3D DETECTION ON DENSE POINT CLOUD")
        print(f"     Total Accumulated Points: {total_pts:,d} across {len(frames)} frames")
        print(f"     Model Backend: {self.algorithm.upper()} | Target Frame: '{self.target_frame}'")
        print("=======================================================")

        # 1. Publish dense accumulated cloud on /livox/collected_points (Latched)
        cloud_msg = self.array_to_cloud(header, merged_points)
        self.pub_collected_points.publish(cloud_msg)
        print(" [✓] Published dense cloud to topic '/livox/collected_points' (Latched)")

        # 2. Publish front-facing <= 15m dense snapshot cloud on /livox/collected_points_front_15m (Latched)
        snapshot_front_pts = self._filter_front_15m(merged_points)
        if snapshot_front_pts.shape[0] > 0:
            snapshot_front_msg = self.array_to_cloud(header, snapshot_front_pts)
            self.pub_snapshot_front_cloud.publish(snapshot_front_msg)
            print(f" [✓] Published front 15m subset ({snapshot_front_pts.shape[0]:,d} pts) to '/livox/collected_points_front_15m' (Latched)")

        # 2. Run single 3D detection pass
        t0 = time.time()
        boxes, scores, labels = self.backend.infer(merged_points)
        infer_time = (time.time() - t0) * 1000.0

        if boxes.shape[0] > 0 and self.max_distance > 0:
            dist2d = np.hypot(boxes[:, 0], boxes[:, 1])
            keep = dist2d <= self.max_distance
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # 3. Transform detected 3D boxes to pelvis frame
        transformed_boxes, output_frame = self._transform_boxes_to_target(boxes, source_frame, header.stamp)

        # 4. Filter for human/pedestrian classes and calculate distance
        det_array = self.to_detection_array(transformed_boxes, scores, labels, header.stamp, output_frame)
        
        humans_with_dist: List[Tuple[float, Detection3D]] = []
        for det in det_array.detections:
            is_human = False
            for hyp in det.results:
                cls_id = hyp.hypothesis.class_id.lower()
                if cls_id in ("pedestrian", "human") and hyp.hypothesis.score >= self.score_threshold:
                    is_human = True
                    break
            if is_human:
                pos = det.bbox.center.position
                dist = math.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
                humans_with_dist.append((dist, det))

        # Sort humans ascending by distance from pelvis (closest first = #1)
        humans_with_dist.sort(key=lambda x: x[0])

        # 5. Publish latched sorted humans and visual markers
        sorted_array = Detection3DArray()
        sorted_array.header.stamp = header.stamp
        sorted_array.header.frame_id = output_frame
        for _, det in humans_with_dist:
            sorted_array.detections.append(det)

        self.pub_sorted_humans.publish(sorted_array)

        # Publish 3D bounding box cubes
        box_markers = self.to_markers(transformed_boxes, scores, labels, header.stamp, output_frame)
        self.pub_detection_markers.publish(box_markers)

        # Publish ranked badges (#1, #2, etc.) and ground circles
        sorted_markers = self._create_sorted_markers(sorted_array.header, humans_with_dist)
        self.pub_sorted_markers.publish(sorted_markers)

        with self.lock:
            self.sorted_humans = humans_with_dist
            self.state = "READY"

        print(f" [✓] Detection finished in {infer_time:.1f}ms. Found {len(humans_with_dist)} human(s).")
        self._display_menu(humans_with_dist, output_frame)

    def _display_menu(self, humans: List[Tuple[float, Detection3D]], frame_id: str) -> None:
        """Display stable, static CLI menu."""
        print(f"\n=======================================================")
        print(f"  [G1 Snapshot Detection Menu] - Frame: '{frame_id}'")
        print(f"=======================================================")
        if not humans:
            print("  [!] No humans detected in this snapshot.")
        else:
            for rank, (dist, det) in enumerate(humans, start=1):
                pos = det.bbox.center.position
                score = det.results[0].hypothesis.score if det.results else 0.0
                tag = " (Closest)" if rank == 1 else ""
                print(f"  [{rank}]  Distance: {dist:.2f} m{tag} | Position: (x={pos.x:+.2f}, y={pos.y:+.2f}, z={pos.z:+.2f}) | Conf: {score:.2f}")
        print(f"-------------------------------------------------------")
        print(f"  [R]  Re-trigger: Collect a new {self.collect_duration_sec:.1f}s snapshot")
        print(f"  [0]  Clear current selection")
        print(f"=======================================================")
        prompt_range = f"[1-{len(humans)}]" if humans else "[R]"
        print(f"Enter human number to approach {prompt_range} or 'R' to re-collect: ", end="", flush=True)

    def _cli_input_loop(self) -> None:
        """Handle CLI user keyboard input."""
        while rclpy.ok():
            try:
                line = sys.stdin.readline()
                if line is None:
                    time.sleep(0.2)
                    continue

                cmd = line.strip()

                # If user presses Enter (empty line) or types 's' / 'snap' / 'r' / 'rescan'
                if not cmd or cmd.upper() in ("S", "SNAP", "SNAPSHOT", "R", "RELOAD", "RETRY", "REFRESH", "SCAN"):
                    with self.lock:
                        curr_state = self.state
                    if curr_state != "COLLECTING":
                        self.trigger_collection()
                    continue

                if cmd == "0":
                    print("\n[*] Selection cleared.")
                    with self.lock:
                        self.selected_target = None
                        self.selected_rank = None
                    self._clear_highlight()
                    self._display_idle_prompt()
                    continue

                try:
                    choice = int(cmd)
                except ValueError:
                    print(f"\n[!] Invalid input '{cmd}'. Press [ENTER] to take snapshot, or enter human #.")
                    continue

                with self.lock:
                    num_humans = len(self.sorted_humans)
                    if 1 <= choice <= num_humans:
                        dist, selected_det = self.sorted_humans[choice - 1]
                        self.selected_target = selected_det
                        self.selected_rank = choice
                        header = self.last_header
                        self._publish_selection(selected_det, choice, dist, header)
                    else:
                        print(f"\n[!] Selection {choice} out of range [1-{num_humans}].")

            except Exception as e:
                self.get_logger().debug(f"CLI loop error: {e}")
                time.sleep(0.2)

    def on_select_human_id(self, msg: Int32) -> None:
        """Handle human selection from ROS topic /g1/select_human_id."""
        choice = int(msg.data)
        with self.lock:
            num_humans = len(self.sorted_humans)
            if choice == 0:
                self.selected_target = None
                self.selected_rank = None
                self._clear_highlight()
                self.get_logger().info("Selection cleared via /g1/select_human_id.")
                return

            if 1 <= choice <= num_humans:
                dist, selected_det = self.sorted_humans[choice - 1]
                self.selected_target = selected_det
                self.selected_rank = choice
                header = self.last_header
                self._publish_selection(selected_det, choice, dist, header)
                self.get_logger().info(
                    f"[✓] Selected Human #{choice} (dist: {dist:.2f}m) via /g1/select_human_id topic."
                )
            else:
                self.get_logger().warning(
                    f"Selection #{choice} out of range [1-{num_humans}] on /g1/select_human_id."
                )

    def _publish_selection(self, det: Detection3D, rank: int, dist: float, header) -> None:
        """Publish selected target pose, index, and beacon marker."""
        pos = det.bbox.center.position

        # 1. PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header = det.header
        pose_msg.pose.position = pos
        pose_msg.pose.orientation = det.bbox.center.orientation
        self.pub_selected_pose.publish(pose_msg)

        # 2. Index
        self.pub_selected_index.publish(Int32(data=rank))

        # 3. Beacon Cylinder in RViz
        markers = MarkerArray()
        beam = Marker()
        beam.header = det.header
        beam.ns = "selected_human_beacon"
        beam.id = 999
        beam.type = Marker.CYLINDER
        beam.action = Marker.ADD
        beam.pose.position.x = pos.x
        beam.pose.position.y = pos.y
        beam.pose.position.z = pos.z + 1.2
        beam.pose.orientation.w = 1.0
        beam.scale.x = 0.15
        beam.scale.y = 0.15
        beam.scale.z = 2.0
        beam.color.r = 0.0
        beam.color.g = 1.0
        beam.color.b = 0.0
        beam.color.a = 0.9
        markers.markers.append(beam)
        self.pub_selected_marker.publish(markers)

        print(f"\n=======================================================")
        print(f" [✓] TARGET LOCKED: Human #{rank} at {dist:.2f} m")
        print(f"     Coordinates (pelvis): x={pos.x:+.2f} m, y={pos.y:+.2f} m, z={pos.z:+.2f} m")
        print(f"     Published to '/g1/selected_human'. Locomotion approach dispatched.")
        print(f"=======================================================\n")
        print("Enter another human number or 'R' to collect new snapshot: ", end="", flush=True)

    def _clear_highlight(self) -> None:
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        self.pub_selected_marker.publish(markers)

    def _transform_boxes_to_target(
        self, boxes: np.ndarray, source_frame: str, stamp
    ) -> Tuple[np.ndarray, str]:
        """Transform (M, 7) bounding boxes to self.target_frame using TF2."""
        if boxes.shape[0] == 0:
            return boxes, (self.target_frame if self.target_frame else source_frame)

        if not self.target_frame or self.target_frame == source_frame:
            return boxes, source_frame

        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
            t = tf.transform.translation
            q = tf.transform.rotation

            trans = np.array([t.x, t.y, t.z], dtype=np.float64)
            rot_mat = quaternion_to_matrix(q.x, q.y, q.z, q.w)
            delta_yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])

            transformed = np.zeros_like(boxes)
            for i in range(boxes.shape[0]):
                center_src = boxes[i, :3].astype(np.float64)
                center_target = rot_mat @ center_src + trans
                transformed[i, :3] = center_target
                transformed[i, 3:6] = boxes[i, 3:6]
                transformed[i, 6] = boxes[i, 6] + delta_yaw

            return transformed, self.target_frame
        except TransformException as ex:
            self.get_logger().warning(
                f"TF transform {source_frame} -> {self.target_frame} failed ({ex}); using {source_frame}"
            )
            return boxes, source_frame

    def _create_sorted_markers(
        self, header, humans_with_dist: List[Tuple[float, Detection3D]]
    ) -> MarkerArray:
        """Create numbered visual rank tags and ground rings."""
        markers = MarkerArray()

        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for rank, (dist, det) in enumerate(humans_with_dist, start=1):
            pos = det.bbox.center.position
            size = det.bbox.size

            # Floating text badge above human head
            tag = Marker()
            tag.header = header
            tag.ns = "human_rank_tags"
            tag.id = rank
            tag.type = Marker.TEXT_VIEW_FACING
            tag.action = Marker.ADD
            tag.pose.position.x = pos.x
            tag.pose.position.y = pos.y
            tag.pose.position.z = pos.z + (size.z / 2.0 if size.z > 0 else 0.8) + 0.35
            tag.pose.orientation.w = 1.0
            tag.scale.z = 0.35
            tag.text = f"#{rank} [{dist:.2f} m]"
            if rank == 1:
                tag.color.r, tag.color.g, tag.color.b, tag.color.a = 0.0, 1.0, 0.0, 1.0
            else:
                tag.color.r, tag.color.g, tag.color.b, tag.color.a = 1.0, 0.85, 0.2, 1.0
            markers.markers.append(tag)

            # Ground target cylinder
            ring = Marker()
            ring.header = header
            ring.ns = "human_footprints"
            ring.id = rank + 100
            ring.type = Marker.CYLINDER
            ring.action = Marker.ADD
            ring.pose.position.x = pos.x
            ring.pose.position.y = pos.y
            ring.pose.position.z = pos.z - (size.z / 2.0 if size.z > 0 else 0.8)
            ring.pose.orientation.w = 1.0
            ring.scale.x = 0.6
            ring.scale.y = 0.6
            ring.scale.z = 0.05
            if rank == 1:
                ring.color.r, ring.color.g, ring.color.b, ring.color.a = 0.0, 1.0, 0.2, 0.7
            else:
                ring.color.r, ring.color.g, ring.color.b, ring.color.a = 1.0, 0.8, 0.0, 0.5
            markers.markers.append(ring)

        return markers

    def to_markers(
        self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, stamp, frame: str
    ) -> MarkerArray:
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            cls_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else "unknown"
            r, g, b = CLASS_COLORS.get(cls_name, (1.0, 1.0, 1.0))
            qx, qy, qz, qw = yaw_to_quaternion(float(box[6]))

            box_marker = Marker()
            box_marker.header.frame_id = frame
            box_marker.header.stamp = stamp
            box_marker.ns = "livox_3d_boxes"
            box_marker.id = i * 2
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD
            box_marker.pose.position.x = float(box[0])
            box_marker.pose.position.y = float(box[1])
            box_marker.pose.position.z = float(box[2])
            box_marker.pose.orientation.x = qx
            box_marker.pose.orientation.y = qy
            box_marker.pose.orientation.z = qz
            box_marker.pose.orientation.w = qw
            box_marker.scale.x = float(box[3])
            box_marker.scale.y = float(box[4])
            box_marker.scale.z = float(box[5])
            box_marker.color.r = r
            box_marker.color.g = g
            box_marker.color.b = b
            box_marker.color.a = 0.45
            markers.markers.append(box_marker)

            text_marker = Marker()
            text_marker.header.frame_id = frame
            text_marker.header.stamp = stamp
            text_marker.ns = "livox_labels"
            text_marker.id = i * 2 + 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(box[0])
            text_marker.pose.position.y = float(box[1])
            text_marker.pose.position.z = float(box[2] + box[5] / 2.0 + 0.25)
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.25
            dist = np.linalg.norm(box[:3])
            text_marker.text = f"{cls_name} ({score:.2f}) [{dist:.2f}m in {frame}]"
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.95
            markers.markers.append(text_marker)

        return markers

    def to_detection_array(
        self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, stamp, frame: str
    ) -> Detection3DArray:
        array = Detection3DArray()
        array.header.stamp = stamp
        array.header.frame_id = frame

        for box, score, label in zip(boxes, scores, labels):
            detection = Detection3D()
            detection.header = array.header

            hyp = ObjectHypothesisWithPose()
            cls_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else "unknown"
            hyp.hypothesis.class_id = cls_name
            hyp.hypothesis.score = float(score)
            detection.results.append(hyp)

            bbox = BoundingBox3D()
            bbox.center.position.x = float(box[0])
            bbox.center.position.y = float(box[1])
            bbox.center.position.z = float(box[2])
            qx, qy, qz, qw = yaw_to_quaternion(float(box[6]))
            bbox.center.orientation.x = qx
            bbox.center.orientation.y = qy
            bbox.center.orientation.z = qz
            bbox.center.orientation.w = qw
            bbox.size.x = float(box[3])
            bbox.size.y = float(box[4])
            bbox.size.z = float(box[5])
            detection.bbox = bbox

            array.detections.append(detection)
        return array

    @staticmethod
    def cloud_to_array(msg: PointCloud2) -> np.ndarray:
        available = {f.name for f in msg.fields}
        fields = ["x", "y", "z"] + (["intensity"] if "intensity" in available else [])
        raw = point_cloud2.read_points(msg, field_names=fields, skip_nans=True)
        if raw.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32)
        points = np.stack([raw[name] for name in fields], axis=-1).astype(np.float32)
        if points.shape[1] == 3:
            points = np.hstack([points, np.zeros((points.shape[0], 1), dtype=np.float32)])
        return points

    @staticmethod
    def custom_msg_to_array(msg) -> np.ndarray:
        if msg.point_num == 0:
            return np.zeros((0, 4), dtype=np.float32)
        points = np.zeros((msg.point_num, 4), dtype=np.float32)
        for i, p in enumerate(msg.points):
            points[i, 0] = p.x
            points[i, 1] = p.y
            points[i, 2] = p.z
            points[i, 3] = p.reflectivity / 255.0
        return points

    @staticmethod
    def array_to_cloud(header, points: np.ndarray) -> PointCloud2:
        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = points.shape[0]
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = points.astype(np.float32).tobytes()
        return cloud


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LivoxSnapshotPipelineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
