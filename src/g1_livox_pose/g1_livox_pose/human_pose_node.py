"""3D human pose estimation node for g1_livox_pose."""

from __future__ import annotations

import time
import threading
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray

from .backends import get_backend
from .common import crop_points_in_box, detection_to_box7, quaternion_to_matrix
from g1_livox_pose_msgs.msg import PersonPose3D, PersonPose3DArray


class HumanPoseNode(Node):
    """Runs a pose backend on each human detection against the latest cloud.

    Subscribes:
        <input_cloud_topic> (PointCloud2, latched snapshot cloud)
        <input_detections_topic> (Detection3DArray, e.g. /g1/sorted_humans)

    Publishes:
        <output_topic> (PersonPose3DArray) — one PersonPose3D per human,
        joints transformed into `target_frame` using TF AT THE CLOUD STAMP.
    """

    def __init__(self):
        super().__init__("g1_human_pose")

        self.declare_parameter("backend", "debug")
        self.declare_parameter("input_cloud_topic", "/livox/collected_points")
        self.declare_parameter("input_detections_topic", "/g1/sorted_humans")
        self.declare_parameter("output_topic", "/g1/human_poses")
        self.declare_parameter("target_frame", "pelvis")
        self.declare_parameter("crop_margin", 0.30)
        self.declare_parameter("min_crop_points", 20)
        self.declare_parameter("max_cloud_age_sec", 2.0)

        self.backend_name = self.get_parameter("backend").value
        self.cloud_topic = self.get_parameter("input_cloud_topic").value
        self.det_topic = self.get_parameter("input_detections_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.crop_margin = float(self.get_parameter("crop_margin").value)
        self.min_crop_points = int(self.get_parameter("min_crop_points").value)
        self.max_cloud_age = float(self.get_parameter("max_cloud_age_sec").value)

        self.backend = get_backend(self.backend_name)
        self.backend.load()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._lock = threading.Lock()
        self._cloud: Optional[Tuple[np.ndarray, object, str]] = None

        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub_cloud = self.create_subscription(
            PointCloud2, self.cloud_topic, self.on_cloud, latched_qos
        )
        self.sub_det = self.create_subscription(
            Detection3DArray, self.det_topic, self.on_detections, latched_qos
        )
        self.pub_poses = self.create_publisher(PersonPose3DArray, self.output_topic, 10)

        self.get_logger().info(
            f"HumanPoseNode ready: backend={self.backend_name} "
            f"cloud='{self.cloud_topic}' detections='{self.det_topic}' "
            f"-> '{self.output_topic}' (target_frame='{self.target_frame}')"
        )

    def on_cloud(self, msg: PointCloud2) -> None:
        available = {f.name for f in msg.fields}
        fields = ["x", "y", "z"] + (["intensity"] if "intensity" in available else [])
        raw = point_cloud2.read_points(msg, field_names=fields, skip_nans=True)
        if raw.shape[0] == 0:
            return
        pts = np.stack([raw[name] for name in fields], axis=-1).astype(np.float32)
        if pts.shape[1] == 3:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)])
        with self._lock:
            self._cloud = (pts, msg.header.stamp, msg.header.frame_id)
        self.get_logger().info(
            f"Snapshot cloud stored: {pts.shape[0]} pts @ "
            f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} [{msg.header.frame_id}]",
            throttle_duration_sec=5.0,
        )

    def on_detections(self, arr: Detection3DArray) -> None:
        with self._lock:
            cloud = self._cloud
        if cloud is None or cloud[0].shape[0] == 0:
            self.get_logger().warning("Detections received but no cloud available yet.", throttle_duration_sec=5.0)
            return

        points, cloud_stamp, cloud_frame = cloud
        humans = []
        for det in arr.detections:
            for hyp in det.results:
                if hyp.hypothesis.class_id.lower() in ("pedestrian", "human"):
                    humans.append(det)
                    break

        out = PersonPose3DArray()
        t_start = time.time()

        for det in humans:
            box7 = detection_to_box7(det)
            crop = crop_points_in_box(points, box7, self.crop_margin)
            if crop.shape[0] < self.min_crop_points:
                self.get_logger().warning(
                    f"Crop too small ({crop.shape[0]} pts); skipping person.",
                    throttle_duration_sec=5.0,
                )
                continue
            try:
                kp, valid, score = self.backend.infer(crop, box7)
            except Exception as ex:
                self.get_logger().error(f"Backend '{self.backend_name}' failed: {ex}")
                continue
            kp_t = self._transform_keypoints(kp, cloud_frame, self.target_frame, cloud_stamp)
            person = PersonPose3D()
            person.header.stamp = cloud_stamp
            person.header.frame_id = self.target_frame
            person.track_id = 0
            person.joints = kp_t.astype(np.float32).flatten().tolist()
            person.valid = [int(v) for v in valid]
            person.pose_score = float(score)
            out.people.append(person)

        out.header.stamp = cloud_stamp
        out.header.frame_id = self.target_frame
        if out.people:
            self.pub_poses.publish(out)

        elapsed_ms = (time.time() - t_start) * 1000.0
        self.get_logger().info(
            f"Pose pass done: {len(humans)} det(s), {len(out.people)} pose(s), {elapsed_ms:.1f} ms",
            throttle_duration_sec=2.0,
        )

    def _transform_keypoints(
        self, kp: np.ndarray, source_frame: str, target_frame: str, stamp
    ) -> np.ndarray:
        """Transform (K,3) joints with TF at the capture stamp (not 'now')."""
        if not target_frame or target_frame == source_frame:
            return kp
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp, timeout=Duration(seconds=0.3)
            )
        except TransformException as ex:
            self.get_logger().warning(
                f"TF {source_frame}->{target_frame} @ capture stamp failed ({ex}); "
                f"keeping {source_frame} coords."
            )
            return kp
        t = tf.transform.translation
        q = tf.transform.rotation
        rot = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z])
        return (rot @ kp.T).T + trans


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HumanPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
