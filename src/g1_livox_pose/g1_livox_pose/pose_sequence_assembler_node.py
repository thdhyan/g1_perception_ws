"""Assembles per-person pose streams into time-continuous skeleton sequences.

Subscribes:
    <input_topic> (PersonPose3DArray) - single-instant poses, any frame.

Publishes:
    <output_topic> (SkeletonSequenceArray, latched) - one SkeletonSequence
        per tracked person; every pose re-transformed into `sequence_frame`
        (default `odom`) AT ITS OWN STAMP so successive frames live in one
        stable coordinate system (no robot ego-motion baked into motion).
    <marker_topic> (MarkerArray, latched) - skeletons + root trajectories
        in `sequence_frame`.

Optionally appends one JSON line per accepted pose frame to `log_path`
(epoch seconds + joints + validity) for future skeleton-token training.
"""

from __future__ import annotations

import colorsys
import json
from collections import deque

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rclpy.duration import Duration as RosDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, QoSProfile
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .common import BONES, NUM_JOINTS, quaternion_to_matrix
from g1_livox_pose_msgs.msg import PersonPose3D, PersonPose3DArray, SkeletonSequence, SkeletonSequenceArray


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class Track:
    __slots__ = ("tid", "buf", "last_pos", "last_t")

    def __init__(self, tid: int, pos: np.ndarray, last_wall_t: float, max_frames: int):
        self.tid = tid
        self.buf = deque(maxlen=max_frames)
        self.last_pos = pos
        self.last_t = last_wall_t


class PoseSequenceAssemblerNode(Node):
    def __init__(self):
        super().__init__("g1_pose_sequence_assembler")

        self.declare_parameter("input_topic", "/g1/human_poses")
        self.declare_parameter("output_topic", "/g1/human_pose_sequences")
        self.declare_parameter("marker_topic", "/g1/skeleton_markers")
        self.declare_parameter("sequence_frame", "odom")
        self.declare_parameter("max_frames", 150)
        self.declare_parameter("gate_speed_mps", 1.5)
        self.declare_parameter("max_gate_m", 3.0)
        self.declare_parameter("track_timeout_sec", 0.8)
        self.declare_parameter("min_pose_score", 0.0)
        self.declare_parameter("log_path", "")

        p = self.get_parameter
        self.input_topic = p("input_topic").value
        self.output_topic = p("output_topic").value
        self.marker_topic = p("marker_topic").value
        self.sequence_frame = p("sequence_frame").value
        self.max_frames = int(p("max_frames").value)
        self.gate_speed = float(p("gate_speed_mps").value)
        self.max_gate = float(p("max_gate_m").value)
        self.track_timeout = float(p("track_timeout_sec").value)
        self.min_score = float(p("min_pose_score").value)
        log_path = p("log_path").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tracks: dict[int, Track] = {}
        self.next_id = 1

        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub_poses = self.create_subscription(
            PersonPose3DArray, self.input_topic, self.on_poses, 10
        )
        self.pub_seq = self.create_publisher(SkeletonSequenceArray, self.output_topic, latched_qos)
        self.pub_markers = self.create_publisher(MarkerArray, self.marker_topic, latched_qos)

        self._log_file = None
        if log_path:
            try:
                self._log_file = open(log_path, "a")
                self.get_logger().info(f"Logging poses to {log_path}")
            except OSError as ex:
                self.get_logger().error(f"Cannot open log {log_path}: {ex}")

        self.get_logger().info(
            f"PoseSequenceAssembler ready: '{self.input_topic}' -> '{self.output_topic}' "
            f"(sequence_frame='{self.sequence_frame}', max_frames={self.max_frames})"
        )

    def on_poses(self, arr: PersonPose3DArray) -> None:
        wall_now = self.get_clock().now().nanoseconds * 1e-9

        for person in arr.people:
            if person.pose_score < self.min_score:
                continue
            if len(person.joints) != NUM_JOINTS * 3 or len(person.valid) != NUM_JOINTS:
                self.get_logger().warning(
                    f"Malformed pose ({len(person.joints)} floats, "
                    f"{len(person.valid)} valid flags); skipping.",
                    throttle_duration_sec=5.0,
                )
                continue
            kp = np.array(person.joints, dtype=np.float64).reshape(NUM_JOINTS, 3)
            kp_seq, ok = self._to_sequence_frame(kp, person.header.frame_id, person.header.stamp)
            if not ok:
                continue

            t_sec = stamp_to_sec(person.header.stamp)
            pos = kp_seq.mean(axis=0)

            tid = person.track_id
            if tid == 0 or tid not in self.tracks:
                tid = self._associate(pos, t_sec)
                if tid == 0:
                    tid = self.next_id
                    self.next_id += 1
                    self.tracks[tid] = Track(tid, pos, wall_now, self.max_frames)

            track = self.tracks[tid]
            track.buf.append((t_sec, kp_seq, [int(v) for v in person.valid]))
            track.last_pos = pos
            track.last_t = wall_now

            if self._log_file is not None:
                rec = {
                    "t": t_sec,
                    "track_id": tid,
                    "frame": self.sequence_frame,
                    "joints": [round(float(v), 4) for v in kp_seq.flatten()],
                    "valid": [int(v) for v in person.valid],
                    "score": float(person.pose_score),
                }
                self._log_file.write(json.dumps(rec) + "\n")
                self._log_file.flush()

        self._expire_tracks(wall_now)
        self._publish()

    def _associate(self, pos: np.ndarray, t_sec: float) -> int:
        """Nearest active track within a speed-aware gate; returns 0 if none."""
        best_tid, best_d = 0, float("inf")
        for tid, tr in self.tracks.items():
            if not tr.buf:
                continue
            dt = max(t_sec - tr.buf[-1][0], 0.05)
            gate = min(self.gate_speed * dt, self.max_gate)
            d = float(np.linalg.norm(pos - tr.last_pos))
            if d <= gate and d < best_d:
                best_tid, best_d = tid, d
        return best_tid

    def _expire_tracks(self, wall_now: float) -> None:
        stale = [
            tid for tid, tr in self.tracks.items()
            if wall_now - tr.last_t > self.track_timeout
        ]
        for tid in stale:
            del self.tracks[tid]

    def _to_sequence_frame(self, kp: np.ndarray, src_frame: str, stamp):
        if src_frame == self.sequence_frame:
            return kp, True
        try:
            tf = self.tf_buffer.lookup_transform(
                self.sequence_frame, src_frame, stamp, timeout=RosDuration(seconds=0.3)
            )
        except TransformException as ex:
            self.get_logger().warning(
                f"TF {src_frame}->{self.sequence_frame} @ stamp failed ({ex}); "
                f"pose dropped to keep sequence consistent.",
                throttle_duration_sec=5.0,
            )
            return kp, False
        t = tf.transform.translation
        q = tf.transform.rotation
        rot = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z])
        return (rot @ kp.T).T + trans, True

    def _publish(self) -> None:
        out = SkeletonSequenceArray()
        out.header.frame_id = self.sequence_frame
        clock_now = self.get_clock().now().to_msg()
        out.header.stamp = clock_now

        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for tid, tr in sorted(self.tracks.items()):
            frames = list(tr.buf)
            if not frames:
                continue
            seq = SkeletonSequence()
            seq.header.frame_id = self.sequence_frame
            seq.header.stamp = clock_now
            seq.track_id = tid
            window_sec = max(frames[-1][0] - frames[0][0], 0.0)
            seq.window = Duration(
                sec=int(window_sec), nanosec=int(round((window_sec % 1.0) * 1e9))
            )

            seq_frames = []
            for t_sec, kp, valid in frames:
                person = PersonPose3D()
                person.header.stamp.sec = int(t_sec)
                person.header.stamp.nanosec = int(round((t_sec % 1.0) * 1e9))
                person.header.frame_id = self.sequence_frame
                person.track_id = tid
                person.joints = kp.astype(np.float32).flatten().tolist()
                person.valid = list(valid)
                person.pose_score = 1.0
                seq_frames.append(person)
            seq.frames = seq_frames
            out.sequences.append(seq)

            color = self._color_for(tid)
            self._append_track_markers(markers, tid, frames, color)

        self.pub_seq.publish(out)
        self.pub_markers.publish(markers)

    @staticmethod
    def _color_for(tid: int):
        h = (tid * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        return r, g, b

    def _append_track_markers(self, markers: MarkerArray, tid: int, frames, color) -> None:
        t_latest, kp, valid = frames[-1]

        def hdr(m, ns, mid):
            m.header.frame_id = self.sequence_frame
            m.header.stamp.sec = int(t_latest)
            m.header.stamp.nanosec = int(round((t_latest % 1.0) * 1e9))
            m.ns = ns
            m.id = mid
            m.color.r, m.color.g, m.color.b, m.color.a = *color, 1.0

        bone_pts = []
        bone_cols = []
        for i, j in BONES:
            if valid[i] and valid[j]:
                for idx in (i, j):
                    pt = Point()
                    pt.x, pt.y, pt.z = (float(v) for v in kp[idx])
                    bone_pts.append(pt)
                    cl = ColorRGBA()
                    cl.r, cl.g, cl.b, cl.a = *color, 1.0
                    bone_cols.append(cl)
        if bone_pts:
            bones = Marker()
            bones.type = Marker.LINE_LIST
            hdr(bones, f"person_{tid}", tid * 10)
            bones.scale.x = 0.03
            bones.points = bone_pts
            bones.colors = bone_cols
            markers.markers.append(bones)

        joint_pts = []
        joint_cols = []
        for ji in range(NUM_JOINTS):
            if valid[ji]:
                pt = Point()
                pt.x, pt.y, pt.z = (float(v) for v in kp[ji])
                joint_pts.append(pt)
                cl = ColorRGBA()
                cl.r, cl.g, cl.b, cl.a = *color, 1.0
                joint_cols.append(cl)
        if joint_pts:
            joints_m = Marker()
            joints_m.type = Marker.SPHERE_LIST
            hdr(joints_m, f"person_{tid}", tid * 10 + 1)
            joints_m.scale.x = joints_m.scale.y = joints_m.scale.z = 0.06
            joints_m.points = joint_pts
            joints_m.colors = joint_cols
            markers.markers.append(joints_m)

        if len(frames) >= 2:
            trail_pts = []
            for _, kpf, _v in frames:
                root = (kpf[4] + kpf[10]) / 2.0
                pt = Point()
                pt.x, pt.y, pt.z = (float(v) for v in root)
                trail_pts.append(pt)
            if len(trail_pts) >= 2:
                trail = Marker()
                trail.type = Marker.LINE_STRIP
                hdr(trail, f"person_{tid}_trail", tid * 10 + 2)
                trail.scale.x = 0.02
                trail.color.a = 0.6
                trail.points = trail_pts
                markers.markers.append(trail)

        label = Marker()
        label.type = Marker.TEXT_VIEW_FACING
        hdr(label, f"person_{tid}_label", tid * 10 + 3)
        label.scale.z = 0.15
        head_z = float(kp[13][2]) if valid[13] else float(kp[:, 2].max())
        label.pose.position.x = float(kp[:, 0].mean())
        label.pose.position.y = float(kp[:, 1].mean())
        label.pose.position.z = head_z + 0.25
        label.text = f"#{tid}"
        markers.markers.append(label)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseSequenceAssemblerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(node, "_log_file", None) is not None:
            node._log_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
