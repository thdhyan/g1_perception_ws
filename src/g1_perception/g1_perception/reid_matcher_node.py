#!/usr/bin/env python3
"""
reid_matcher_node.py — Match live ped detections against named enrolled ReID templates.

Subscribes:
    /g1/detections/livox  (vision_msgs/Detection3DArray)
    /livox/lidar          (sensor_msgs/PointCloud2)

Publishes:
    /g1/reid_target      (geometry_msgs/PoseStamped) — best-match ped pose
    /g1/reid_match_name  (std_msgs/String) — name of matched person (e.g. "Alice")
    /g1/reid_markers     (visualization_msgs/MarkerArray) — all peds coloured by similarity
    /g1/reid_scores      (std_msgs/Float32MultiArray) — per-template best sim, descending

Loads all reid_data/enrolled_*.npy files. Hot-reloads when any change on disk.
If multiple persons enrolled, matches the detection against ALL templates and
publishes the person+detection with the highest cosine similarity.

Usage:
    ros2 run g1_perception reid_matcher_node
    ros2 topic echo /g1/reid_match_name
    ros2 topic echo /g1/reid_target
"""

import math
import os
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import MarkerArray, Marker
from vision_msgs.msg import Detection3DArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from builtin_interfaces.msg import Duration
import message_filters
import torch
import torch.nn.functional as F

from g1_perception.reid_model import ReIDModel

# ── defaults ──────────────────────────────────────────────────────────────────
_WS = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
_REID_DATA      = os.path.normpath(os.path.join(_WS, 'reid_data'))
_MODEL_DEFAULT  = os.path.join(_REID_DATA, 'model_identity.pt')
_ENROLL_DIR = _REID_DATA   # scan for enrolled_*.npy here

N_PTS = 256
RNG   = np.random.default_rng(1)


def _extract_crop(pts_xyz: np.ndarray, box7) -> np.ndarray:
    cx, cy, cz, dx, dy, dz, yaw = box7.astype(float)
    valid = ~((pts_xyz[:, 0] == 0.0) & (pts_xyz[:, 1] == 0.0))
    pts = pts_xyz[valid] - np.array([cx, cy, cz])
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = (R @ pts.T).T
    half = np.array([dx / 2, dy / 2, dz / 2])
    mask = np.all(np.abs(pts) <= half, axis=1)
    p = pts[mask]
    if len(p) == 0:
        return np.zeros((N_PTS, 3), dtype=np.float32)
    idx = RNG.choice(len(p), N_PTS, replace=True)
    return p[idx].astype(np.float32)


def _quat_yaw(qx, qy, qz, qw) -> float:
    return 2.0 * math.atan2(qz, qw)


def _sim_to_rgb(sim: float):
    """Map cosine similarity [0,1] → (r,g,b) from blue→green→red."""
    t = float(np.clip(sim, 0.0, 1.0))
    if t < 0.5:
        r, g, b = 0.0, t * 2.0, 1.0 - t * 2.0
    else:
        r, g, b = (t - 0.5) * 2.0, 1.0 - (t - 0.5) * 2.0, 0.0
    return r, g, b


class ReIDMatcherNode(Node):
    def __init__(self):
        super().__init__('reid_matcher')

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter('model_path',      _MODEL_DEFAULT)
        self.declare_parameter('n_classes',       2)
        self.declare_parameter('emb_dim',         128)
        self.declare_parameter('min_score',       0.15)
        self.declare_parameter('max_range',       5.0)
        self.declare_parameter('sim_threshold',   0.55)   # publish reid_target only above this
        self.declare_parameter('detection_topic', '/g1/detections/livox')
        self.declare_parameter('lidar_topic',     '/livox/lidar')
        self.declare_parameter('target_frame',    'livox_frame')

        model_path    = self.get_parameter('model_path').value
        self.enrolled_dir = _ENROLL_DIR
        n_classes     = self.get_parameter('n_classes').value
        emb_dim       = self.get_parameter('emb_dim').value
        self.min_score    = self.get_parameter('min_score').value
        self.max_range    = self.get_parameter('max_range').value
        self.sim_thresh   = self.get_parameter('sim_threshold').value
        det_topic    = self.get_parameter('detection_topic').value
        lidar_topic  = self.get_parameter('lidar_topic').value
        self.target_frame = self.get_parameter('target_frame').value

        # ── model ─────────────────────────────────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = ReIDModel(n_classes=n_classes, emb_dim=emb_dim)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt)
            self.get_logger().info(f'Model loaded: {model_path}')
        else:
            self.get_logger().warn(f'Model not found: {model_path}')
        self.model.to(self.device).eval()

        # ── enrolled templates: {name → (128,) numpy} ─────────────────────────
        self._lock       = threading.Lock()
        self._templates  = {}    # name → np.ndarray (128,)
        self._tmpl_mtimes = {}   # name → float mtime
        self._load_all_templates()

        # ── publishers ────────────────────────────────────────────────────────
        self.pub_target  = self.create_publisher(PoseStamped,      '/g1/reid_target',     10)
        self.pub_name    = self.create_publisher(String,            '/g1/reid_match_name', 10)
        self.pub_markers = self.create_publisher(MarkerArray,       '/g1/reid_markers',    10)
        self.pub_scores  = self.create_publisher(Float32MultiArray, '/g1/reid_scores',     10)

        # ── template hot-reload timer (1 Hz) ──────────────────────────────────
        self.create_timer(1.0, self._check_template_reload)

        # ── synced subscriptions ───────────────────────────────────────────────
        qos = QoSPresetProfiles.SENSOR_DATA.value
        self._sub_det   = message_filters.Subscriber(self, Detection3DArray, det_topic,    qos_profile=qos)
        self._sub_lidar = message_filters.Subscriber(self, PointCloud2,      lidar_topic,  qos_profile=qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_det, self._sub_lidar], queue_size=5, slop=0.15
        )
        self._sync.registerCallback(self._cb_sync)

        with self._lock:
            names = list(self._templates.keys())
        self.get_logger().info(
            f'ReID matcher ready | sim_thresh={self.sim_thresh} | '
            f'enrolled={names if names else "NONE (enroll first)"}'
        )

    # ── template I/O ─────────────────────────────────────────────────────────

    def _load_all_templates(self):
        """Scan enrolled_dir for enrolled_*.npy and load/reload changed ones."""
        try:
            fnames = [f for f in os.listdir(self.enrolled_dir)
                      if f.startswith('enrolled_') and f.endswith('.npy')]
        except OSError:
            return
        loaded = []
        for fname in fnames:
            name = fname[len('enrolled_'):-len('.npy')]
            path = os.path.join(self.enrolled_dir, fname)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            with self._lock:
                if self._tmpl_mtimes.get(name) == mtime:
                    continue   # unchanged
            try:
                tmpl = np.load(path).astype(np.float32)
                norm = np.linalg.norm(tmpl)
                if norm > 0:
                    tmpl /= norm
                with self._lock:
                    self._templates[name]   = tmpl
                    self._tmpl_mtimes[name] = mtime
                self.get_logger().info(f'Template loaded: "{name}"  norm={norm:.4f}')
                loaded.append(name)
            except Exception as e:
                self.get_logger().warn(f'Template load failed ({fname}): {e}')
        # Remove templates whose files were deleted.
        try:
            on_disk = {f[len('enrolled_'):-len('.npy')]
                       for f in os.listdir(self.enrolled_dir)
                       if f.startswith('enrolled_') and f.endswith('.npy')}
        except OSError:
            on_disk = set()
        with self._lock:
            gone = [n for n in list(self._templates) if n not in on_disk]
            for n in gone:
                del self._templates[n]
                self._tmpl_mtimes.pop(n, None)
                self.get_logger().info(f'Template removed (file deleted): "{n}"')

    def _check_template_reload(self):
        self._load_all_templates()

    # ── main sync callback ────────────────────────────────────────────────────

    def _cb_sync(self, det_msg: Detection3DArray, cloud_msg: PointCloud2):
        with self._lock:
            templates = {n: t.copy() for n, t in self._templates.items()}

        if not templates:
            return   # nothing enrolled yet

        peds = self._parse_peds(det_msg)
        if not peds:
            self._publish_empty_markers(det_msg.header)
            return

        pts_xyz = self._parse_cloud(cloud_msg)
        if pts_xyz is None:
            return

        # Embed all peds in one batch.
        crops = np.stack(
            [_extract_crop(pts_xyz, p['box7']) for p in peds], axis=0
        )  # (M, N_PTS, 3)
        crops_t = torch.from_numpy(crops).to(self.device)
        with torch.no_grad():
            emb, _ = self.model(crops_t)   # (M, 128)
        emb_np = emb.cpu().numpy()         # (M, 128)

        # For each enrolled template, compute sim against all peds.
        # Find global (name, ped_idx) with highest sim.
        best_sim  = -1.0
        best_idx  = 0
        best_name = ''
        all_sims  = np.zeros(len(peds), dtype=np.float32)  # for marker colours

        names_sorted = sorted(templates.keys())
        score_report = []   # per-template best sim, sorted by name

        for name in names_sorted:
            tmpl = templates[name]
            sims = emb_np @ tmpl   # (M,)
            idx  = int(np.argmax(sims))
            sim  = float(sims[idx])
            score_report.append(sim)
            if sim > best_sim:
                best_sim  = sim
                best_idx  = idx
                best_name = name
                all_sims  = sims   # colour markers by the winning template's sims

        # Publish markers (coloured by winning template's similarity scores).
        self._publish_markers(det_msg.header, peds, all_sims, best_idx, best_name)

        # Publish per-template scores (sorted by name, same order each call).
        score_msg = Float32MultiArray()
        score_msg.data = score_report
        self.pub_scores.publish(score_msg)

        # Publish target + name only if above threshold.
        if best_sim >= self.sim_thresh:
            pose_msg = PoseStamped()
            pose_msg.header = det_msg.header
            pose_msg.header.frame_id = self.target_frame
            p = peds[best_idx]['box7']
            pose_msg.pose.position.x = float(p[0])
            pose_msg.pose.position.y = float(p[1])
            pose_msg.pose.position.z = float(p[2])
            yaw = float(p[6])
            pose_msg.pose.orientation.z = math.sin(yaw / 2.0)
            pose_msg.pose.orientation.w = math.cos(yaw / 2.0)
            self.pub_target.publish(pose_msg)
            self.pub_name.publish(String(data=best_name))
            self.get_logger().debug(
                f'Reid match: "{best_name}"  range={peds[best_idx]["r"]:.2f}m  sim={best_sim:.3f}'
            )
        else:
            self.get_logger().debug(
                f'No match above threshold (best={best_name} sim={best_sim:.3f} < {self.sim_thresh})'
            )

    # ── markers ───────────────────────────────────────────────────────────────

    def _publish_markers(self, header, peds, sims, best_idx, best_name=''):
        ma = MarkerArray()
        # Delete previous markers.
        del_marker = Marker()
        del_marker.header = header
        del_marker.ns     = 'reid'
        del_marker.action = Marker.DELETEALL
        ma.markers.append(del_marker)

        for i, (ped, sim) in enumerate(zip(peds, sims)):
            m = Marker()
            m.header   = header
            m.header.frame_id = self.target_frame
            m.ns       = 'reid'
            m.id       = i
            m.type     = Marker.CYLINDER
            m.action   = Marker.ADD
            box7 = ped['box7']
            m.pose.position.x = float(box7[0])
            m.pose.position.y = float(box7[1])
            m.pose.position.z = float(box7[2])
            m.pose.orientation.w = 1.0
            m.scale.x = float(box7[3]) or 0.6
            m.scale.y = float(box7[4]) or 0.6
            m.scale.z = float(box7[5]) or 1.8
            r, g, b = _sim_to_rgb(sim)
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.75 if i == best_idx else 0.45
            m.lifetime = Duration(sec=1, nanosec=0)
            ma.markers.append(m)

            # Text label: similarity score.
            t = Marker()
            t.header   = header
            t.header.frame_id = self.target_frame
            t.ns       = 'reid_text'
            t.id       = i
            t.type     = Marker.TEXT_VIEW_FACING
            t.action   = Marker.ADD
            t.pose.position.x = float(box7[0])
            t.pose.position.y = float(box7[1])
            t.pose.position.z = float(box7[2]) + float(box7[5]) / 2.0 + 0.2
            t.pose.orientation.w = 1.0
            t.scale.z = 0.3
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a = 1.0
            label = f'{best_name}:{sim:.2f}' if (i == best_idx and best_name) else f'{sim:.2f}'
            t.text = label + ('★' if i == best_idx else '')
            t.lifetime = Duration(sec=1, nanosec=0)
            ma.markers.append(t)

        self.pub_markers.publish(ma)

    def _publish_empty_markers(self, header):
        ma = MarkerArray()
        del_m = Marker()
        del_m.header = header
        del_m.ns     = 'reid'
        del_m.action = Marker.DELETEALL
        ma.markers.append(del_m)
        self.pub_markers.publish(ma)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parse_peds(self, det_msg: Detection3DArray):
        peds = []
        for det in det_msg.detections:
            if det.results:
                cid   = det.results[0].hypothesis.class_id
                score = det.results[0].hypothesis.score
                if cid not in ('pedestrian', 'human', 'person', '1', '2', 2):
                    continue
            else:
                score = 0.0
            if score < self.min_score:
                continue
            p = det.bbox.center.position
            r = math.hypot(p.x, p.y)
            if r > self.max_range:
                continue
            yaw = _quat_yaw(
                det.bbox.center.orientation.x,
                det.bbox.center.orientation.y,
                det.bbox.center.orientation.z,
                det.bbox.center.orientation.w,
            )
            dx = det.bbox.size.x or 0.8
            dy = det.bbox.size.y or 0.8
            dz = det.bbox.size.z or 1.8
            box7 = np.array([p.x, p.y, p.z, dx, dy, dz, yaw], dtype=np.float32)
            peds.append({'r': r, 'box7': box7})
        return peds

    def _parse_cloud(self, cloud_msg: PointCloud2):
        try:
            gen = pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)
            arr = np.array(list(gen), dtype=np.float32)
            if arr.ndim == 1 and len(arr) == 0:
                return None
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3]
            return np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f'Cloud parse error: {e}')
            return None


def main(args=None):
    rclpy.init(args=args)
    node = ReIDMatcherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
