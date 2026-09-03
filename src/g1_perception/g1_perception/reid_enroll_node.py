#!/usr/bin/env python3
"""
reid_enroll_node.py — Enroll named persons by capturing their LiDAR crops.

Subscribes:
    /g1/detections/livox  (vision_msgs/Detection3DArray)
    /livox/lidar          (sensor_msgs/PointCloud2)
    ~/enroll              (std_msgs/String) — person name to enroll; triggers capture
    ~/clear               (std_msgs/String) — name to delete ("" or "*" clears all)

Publishes:
    ~/status  (std_msgs/String) — human-readable enrollment progress

Each enrolled person is saved as reid_data/enrolled_{name}.npy.
The matcher loads all enrolled_*.npy files automatically.

Usage (real robot):
    ros2 run g1_perception reid_enroll_node

    # Enroll "Alice" (stand her within max_range, then publish):
    ros2 topic pub --once /reid_enroll/enroll std_msgs/msg/String "data: 'Alice'"

    # Clear one person:
    ros2 topic pub --once /reid_enroll/clear std_msgs/msg/String "data: 'Alice'"

    # Clear all:
    ros2 topic pub --once /reid_enroll/clear std_msgs/msg/String "data: ''"
"""

import math
import os
import re
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
import message_filters
import torch

from g1_perception.reid_model import ReIDModel

# ── defaults ──────────────────────────────────────────────────────────────────
_WS = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
_REID_DATA = os.path.normpath(os.path.join(_WS, 'reid_data'))
_MODEL_DEFAULT  = os.path.join(_REID_DATA, 'model_identity.pt')
_ENROLL_DEFAULT = os.path.join(_REID_DATA, 'enrolled_target.npy')

N_PTS   = 256   # points per crop
RNG     = np.random.default_rng(0)


def _extract_crop(pts_xyz: np.ndarray, box7) -> np.ndarray:
    """Crop & resample points inside box. Returns (N_PTS, 3) float32."""
    cx, cy, cz, dx, dy, dz, yaw = box7.astype(float)
    # Drop Mid-360 no-return placeholders (0,0,0).
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
    """Extract yaw from a quaternion (z-up convention)."""
    return 2.0 * math.atan2(qz, qw)


class ReIDEnrollNode(Node):
    def __init__(self):
        super().__init__('reid_enroll')

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter('model_path',    _MODEL_DEFAULT)
        self.declare_parameter('enrolled_path', _ENROLL_DEFAULT)
        self.declare_parameter('n_classes',     2)
        self.declare_parameter('emb_dim',       128)
        self.declare_parameter('n_enroll',      30)       # crops to collect
        self.declare_parameter('min_score',     0.15)
        self.declare_parameter('max_range',     5.0)      # metres
        self.declare_parameter('detection_topic', '/g1/detections/livox')
        self.declare_parameter('lidar_topic',     '/livox/lidar')

        model_path    = self.get_parameter('model_path').value
        enrolled_path = self.get_parameter('enrolled_path').value
        n_classes     = self.get_parameter('n_classes').value
        emb_dim       = self.get_parameter('emb_dim').value
        self.n_enroll    = self.get_parameter('n_enroll').value
        self.min_score   = self.get_parameter('min_score').value
        self.max_range   = self.get_parameter('max_range').value
        self.enrolled_dir = os.path.dirname(enrolled_path)  # reid_data/
        det_topic   = self.get_parameter('detection_topic').value
        lidar_topic = self.get_parameter('lidar_topic').value

        # ── model ─────────────────────────────────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = ReIDModel(n_classes=n_classes, emb_dim=emb_dim)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt)
            self.get_logger().info(f'Model loaded: {model_path}')
        else:
            self.get_logger().warn(f'Model not found: {model_path}  (running without weights)')
        self.model.to(self.device).eval()

        # ── enrollment state ──────────────────────────────────────────────────
        self._lock         = threading.Lock()
        self._collecting   = False
        self._enroll_name  = ''       # name being collected
        self._enroll_crops = []       # list of (N_PTS, 3) arrays

        # ── publisher ─────────────────────────────────────────────────────────
        self.pub_status = self.create_publisher(String, '~/status', 10)

        # ── topic triggers (String data = person name) ────────────────────────
        self.create_subscription(String, '~/enroll', self._cb_enroll, 10)
        self.create_subscription(String, '~/clear',  self._cb_clear,  10)

        # ── synced subscriptions ───────────────────────────────────────────────
        qos = QoSPresetProfiles.SENSOR_DATA.value
        self._sub_det   = message_filters.Subscriber(self, Detection3DArray, det_topic,   qos_profile=qos)
        self._sub_lidar = message_filters.Subscriber(self, PointCloud2,      lidar_topic, qos_profile=qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_det, self._sub_lidar], queue_size=5, slop=0.15
        )
        self._sync.registerCallback(self._cb_sync)

        # List existing enrolled persons.
        existing = self._list_enrolled()
        if existing:
            self.get_logger().info(f'Already enrolled: {existing}')
        self.get_logger().info(
            'ReID enroll node ready.\n'
            '  Enroll: ros2 topic pub --once ~/enroll std_msgs/msg/String "data: \'Alice\'"\n'
            '  Clear:  ros2 topic pub --once ~/clear  std_msgs/msg/String "data: \'Alice\'"'
        )

    # ── topic callbacks ───────────────────────────────────────────────────────

    def _cb_enroll(self, msg: String):
        name = msg.data.strip()
        if not name:
            self.get_logger().warn('Enroll: empty name — ignored.')
            return
        # Sanitise: allow only alphanumeric + _ -
        safe = re.sub(r'[^\w\-]', '_', name)
        with self._lock:
            if self._collecting:
                self._publish_status(
                    f'Already collecting "{self._enroll_name}" — wait for it to finish first.'
                )
                return
            self._collecting   = True
            self._enroll_name  = safe
            self._enroll_crops = []
        self._publish_status(
            f'Enrollment started for "{safe}". Collecting {self.n_enroll} crops '
            f'of nearest person within {self.max_range}m…'
        )

    def _cb_clear(self, msg: String):
        name = msg.data.strip()
        with self._lock:
            self._collecting   = False
            self._enroll_name  = ''
            self._enroll_crops = []
        if not name or name == '*':
            # Clear all enrolled templates.
            removed = []
            for fname in os.listdir(self.enrolled_dir):
                if fname.startswith('enrolled_') and fname.endswith('.npy'):
                    os.remove(os.path.join(self.enrolled_dir, fname))
                    removed.append(fname)
            status = f'Cleared all enrolled templates: {removed}' if removed else 'No templates to clear.'
        else:
            safe = re.sub(r'[^\w\-]', '_', name)
            path = os.path.join(self.enrolled_dir, f'enrolled_{safe}.npy')
            if os.path.exists(path):
                os.remove(path)
                status = f'Cleared: enrolled_{safe}.npy'
            else:
                status = f'No template found for "{safe}".'
        self._publish_status(status)

    # ── sync callback ─────────────────────────────────────────────────────────

    def _cb_sync(self, det_msg: Detection3DArray, cloud_msg: PointCloud2):
        with self._lock:
            if not self._collecting:
                return
            n_have = len(self._enroll_crops)

        peds = self._parse_peds(det_msg)
        if not peds:
            return

        nearest = min(peds, key=lambda d: d['r'])
        if nearest['r'] > self.max_range:
            return

        pts_xyz = self._parse_cloud(cloud_msg)
        if pts_xyz is None:
            return

        crop = _extract_crop(pts_xyz, nearest['box7'])
        if crop.max() == 0.0:
            return   # empty crop

        with self._lock:
            if not self._collecting:
                return
            self._enroll_crops.append(crop)
            n_have = len(self._enroll_crops)
            name   = self._enroll_name

        self._publish_status(f'[{name}] Collecting… {n_have}/{self.n_enroll}')

        if n_have >= self.n_enroll:
            self._finish_enrollment()

    # ── enrollment finish ─────────────────────────────────────────────────────

    def _finish_enrollment(self):
        with self._lock:
            if not self._collecting:
                return
            crops = list(self._enroll_crops)
            name  = self._enroll_name
            self._collecting   = False
            self._enroll_name  = ''
            self._enroll_crops = []

        crops_t = torch.from_numpy(
            np.stack(crops, axis=0)   # (N_ENROLL, N_PTS, 3)
        ).to(self.device)

        with torch.no_grad():
            emb, _ = self.model(crops_t)          # (N_ENROLL, 128)
        mean_emb = emb.mean(dim=0)                # (128,)
        mean_emb = torch.nn.functional.normalize(mean_emb, dim=0)
        template = mean_emb.cpu().numpy()          # (128,) float32

        out_path = os.path.join(self.enrolled_dir, f'enrolled_{name}.npy')
        np.save(out_path, template)
        self.get_logger().info(
            f'Enrolled "{name}" → {out_path}  '
            f'({len(crops)} crops, norm={np.linalg.norm(template):.4f})'
        )
        self._publish_status(
            f'Enrollment complete! "{name}" saved to {out_path}'
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _list_enrolled(self):
        names = []
        for fname in sorted(os.listdir(self.enrolled_dir)):
            if fname.startswith('enrolled_') and fname.endswith('.npy'):
                names.append(fname[len('enrolled_'):-len('.npy')])
        return names

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parse_peds(self, det_msg: Detection3DArray):
        """Return list of ped dicts with box7 and range r."""
        peds = []
        for det in det_msg.detections:
            if det.results and det.results[0].hypothesis.class_id not in ('pedestrian', 'human', 'person', '1', '2', 2):
                continue
            score = det.results[0].hypothesis.score if det.results else 0.0
            if score < self.min_score:
                continue
            p = det.bbox.center.position
            r = math.hypot(p.x, p.y)
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
        """Return (N, 3) float32 or None on error."""
        try:
            gen = pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)
            arr = np.array(list(gen), dtype=np.float32)
            if arr.ndim == 1 and len(arr) == 0:
                return None
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3]
            # structured array case
            return np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32)
        except Exception as e:
            self.get_logger().warn(f'Cloud parse error: {e}')
            return None

    def _publish_status(self, msg: str):
        self.get_logger().info(msg)
        self.pub_status.publish(String(data=msg))


def main(args=None):
    rclpy.init(args=args)
    node = ReIDEnrollNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
