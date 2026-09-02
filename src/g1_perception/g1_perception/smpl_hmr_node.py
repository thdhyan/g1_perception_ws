#!/usr/bin/env python3
"""
smpl_hmr_node.py — Live SMPL body-mesh + β-tracker from LiDAR detections.

Pipeline:
  /livox/mid360/points  (PointCloud2)     ─┐
                                            ├─ ApproxTimeSyncer
  /g1/detections/livox  (Detection3DArray) ─┘
        │
        ▼
   Crop per-person point cloud (256 pts, de-yawed)
        │
        ▼
   LiDAR-HMR (VoteHMR / PMG, humanm3 ckpt)  →  β (10), θ (72)
        │
        ▼
   BetaTracker:  β lookup table  →  stable person IDs
        │
        ▼
  /g1/smpl/mesh       MarkerArray  – TRIANGLE_LIST, one marker/person
  /g1/smpl/joints     MarkerArray  – SPHERE,         24 joints/person
  /g1/smpl/skeleton   MarkerArray  – LINE_LIST,       skeleton edges
  /g1/smpl/tracks     String       – JSON: [{id, beta, x, y, z}, ...]

Tracking is pure β — no position gating, no temporal info.
Each person = one β vector in a lookup table. New detection compares
against all stored βs by cosine similarity; match → same person,
no match → new person.

Run (with venv active, after ros2 setup):
    ros2 run g1_perception smpl_hmr_node

Or:
    ros2 launch g1_perception smpl_hmr.launch.py
"""

from __future__ import annotations

import json
import math
import os
import site
import sys
import threading

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA, Int32, String
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

import message_filters
import sensor_msgs_py.point_cloud2 as pc2

# ── workspace root: file lives at <ws>/{src,build}/g1_perception/g1_perception/,
# so 3 levels up from its dirname reaches <ws> either way (symlink-install keeps
# the build-tree copy, not the src one, so this must work from both).
_WS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)

# ros2 launch runs this under system python3 with the venv's site-packages
# only on PYTHONPATH (not the actual interpreter), so .egg-installed deps
# (pointops, tensorboardx, ...) never get their easy-install.pth processed —
# site.addsitedir() does that registration explicitly.
_venv_site = os.path.join(
    _WS_ROOT, '.venv', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}',
    'site-packages')
if os.path.isdir(_venv_site):
    site.addsitedir(_venv_site)
_HMR_DIR = os.path.join(_WS_ROOT, 'LiDAR-HMR')

# SMPL-24 skeleton edges (parent → child pairs)
_SMPL_EDGES = [
    (0, 1), (0, 2), (0, 3),   # pelvis → L/R hip, spine1
    (1, 4), (2, 5),            # hips   → knees
    (4, 7), (5, 8),            # knees  → ankles
    (7, 10), (8, 11),          # ankles → feet
    (3, 6), (6, 9),            # spine chain
    (9, 12), (12, 15),         # spine  → neck → head
    (9, 13), (9, 14),          # spine  → collars
    (13, 16), (14, 17),        # collars → shoulders
    (16, 18), (17, 19),        # shoulders → elbows
    (18, 20), (19, 21),        # elbows → wrists
    (20, 22), (21, 23),        # wrists → hands
]

# Per-person palette (RGBA, pre-multiplied alpha=0.7 for mesh, 1.0 for joints)
_PALETTE = [
    (0.26, 0.85, 0.68),   # teal
    (0.98, 0.78, 0.31),   # amber
    (1.00, 0.48, 0.45),   # coral
    (0.60, 0.80, 1.00),   # sky
    (0.85, 0.60, 1.00),   # violet
    (0.40, 1.00, 0.60),   # mint
]


def _color(idx: int, alpha: float) -> ColorRGBA:
    r, g, b = _PALETTE[idx % len(_PALETTE)]
    return ColorRGBA(r=r, g=g, b=b, a=alpha)


def _quat_to_yaw(qx, qy, qz, qw) -> float:
    """Extract yaw from a quaternion (rotation about Z only)."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _extract_crop(pts_xyz: np.ndarray, box7: np.ndarray,
                  n_pts: int = 256, rng=None) -> np.ndarray:
    """Crop + de-yaw point cloud to a person box, sample n_pts points."""
    if rng is None:
        rng = np.random.default_rng(0)
    cx, cy, cz, dx, dy, dz, yaw = box7.astype(float)
    valid = ~((pts_xyz[:, 0] == 0.0) & (pts_xyz[:, 1] == 0.0))
    pts = pts_xyz[valid] - np.array([cx, cy, cz])
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = (R @ pts.T).T
    mask = ((np.abs(pts[:, 0]) <= dx / 2.0) &
            (np.abs(pts[:, 1]) <= dy / 2.0) &
            (np.abs(pts[:, 2]) <= dz / 2.0))
    pts_in = pts[mask]
    if len(pts_in) == 0:
        return np.zeros((n_pts, 3), dtype=np.float32)
    return pts_in[rng.choice(len(pts_in), n_pts, replace=True)].astype(np.float32)


def _verts_to_world(verts_local: np.ndarray, cx: float, cy: float, cz: float,
                    yaw: float) -> np.ndarray:
    """Apply inverse de-yaw + translate to put SMPL verts in LiDAR world frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (R @ verts_local.T).T + np.array([cx, cy, cz], dtype=np.float32)


# ── β lookup-table tracker ────────────────────────────────────────────────────

class BetaTracker:
    """Pure body-shape identity tracker. No position, no temporal info.

    Maintains a table:  person_id → β (10-d).
    Each frame: for each detected β, find closest stored β by cosine sim.
    If sim ≥ threshold → same person (EMA-update stored β).
    Otherwise → new person.

    Notes on HMR β quality:
      - LiDAR-HMR β estimates are noisy per-frame (robot-height sensor,
        sparse crops, no face points). Cosine similarity between same-person
        frames can be as low as 0.4. Threshold must be permissive.
      - LRU eviction caps table size so it doesn't grow unboundedly.
      - debug_sims=True logs max-sim scores to calibrate threshold.
    """

    def __init__(self, cos_thresh: float = 0.50, ema_alpha: float = 0.20,
                 max_table: int = 30, debug_sims: bool = False):
        self.table: dict[int, np.ndarray] = {}         # pid → β (10,)
        self._lru: dict[int, int] = {}                 # pid → frame_idx (for eviction)
        self._frame: int = 0
        self._next_id = 1
        self.cos_thresh = cos_thresh
        self.ema_alpha = ema_alpha                      # weight on incoming β for EMA
        self.max_table = max_table                      # evict oldest when full
        self.debug_sims = debug_sims
        self._all_ids: set[int] = set()                # cumulative unique IDs
        self._sim_log: list[float] = []                # recent best-sims (debug)

    def _norm(self, v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    def _evict_lru(self):
        """Remove the least-recently-used entry when table is full."""
        if len(self.table) >= self.max_table:
            oldest_pid = min(self._lru, key=lambda p: self._lru[p])
            del self.table[oldest_pid]
            del self._lru[oldest_pid]

    def match(self, betas: np.ndarray) -> list[int]:
        """betas: (K, 10). Returns list of person IDs, one per detection."""
        self._frame += 1
        ids = []
        claimed: set[int] = set()

        for beta in betas:
            bn = self._norm(beta)

            best_pid, best_sim = None, -1.0
            for pid, stored in self.table.items():
                if pid in claimed:
                    continue
                sim = float(np.dot(bn, self._norm(stored)))
                if sim > best_sim:
                    best_sim, best_pid = sim, pid

            if self.debug_sims and best_sim > -1.0:
                self._sim_log.append(best_sim)
                if len(self._sim_log) >= 50:
                    avg = sum(self._sim_log) / len(self._sim_log)
                    mn  = min(self._sim_log)
                    mx  = max(self._sim_log)
                    print(f'[BetaTracker] sim stats over 50 matches: '
                          f'avg={avg:.3f} min={mn:.3f} max={mx:.3f} '
                          f'thresh={self.cos_thresh:.2f}', flush=True)
                    self._sim_log.clear()

            if best_pid is not None and best_sim >= self.cos_thresh:
                # EMA update: blend new β into stored β
                self.table[best_pid] = (
                    (1.0 - self.ema_alpha) * self.table[best_pid]
                    + self.ema_alpha * beta
                )
                self._lru[best_pid] = self._frame
                ids.append(best_pid)
                claimed.add(best_pid)
            else:
                self._evict_lru()
                pid = self._next_id
                self._next_id += 1
                self.table[pid] = beta.copy()
                self._lru[pid] = self._frame
                self._all_ids.add(pid)
                ids.append(pid)
                claimed.add(pid)

        return ids

    @property
    def unique_total(self) -> int:
        """Total distinct person IDs assigned since node start."""
        return len(self._all_ids)

    @property
    def active_count(self) -> int:
        """IDs currently in the lookup table (≤ max_table)."""
        return len(self.table)

    def get_table(self) -> dict[int, list[float]]:
        """Return {pid: beta_list} for JSON serialisation."""
        return {pid: b.tolist() for pid, b in self.table.items()}


class SMPLHMRNode(Node):
    def __init__(self):
        super().__init__('smpl_hmr_node')

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter('checkpoint',       'humanm3')
        self.declare_parameter('config_path',      'configs/mesh/humanm3.yaml')
        self.declare_parameter('weights_path',     '')   # auto-resolved if empty
        self.declare_parameter('device',           'cuda')
        self.declare_parameter('min_score',        0.15)
        self.declare_parameter('max_range',        6.0)
        self.declare_parameter('n_pts',            256)
        self.declare_parameter('sync_slop',        0.15)   # seconds
        self.declare_parameter('marker_lifetime',  0.5)    # seconds
        self.declare_parameter('detection_topic',  '/g1/detections/livox')
        self.declare_parameter('cloud_topic',      '/livox/mid360/points')
        self.declare_parameter('show_mesh',        True)
        self.declare_parameter('show_skeleton',    True)
        self.declare_parameter('show_boxes',       False)
        self.declare_parameter('beta_cos_thresh',  0.50)   # β match threshold
        self.declare_parameter('beta_ema_alpha',   0.20)   # EMA weight on incoming β
        self.declare_parameter('beta_max_table',   30)     # LRU cap on tracker table
        self.declare_parameter('beta_debug_sims',  True)   # log sim score stats

        ckpt  = self.get_parameter('checkpoint').value
        cfg   = self.get_parameter('config_path').value
        wpath = self.get_parameter('weights_path').value or \
                os.path.join(_HMR_DIR, 'ckpts', ckpt, 'lidar_hmr_mesh.pth')
        self._device   = self.get_parameter('device').value
        self._min_score = self.get_parameter('min_score').value
        self._max_range = self.get_parameter('max_range').value
        self._n_pts    = self.get_parameter('n_pts').value
        self._lifetime = Duration(
            sec=int(self.get_parameter('marker_lifetime').value),
            nanosec=int((self.get_parameter('marker_lifetime').value % 1) * 1e9))
        self._show_mesh     = self.get_parameter('show_mesh').value
        self._show_skeleton = self.get_parameter('show_skeleton').value
        self._show_boxes    = self.get_parameter('show_boxes').value

        self._rng = np.random.default_rng(0)

        # ── load models (slow — done once at startup) ─────────────────────────
        self.get_logger().info(f'Loading LiDAR-HMR ({ckpt}) from {wpath} …')
        self._extractor = self._load_hmr(cfg, wpath)

        self.get_logger().info('Loading SMPL mesh decoder …')
        self._decoder = self._load_smpl_decoder()
        self._faces = self._decoder.faces  # (13776, 3) int32

        # Precompute face-expanded vertex indices for TRIANGLE_LIST (no rebuild per frame)
        self._face_flat = self._faces.flatten()   # (41328,) index array

        # Pre-allocate Point pools (one pool of 41328 pts per palette slot).
        # Avoids creating ~41k Point() objects every frame per person.
        _n_mesh = int(self._face_flat.shape[0])
        self._mesh_pt_pools = [
            [Point() for _ in range(_n_mesh)]
            for _ in range(len(_PALETTE))
        ]
        # Joint pool: 24 joints × n_palette
        self._joint_pt_pools = [
            [Point() for _ in range(24)]
            for _ in range(len(_PALETTE))
        ]
        # Skeleton edge pool: 2 endpoints × n_edges × n_palette
        _n_edges = len(_SMPL_EDGES)
        self._skel_pt_pools = [
            [Point() for _ in range(_n_edges * 2)]
            for _ in range(len(_PALETTE))
        ]

        # ── β tracker ─────────────────────────────────────────────────────────
        self._tracker = BetaTracker(
            cos_thresh=self.get_parameter('beta_cos_thresh').value,
            ema_alpha=self.get_parameter('beta_ema_alpha').value,
            max_table=self.get_parameter('beta_max_table').value,
            debug_sims=self.get_parameter('beta_debug_sims').value)

        # ── publishers ────────────────────────────────────────────────────────
        self._pub_mesh       = self.create_publisher(MarkerArray, '/g1/smpl/mesh',       10)
        self._pub_joint      = self.create_publisher(MarkerArray, '/g1/smpl/joints',     10)
        self._pub_skel       = self.create_publisher(MarkerArray, '/g1/smpl/skeleton',   10)
        self._pub_boxes      = self.create_publisher(MarkerArray, '/g1/smpl/boxes',      10)
        self._pub_tracks     = self.create_publisher(String,      '/g1/smpl/tracks',     10)
        self._pub_unique_ids = self.create_publisher(Int32,       '/g1/smpl/unique_ids', 10)

        # ── subscribers (time-synced) ─────────────────────────────────────────
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        cloud_topic = self.get_parameter('cloud_topic').value
        det_topic   = self.get_parameter('detection_topic').value
        slop        = self.get_parameter('sync_slop').value

        self._sub_cloud = message_filters.Subscriber(
            self, PointCloud2, cloud_topic, qos_profile=best_effort)
        self._sub_dets  = message_filters.Subscriber(
            self, Detection3DArray, det_topic)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_cloud, self._sub_dets],
            queue_size=10, slop=slop)
        self._sync.registerCallback(self._synced_callback)

        # Serialize HMR inference (GPU not thread-safe with ROS callbacks)
        self._infer_lock = threading.Lock()
        self._prev_pids: set[int] = set()  # for marker cleanup

        self.get_logger().info(
            f'smpl_hmr_node ready.\n'
            f'  cloud : {cloud_topic}\n'
            f'  dets  : {det_topic}\n'
            f'  ckpt  : {ckpt}  device={self._device}\n'
            f'  out   : /g1/smpl/{{mesh,joints,skeleton,boxes}}')

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_hmr(self, config_path: str, weights_path: str):
        """Load LiDAR-HMR (full backbone + SMPL regressor)."""
        import torch
        if _HMR_DIR not in sys.path:
            sys.path.insert(0, _HMR_DIR)

        cwd = os.getcwd()
        os.chdir(_HMR_DIR)
        try:
            from models.pmg_config import config, update_config
            update_config(config_path)
            from models.pose_mesh_net import LiDAR_HMR
            dev = torch.device(self._device)
            model = LiDAR_HMR(pmg_cfg=config, train_pmg=True, device=str(dev))
        finally:
            os.chdir(cwd)

        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location='cpu')
            for key in ('net', 'state_dict'):
                if key in state:
                    state = state[key]
                    break
            model.load_state_dict(state)
            self.get_logger().info(f'  weights loaded: {weights_path}')
        else:
            self.get_logger().warn(f'  weights not found: {weights_path} — running with random init!')

        model.to(torch.device(self._device))
        model.eval()

        # keep torch ref to avoid import overhead later
        self._torch = torch
        return model

    def _load_smpl_decoder(self):
        """CPU-only SMPL forward (mesh_utils.py in ws root)."""
        if _WS_ROOT not in sys.path:
            sys.path.insert(0, _WS_ROOT)
        from mesh_utils import SMPLMeshDecoder
        return SMPLMeshDecoder(device='cpu')

    # ── main callback ─────────────────────────────────────────────────────────

    def _synced_callback(self, cloud_msg: PointCloud2, dets_msg: Detection3DArray):
        header = cloud_msg.header

        # ── parse point cloud ──────────────────────────────────────────────
        # read_points returns a structured array (named fields), not plain
        # rows — stack the named columns into a plain (N,3) float32 array.
        pts_struct = pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'),
                                     skip_nans=True)
        if pts_struct.shape[0] == 0:
            return
        pts = np.stack([pts_struct['x'], pts_struct['y'], pts_struct['z']],
                       axis=-1).astype(np.float32)  # (N, 3)

        # ── filter detections ──────────────────────────────────────────────
        boxes7: list[np.ndarray] = []
        scores: list[float] = []
        for det in dets_msg.detections:
            score = max((r.hypothesis.score for r in det.results), default=0.0)
            if score < self._min_score:
                continue
            p = det.bbox.center.position
            s = det.bbox.size
            q = det.bbox.center.orientation
            cx, cy, cz = p.x, p.y, p.z
            if self._max_range > 0 and math.hypot(cx, cy) > self._max_range:
                continue
            yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
            boxes7.append(np.array([cx, cy, cz, s.x, s.y, s.z, yaw], dtype=np.float32))
            scores.append(score)

        if not boxes7:
            self._clear_all(header)
            return

        # ── crop per person ────────────────────────────────────────────────
        crops = np.stack([
            _extract_crop(pts, b, n_pts=self._n_pts, rng=self._rng)
            for b in boxes7], axis=0)  # (B, 256, 3)

        # ── HMR inference (serialized) ────────────────────────────────────
        with self._infer_lock:
            torch = self._torch
            with torch.no_grad():
                pcd = torch.from_numpy(crops).to(self._device)
                out = self._extractor(pcd)
            betas  = out['pose_beta'].cpu().numpy()   # (B, 10)
            thetas = out['pose_theta'].cpu().numpy()  # (B, 72)

        # ── β tracker: assign stable person IDs ─────────────────────────
        person_ids = self._tracker.match(betas)  # list[int], one per detection

        # ── SMPL forward + world-space transform ──────────────────────────
        per_person: list[dict] = []   # [{pid, verts, joints, box7}, ...]

        for i, box7 in enumerate(boxes7):
            cx, cy, cz, _, _, _, yaw = box7.astype(float)
            verts_local, joints_local = self._decoder.vertices(betas[i], thetas[i])
            verts_local  = verts_local[0]   # (6890, 3)
            joints_local = joints_local[0]  # (29, 3) — use first 24 (SMPL joints)

            per_person.append({
                'pid':    person_ids[i],
                'verts':  _verts_to_world(verts_local, cx, cy, cz, yaw),
                'joints': _verts_to_world(joints_local[:24], cx, cy, cz, yaw),
                'box7':   box7,
                'beta':   betas[i],
            })

        # ── publish /g1/smpl/tracks (JSON) + /g1/smpl/unique_ids ────────
        n_active = self._tracker.active_count
        n_unique = self._tracker.unique_total
        track_msg = String()
        track_msg.data = json.dumps({
            'persons': [
                {'id': p['pid'],
                 'beta': p['beta'].tolist(),
                 'x': float(p['box7'][0]),
                 'y': float(p['box7'][1]),
                 'z': float(p['box7'][2])}
                for p in per_person
            ],
            'active_ids': n_active,
            'unique_ids_total': n_unique,   # cumulative unique IDs in scene
            'table': self._tracker.get_table(),
        })
        self._pub_tracks.publish(track_msg)
        uid_msg = Int32()
        uid_msg.data = n_unique
        self._pub_unique_ids.publish(uid_msg)
        # Log unique ID count whenever it increases
        if n_unique > getattr(self, '_last_uid_log', 0):
            self.get_logger().info(
                f'[tracker] active={n_active}  unique_total={n_unique}')
            self._last_uid_log = n_unique

        # ── build and publish markers (keyed by person_id → stable color) ─
        mesh_ma  = MarkerArray()
        joint_ma = MarkerArray()
        skel_ma  = MarkerArray()
        box_ma   = MarkerArray()

        # Clean up markers for person IDs not seen this frame
        seen_pids = {p['pid'] for p in per_person}
        for old_pid in list(self._prev_pids - seen_pids):
            mesh_ma.markers.append( self._delete_marker(old_pid,      'smpl_mesh',   header))
            joint_ma.markers.append(self._delete_marker(old_pid * 2,  'smpl_joints', header))
            skel_ma.markers.append( self._delete_marker(old_pid * 2 + 1, 'smpl_skel', header))
            box_ma.markers.append(  self._delete_marker(old_pid,      'smpl_boxes',  header))
        self._prev_pids = seen_pids

        for p in per_person:
            pid = p['pid']

            if self._show_mesh:
                mesh_ma.markers.append(
                    self._mesh_marker(pid, p['verts'], self._faces, header))

            if self._show_skeleton:
                joint_ma.markers.append(self._joint_marker(pid * 2,     pid, p['joints'], header))
                skel_ma.markers.append( self._skel_marker( pid * 2 + 1, pid, p['joints'], header))

            if self._show_boxes:
                box_ma.markers.append(self._box_marker(pid, p['box7'], header))

        self._pub_mesh.publish(mesh_ma)
        self._pub_joint.publish(joint_ma)
        self._pub_skel.publish(skel_ma)
        if self._show_boxes:
            self._pub_boxes.publish(box_ma)

    # ── marker builders ───────────────────────────────────────────────────────

    def _base_marker(self, mid: int, ns: str, header, mtype: int) -> Marker:
        m = Marker()
        m.header = header
        m.ns     = ns
        m.id     = mid
        m.type   = mtype
        m.action = Marker.ADD
        m.lifetime = self._lifetime
        m.frame_locked = False
        return m

    def _delete_marker(self, mid: int, ns: str, header) -> Marker:
        m = Marker()
        m.header = header
        m.ns   = ns
        m.id   = mid
        m.action = Marker.DELETE
        return m

    def _mesh_marker(self, idx: int, verts: np.ndarray,
                     faces: np.ndarray, header) -> Marker:
        """TRIANGLE_LIST: 13776 triangles = 41328 points.
        Uses pre-allocated Point pool — no per-frame allocation."""
        m = self._base_marker(idx, 'smpl_mesh', header, Marker.TRIANGLE_LIST)
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color = _color(idx, alpha=0.65)

        tri_verts = verts[self._face_flat]   # (41328, 3) numpy, fast indexing
        pool = self._mesh_pt_pools[idx % len(_PALETTE)]
        # Update coordinates in-place — avoids 41k Point() allocations per frame
        for p, v in zip(pool, tri_verts):
            p.x = float(v[0]); p.y = float(v[1]); p.z = float(v[2])
        m.points = pool
        return m

    def _joint_marker(self, mid: int, idx: int,
                      joints: np.ndarray, header) -> Marker:
        """SPHERE_LIST: one sphere per SMPL joint. Pre-allocated pool."""
        m = self._base_marker(mid, 'smpl_joints', header, Marker.SPHERE_LIST)
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color = _color(idx, alpha=1.0)
        pool = self._joint_pt_pools[idx % len(_PALETTE)]
        for p, j in zip(pool, joints):
            p.x = float(j[0]); p.y = float(j[1]); p.z = float(j[2])
        m.points = pool
        return m

    def _skel_marker(self, mid: int, idx: int,
                     joints: np.ndarray, header) -> Marker:
        """LINE_LIST: skeleton edges. Pre-allocated pool (2 pts per edge)."""
        m = self._base_marker(mid, 'smpl_skel', header, Marker.LINE_LIST)
        m.scale.x = 0.015
        m.color = _color(idx, alpha=1.0)
        pool = self._skel_pt_pools[idx % len(_PALETTE)]
        pi = 0
        for a, b in _SMPL_EDGES:
            if a < len(joints) and b < len(joints):
                pool[pi].x = float(joints[a, 0]); pool[pi].y = float(joints[a, 1]); pool[pi].z = float(joints[a, 2])
                pi += 1
                pool[pi].x = float(joints[b, 0]); pool[pi].y = float(joints[b, 1]); pool[pi].z = float(joints[b, 2])
                pi += 1
        m.points = pool[:pi]
        return m

    def _box_marker(self, idx: int, box7: np.ndarray, header) -> Marker:
        """CUBE for the detection bounding box."""
        from geometry_msgs.msg import Vector3
        cx, cy, cz, dx, dy, dz, yaw = box7.astype(float)
        m = self._base_marker(idx, 'smpl_boxes', header, Marker.CUBE)
        m.pose.position    = Point(x=cx, y=cy, z=cz)
        m.pose.orientation.z = math.sin(yaw / 2.0)
        m.pose.orientation.w = math.cos(yaw / 2.0)
        m.scale             = Vector3(x=dx, y=dy, z=dz)
        m.color             = _color(idx, alpha=0.15)
        return m

    def _clear_all(self, header):
        """Publish DELETE ALL for all namespaces."""
        for pub, ns in [(self._pub_mesh, 'smpl_mesh'),
                        (self._pub_joint, 'smpl_joints'),
                        (self._pub_skel, 'smpl_skel'),
                        (self._pub_boxes, 'smpl_boxes')]:
            ma = MarkerArray()
            m  = Marker()
            m.header = header
            m.ns     = ns
            m.id     = 0
            m.action = Marker.DELETEALL
            ma.markers.append(m)
            pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = SMPLHMRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
