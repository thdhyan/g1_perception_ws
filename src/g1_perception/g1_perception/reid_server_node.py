#!/usr/bin/env python3
"""
reid_server_node.py — Live ROS2 ReID lookup table backed by SMPL β.

Subscribes to /g1/smpl/tracks (JSON String from smpl_hmr_node) and
maintains an identity lookup table keyed on stable β vectors.

Publishes:
  /g1/reid/table      (std_msgs/String)  JSON snapshot of the full table
  /g1/reid/matches    (std_msgs/String)  JSON: per-frame track→identity map

Trigger topics (latch-free, fire-and-forget):
  /g1/reid/enroll     (std_msgs/String)  JSON {"track_id": int, "label": str}
  /g1/reid/remove     (std_msgs/Int32)   identity ID to evict
  /g1/reid/clear      (std_msgs/Empty)   wipe table

ROS2 parameters (all keyword-arg forwarded to BetaReIDTable.__init__):
  cos_thresh      float  0.85   cosine sim threshold for match
  ema_alpha       float  0.15   EMA weight on incoming β update
  delta_thresh    float  0.25   β L2-distance to flag as new-person candidate
  max_table_size  int    30     evict LRU when table exceeds this
  auto_enroll     bool   False  auto-add unmatched tracks as new identities
  min_stable_frames int  4      frames a new track must be seen before auto-enroll
  publish_rate_hz float  10.0   /g1/reid/table publish rate (independent of tracks)

Usage:
    ros2 run g1_perception reid_server_node

    # override params:
    ros2 run g1_perception reid_server_node \
        --ros-args -p cos_thresh:=0.80 -p auto_enroll:=true
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Empty, Int32, String


# ── ReID lookup table ──────────────────────────────────────────────────────────

class BetaReIDTable:
    """
    Thread-safe identity lookup table keyed on SMPL β vectors.

    All keyword args are tunable via ROS2 parameters of the same name.
    """

    def __init__(
        self,
        *,
        cos_thresh: float = 0.85,
        ema_alpha: float = 0.15,
        delta_thresh: float = 0.25,
        max_table_size: int = 30,
        auto_enroll: bool = False,
        min_stable_frames: int = 4,
    ):
        self.cos_thresh       = cos_thresh
        self.ema_alpha        = ema_alpha
        self.delta_thresh     = delta_thresh
        self.max_table_size   = max_table_size
        self.auto_enroll      = auto_enroll
        self.min_stable_frames = min_stable_frames

        self._lock   = threading.RLock()
        self._table: dict[int, dict[str, Any]] = {}   # identity_id → entry
        self._next_id = 1

        # track_id → {frames_seen, last_beta} for auto-enroll candidacy
        self._candidates: dict[int, dict[str, Any]] = {}

    # ── public API ─────────────────────────────────────────────────────────────

    def enroll(self, beta: np.ndarray, label: str = "") -> int:
        """Manually enroll a β vector. Returns the new identity ID."""
        with self._lock:
            return self._add_entry(self._norm(beta), label=label, manual=True)

    def remove(self, identity_id: int) -> bool:
        """Remove an identity by ID. Returns True if it existed."""
        with self._lock:
            return self._table.pop(identity_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._table.clear()
            self._candidates.clear()

    def match_tracks(self, tracks: list[dict]) -> dict[int, int | None]:
        """
        Match a list of track dicts (each must have 'id' and 'beta') against
        the table.

        Returns:
            dict[track_id → identity_id | None]
            identity_id is None when no match found and auto_enroll is False.

        Side-effects:
            • Updates EMA β for matched identities.
            • Flags large β-drift entries as 'drifting'.
            • If auto_enroll=True, enrolls stable unmatched tracks.
        """
        result: dict[int, int | None] = {}
        if not tracks:
            return result

        betas = np.array([self._norm(np.asarray(t["beta"], dtype=np.float32))
                          for t in tracks])       # (K, D)

        with self._lock:
            for i, track in enumerate(tracks):
                tid  = int(track["id"])
                beta = betas[i]                   # (D,) L2-normalised

                best_id, best_sim = self._best_match(beta)

                if best_id is not None and best_sim >= self.cos_thresh:
                    # ── matched: EMA update ────────────────────────────────
                    entry = self._table[best_id]
                    old_beta = entry["beta"]
                    new_beta = self._norm(
                        (1 - self.ema_alpha) * old_beta + self.ema_alpha * beta
                    )
                    drift = float(np.linalg.norm(new_beta - old_beta))
                    entry["beta"]       = new_beta
                    entry["last_seen"]  = time.time()
                    entry["frames"]    += 1
                    entry["drifting"]   = drift > self.delta_thresh
                    result[tid]         = best_id
                    self._candidates.pop(tid, None)     # no longer unmatched
                else:
                    # ── unmatched ──────────────────────────────────────────
                    result[tid] = None
                    if self.auto_enroll:
                        cand = self._candidates.setdefault(
                            tid, {"frames": 0, "last_beta": beta})
                        cand["frames"]    += 1
                        cand["last_beta"]  = beta
                        if cand["frames"] >= self.min_stable_frames:
                            new_id = self._add_entry(beta, label=f"auto_{tid}")
                            result[tid] = new_id
                            self._candidates.pop(tid, None)

        return result

    def snapshot(self) -> list[dict]:
        """Return JSON-serialisable list of all table entries."""
        with self._lock:
            out = []
            for iid, e in self._table.items():
                out.append({
                    "identity_id": iid,
                    "label":       e["label"],
                    "frames":      e["frames"],
                    "last_seen":   e["last_seen"],
                    "drifting":    e["drifting"],
                    "manual":      e["manual"],
                    "beta":        e["beta"].tolist(),
                })
            return out

    def params(self) -> dict:
        return {
            "cos_thresh":        self.cos_thresh,
            "ema_alpha":         self.ema_alpha,
            "delta_thresh":      self.delta_thresh,
            "max_table_size":    self.max_table_size,
            "auto_enroll":       self.auto_enroll,
            "min_stable_frames": self.min_stable_frames,
        }

    # ── internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-8)

    def _best_match(self, beta: np.ndarray) -> tuple[int | None, float]:
        if not self._table:
            return None, -1.0
        ids   = list(self._table.keys())
        stored = np.stack([self._table[i]["beta"] for i in ids])  # (N, D)
        sims   = stored @ beta                                      # (N,) cosine
        best_i = int(np.argmax(sims))
        return ids[best_i], float(sims[best_i])

    def _add_entry(self, beta: np.ndarray, label: str = "",
                   manual: bool = False) -> int:
        # Evict LRU if full.
        if len(self._table) >= self.max_table_size:
            oldest = min(self._table, key=lambda k: self._table[k]["last_seen"])
            del self._table[oldest]
        iid = self._next_id
        self._next_id += 1
        self._table[iid] = {
            "beta":      beta.copy(),
            "label":     label or f"person_{iid}",
            "frames":    1,
            "last_seen": time.time(),
            "drifting":  False,
            "manual":    manual,
        }
        return iid


# ── ROS2 node ──────────────────────────────────────────────────────────────────

class ReIDServerNode(Node):
    def __init__(self):
        super().__init__("reid_server_node")

        # ── declare parameters (keyword args forwarded to BetaReIDTable) ──────
        self.declare_parameter("cos_thresh",        0.85)
        self.declare_parameter("ema_alpha",         0.15)
        self.declare_parameter("delta_thresh",      0.25)
        self.declare_parameter("max_table_size",    30)
        self.declare_parameter("auto_enroll",       False)
        self.declare_parameter("min_stable_frames", 4)
        self.declare_parameter("publish_rate_hz",   10.0)

        # Build table from params.
        self._table = BetaReIDTable(
            cos_thresh        = self.get_parameter("cos_thresh").value,
            ema_alpha         = self.get_parameter("ema_alpha").value,
            delta_thresh      = self.get_parameter("delta_thresh").value,
            max_table_size    = self.get_parameter("max_table_size").value,
            auto_enroll       = self.get_parameter("auto_enroll").value,
            min_stable_frames = self.get_parameter("min_stable_frames").value,
        )

        qos_best  = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,  history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_rel   = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,     history=HistoryPolicy.KEEP_LAST, depth=10)

        # ── subscribers ───────────────────────────────────────────────────────
        self.create_subscription(String, "/g1/smpl/tracks",  self._cb_tracks,  qos_best)
        self.create_subscription(String, "/g1/reid/enroll",  self._cb_enroll,  qos_rel)
        self.create_subscription(Int32,  "/g1/reid/remove",  self._cb_remove,  qos_rel)
        self.create_subscription(Empty,  "/g1/reid/clear",   self._cb_clear,   qos_rel)

        # ── publishers ────────────────────────────────────────────────────────
        self._pub_table   = self.create_publisher(String, "/g1/reid/table",   qos_rel)
        self._pub_matches = self.create_publisher(String, "/g1/reid/matches", qos_best)

        # ── table publish timer ───────────────────────────────────────────────
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / rate_hz, self._publish_table)

        self.get_logger().info(
            f"ReID server ready — params: {self._table.params()}"
        )

    # ── callbacks ──────────────────────────────────────────────────────────────

    def _cb_tracks(self, msg: String) -> None:
        """Process incoming track list from smpl_hmr_node."""
        try:
            tracks = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"Bad tracks JSON: {e}")
            return

        # Filter to tracks that carry a beta field.
        valid = [t for t in tracks if "beta" in t and "id" in t]
        if not valid:
            return

        matches = self._table.match_tracks(valid)

        # Publish per-frame match result.
        result = {
            "stamp":   self.get_clock().now().nanoseconds,
            "matches": {str(k): v for k, v in matches.items()},
        }
        self._pub_matches.publish(String(data=json.dumps(result)))

        # Log new auto-enrollments.
        for tid, iid in matches.items():
            if iid is not None:
                entry = self._table.snapshot()
                labels = {e["identity_id"]: e["label"] for e in entry}
                label  = labels.get(iid, "?")
                # Only log first time (frames == 1).
                snap = [e for e in entry if e["identity_id"] == iid]
                if snap and snap[0]["frames"] == 1:
                    self.get_logger().info(
                        f"New identity enrolled: id={iid} label='{label}' "
                        f"from track {tid} (auto)"
                    )

    def _cb_enroll(self, msg: String) -> None:
        """
        Manual enroll trigger.
        Payload: {"track_id": int, "label": str, "beta": [10 floats]}
        OR       {"track_id": int, "label": str}  (uses last known β from table)
        """
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Bad enroll JSON: {e}")
            return

        if "beta" in payload:
            beta  = np.array(payload["beta"], dtype=np.float32)
            label = str(payload.get("label", ""))
            iid   = self._table.enroll(beta, label=label)
            self.get_logger().info(
                f"Manual enroll: identity_id={iid} label='{label}'"
            )
        else:
            self.get_logger().warn(
                "enroll message missing 'beta' field — "
                "send beta from smpl_hmr_node /g1/smpl/tracks output"
            )

    def _cb_remove(self, msg: Int32) -> None:
        removed = self._table.remove(msg.data)
        self.get_logger().info(
            f"Remove identity {msg.data}: {'ok' if removed else 'not found'}"
        )

    def _cb_clear(self, _msg: Empty) -> None:
        self._table.clear()
        self.get_logger().info("Table cleared")

    def _publish_table(self) -> None:
        snap = self._table.snapshot()
        payload = {
            "stamp":      self.get_clock().now().nanoseconds,
            "n_entries":  len(snap),
            "params":     self._table.params(),
            "identities": snap,
        }
        self._pub_table.publish(String(data=json.dumps(payload)))


# ── entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ReIDServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
