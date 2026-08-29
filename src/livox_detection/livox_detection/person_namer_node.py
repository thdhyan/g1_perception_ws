#!/usr/bin/env python3
"""ROS 2 Node: Interactive person naming via keyboard.

Subscribes to /g1/detections/livox (Detection3DArray) and optionally
/g1/detections/reid (PersonReIDArray).  Displays a live numbered list of
pedestrians in the terminal; the operator presses a digit to select one,
types a name, and presses Enter.  Names are persisted to
~/.ros/person_names.json and published on /g1/person_names (std_msgs/String,
JSON payload).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

try:
    from g1_livox_pose_msgs.msg import PersonReIDArray
    HAS_REID = True
except ImportError:
    HAS_REID = False

# Persistent name store
_NAMES_PATH = Path.home() / ".ros" / "person_names.json"

# Display throttle: at most 2 Hz
_DISPLAY_PERIOD = 0.5


def _load_names() -> Dict[str, str]:
    """Load persisted db_id -> name mapping from disk."""
    if _NAMES_PATH.exists():
        try:
            return json.loads(_NAMES_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_names(names: Dict[str, str]) -> None:
    """Persist db_id -> name mapping to disk."""
    _NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _NAMES_PATH.write_text(json.dumps(names, indent=2))


class PersonNamerNode(Node):
    """Interactive terminal node for naming tracked persons."""

    def __init__(self) -> None:
        super().__init__("person_namer")

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._det_sub = self.create_subscription(
            Detection3DArray,
            "/g1/detections/livox",
            self._on_detections,
            qos,
        )

        if HAS_REID:
            self._reid_sub = self.create_subscription(
                PersonReIDArray,
                "/g1/detections/reid",
                self._on_reid,
                qos,
            )
        else:
            self.get_logger().warning(
                "g1_livox_pose_msgs not available — reid db_ids will not be used."
            )

        self._names_pub = self.create_publisher(String, "/g1/person_names", 10)

        # Shared state (protected by _lock)
        self._lock = threading.Lock()
        self._detections: List[dict] = []   # list of {pos, score, db_id}
        self._reid_map: Dict[int, int] = {}  # track_id -> db_id
        self._names: Dict[str, str] = _load_names()
        self._last_display_t: float = 0.0
        self._pending_index: Optional[int] = None  # digit the user pressed

        # Start keyboard thread
        self._running = True
        self._kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._kb_thread.start()

        self.get_logger().info("person_namer ready.  Waiting for detections…")

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _on_detections(self, msg: Detection3DArray) -> None:
        """Parse pedestrian detections and refresh the terminal display."""
        persons: List[dict] = []
        for det in msg.detections:
            # Only keep pedestrian-class hypotheses (label 1 or 'pedestrian')
            if not det.results:
                continue
            hyp = det.results[0]
            label = getattr(hyp, "hypothesis", None)
            class_id = getattr(label, "class_id", "") if label else ""
            score = getattr(label, "score", 0.0) if label else 0.0
            if class_id not in ("1", "pedestrian", "person"):
                continue

            # track_id lives in the detection id field
            track_id_str = det.id  # string in vision_msgs
            try:
                track_id = int(track_id_str)
            except (ValueError, TypeError):
                track_id = -1

            pos = det.bbox.center.position
            persons.append(
                {
                    "pos": (pos.x, pos.y),
                    "score": score,
                    "track_id": track_id,
                    "db_id": -1,  # filled below if reid available
                }
            )

        with self._lock:
            # Merge db_ids from latest reid map
            for p in persons:
                p["db_id"] = self._reid_map.get(p["track_id"], -1)
            self._detections = persons

            stamp = msg.header.stamp
            stamp_sec = stamp.sec + stamp.nanosec * 1e-9
            self._maybe_display(stamp_sec)

    def _on_reid(self, msg) -> None:  # PersonReIDArray
        """Update the track_id -> db_id mapping."""
        mapping: Dict[int, int] = {}
        for entry in msg.persons:
            mapping[entry.track_id] = entry.db_id
        with self._lock:
            self._reid_map = mapping

    # ------------------------------------------------------------------
    # Terminal display
    # ------------------------------------------------------------------

    def _maybe_display(self, stamp_sec: float) -> None:
        """Print the detection list if enough time has elapsed.  Must hold _lock."""
        now = time.monotonic()
        if now - self._last_display_t < _DISPLAY_PERIOD:
            return
        self._last_display_t = now
        self._print_list(stamp_sec)

    def _print_list(self, stamp_sec: float) -> None:
        """Print the current detections.  Must hold _lock."""
        lines = [f"\n=== Persons in view [stamp: {stamp_sec:.2f}] ==="]
        for i, p in enumerate(self._detections, start=1):
            x, y = p["pos"]
            score = p["score"]
            db_id = p["db_id"]
            if db_id >= 0:
                id_str = f"db:{db_id}"
                key = str(db_id)
                name_str = f'"{self._names[key]}"' if key in self._names else "(unnamed)"
            else:
                id_str = "new"
                name_str = "(unnamed)"
            lines.append(
                f"[{i}] pos=({x:5.1f},{y:5.1f}) score={score:.2f}  "
                f"id={id_str} {name_str}"
            )
        lines.append("Press 1-9 to select, then type name + Enter.  'q' to quit.")
        print("\n".join(lines), flush=True)

    # ------------------------------------------------------------------
    # Keyboard input thread
    # ------------------------------------------------------------------

    def _keyboard_loop(self) -> None:
        """Read lines from stdin and handle digit-select + name entry."""
        while self._running:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                break
            line = line.rstrip("\n")

            if line == "q":
                self.get_logger().info("Quit requested via keyboard.")
                rclpy.shutdown()
                break

            # If the line is a single digit, that's the selection
            if len(line) == 1 and line.isdigit():
                idx = int(line)
                with self._lock:
                    if idx < 1 or idx > len(self._detections):
                        print(
                            f"[namer] No detection at index {idx} "
                            f"(only {len(self._detections)} visible). Ignoring.",
                            flush=True,
                        )
                        continue
                    self._pending_index = idx
                print(
                    f"[namer] Selected [{idx}]. Type name and press Enter:",
                    flush=True,
                )
                continue

            # Otherwise treat as a name for the pending selection
            if self._pending_index is not None:
                name = line.strip()
                if not name:
                    print("[namer] Empty name — cancelled.", flush=True)
                    self._pending_index = None
                    continue

                with self._lock:
                    idx = self._pending_index
                    self._pending_index = None
                    if idx < 1 or idx > len(self._detections):
                        print(
                            f"[namer] Detection [{idx}] no longer exists. Ignoring.",
                            flush=True,
                        )
                        continue
                    p = self._detections[idx - 1]
                    db_id = p["db_id"]
                    key = str(db_id) if db_id >= 0 else f"track:{p['track_id']}"
                    self._names[key] = name
                    _save_names(self._names)
                    self._publish_names()
                    print(f"[namer] Saved: {key!r} -> {name!r}", flush=True)
            else:
                # Ignore stray input
                pass

    # ------------------------------------------------------------------
    # Publisher helper
    # ------------------------------------------------------------------

    def _publish_names(self) -> None:
        """Publish the full names dict as JSON.  Must hold _lock."""
        msg = String()
        msg.data = json.dumps(self._names)
        self._names_pub.publish(msg)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        self._running = False
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PersonNamerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
