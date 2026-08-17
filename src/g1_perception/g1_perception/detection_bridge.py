#!/usr/bin/env python3
"""Run G1_sim's detection pipeline against whatever is on ``/livox/mid360/points``.

``G1_sim/detection/detection_node.py`` is deliberately a standalone script (see
its own docstring): it runs against system Python and its own torch/CUDA
dependency set, not the ROS2 workspace's ament_python venv, so heavy inference
never stalls or gets tangled up with this package's build. This node is a thin
wrapper that launches it as a subprocess with the right arguments and forwards
its exit; it does not duplicate any detection logic.

Point the ``detection_node_path`` parameter at your checked-out G1_sim if it is
not a sibling of this workspace.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node

def _find_default_detection_node() -> Path:
    """Locate G1_sim/detection/detection_node.py near this file.

    Walks up from this file's location looking for a ``G1_sim`` sibling of any
    ancestor directory. Works whether this package is run from the ``src``
    tree (``colcon build --symlink-install``) or a copied ``install`` tree,
    since both nest arbitrarily deep under the same ``Projects/thesis`` root
    in the expected layout (``g1_perception_ws`` and ``G1_sim`` as siblings).
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor.parent / "G1_sim" / "detection" / "detection_node.py"
        if candidate.is_file():
            return candidate
    # Fall back to the documented default layout even if not found yet, so the
    # error message on failure points at the expected path.
    return here.parents[1].parent.parent / "G1_sim" / "detection" / "detection_node.py"


DEFAULT_DETECTION_NODE = _find_default_detection_node()


class DetectionBridge(Node):
    def __init__(self) -> None:
        super().__init__("g1_detection_bridge")

        self.declare_parameter("detection_node_path", str(DEFAULT_DETECTION_NODE))
        self.declare_parameter("backend", "livox")
        self.declare_parameter("checkpoint", "pt/livox_model_1.pt")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("score_threshold", 0.4)
        self.declare_parameter("input_topic", "/livox/mid360/points")
        self.declare_parameter("frame", "")

        node_path = Path(self.get_parameter("detection_node_path").value)
        if not node_path.is_file():
            raise FileNotFoundError(
                f"detection_node.py not found at {node_path}; set the "
                "'detection_node_path' parameter to your G1_sim checkout."
            )

        cmd = [
            sys.executable,
            str(node_path),
            "--backend",
            self.get_parameter("backend").value,
            "--checkpoint",
            self.get_parameter("checkpoint").value,
            "--device",
            self.get_parameter("device").value,
            "--score-threshold",
            str(self.get_parameter("score_threshold").value),
            "--input-topic",
            self.get_parameter("input_topic").value,
        ]
        frame = self.get_parameter("frame").value
        if frame:
            cmd += ["--frame", frame]

        self.get_logger().info(f"launching detection_node: {' '.join(cmd)}")
        # cwd = detection/ so its relative --checkpoint default/argument resolves.
        self.proc = subprocess.Popen(cmd, cwd=str(node_path.parent))

        # Poll periodically so a crashed detection_node takes this bridge down
        # with it instead of silently spinning with nothing behind it.
        self.create_timer(2.0, self._check_alive)

    def _check_alive(self) -> None:
        code = self.proc.poll()
        if code is not None:
            self.get_logger().error(f"detection_node exited with code {code}")
            raise SystemExit(code)

    def destroy_node(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = DetectionBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
