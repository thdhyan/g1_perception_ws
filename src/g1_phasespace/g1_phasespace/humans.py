"""PhaseSpace LED groups -> SMPL-X humans for RViz and the G1 planner.

Every LED group from phasespace_ros2 (/phasespace/groups) except the robot
group is a person wearing a hat: the group TF is the head centre. When the
mocap labeler (/mocap/agents) names a group, its frame (<name>, carrying the
labeler's yaw) is used instead, so bodies face the heading set there.

Publishes:
  /phasespace/bodies   MarkerArray (latched)  SMPL-X neutral body per human, frame-locked to its group TF
  /phasespace/humans   std_msgs/String JSON (latched), G1_sim's /sim/humans schema:
                       {"frame", "humans": [{"name", "x", "y", "z", "heading_deg"}]}
                       heading_deg is null for groups with no orientation reference.
"""

import json
import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
COLORS = [(0.1, 1.0, 0.2), (1.0, 0.5, 0.1), (0.2, 0.6, 1.0), (1.0, 0.2, 0.8), (1.0, 1.0, 0.2)]


def load_smplx(path):
    """SMPL-X neutral template as (vertices, faces) in ROS axes, head centre at the origin."""
    d = np.load(path, allow_pickle=True)
    v, f = d["v_template"], d["f"]
    neck_y = (d["J_regressor"] @ v)[12, 1]  # joint 12 = neck
    head_center = v[v[:, 1] > neck_y].mean(0)
    v = v - head_center
    # SMPL-X: Y up, facing +Z, +X = body's left  ->  ROS: Z up, facing +X, +Y = left
    return v[:, [2, 0, 1]], f


class Humans(Node):
    def __init__(self):
        super().__init__("phasespace_humans")
        self.robot_group = self.declare_parameter("robot_group", "dhyan-hat-3").value
        smplx = os.path.expanduser(self.declare_parameter(
            "smplx_model", "~/Projects/HOVER/body_models/smplx/SMPLX_NEUTRAL.npz").value)
        rate = self.declare_parameter("rate", 10.0).value
        self.max_age = Duration(seconds=self.declare_parameter("max_age", 0.5).value)

        self.bodies_pub = self.create_publisher(MarkerArray, "/phasespace/bodies", LATCHED)
        self.humans_pub = self.create_publisher(String, "/phasespace/humans", LATCHED)
        self.create_subscription(String, "/phasespace/groups", self.on_groups, LATCHED)
        self.create_subscription(String, "/mocap/agents", self.on_agents, 10)
        self.labels = {}  # group -> labeler frame name
        self.tf = Buffer()
        TransformListener(self.tf, self)

        self.smpl = None
        if os.path.exists(smplx):
            self.smpl = load_smplx(smplx)
            self.get_logger().info(f"SMPL-X body from {smplx}")
        else:
            self.get_logger().warn(f"No SMPL-X model at {smplx}; bodies disabled, /phasespace/humans still published")
        self.frame, self.humans, self.shown = "phasespace", {}, {}  # shown: group -> frame its body is locked to
        self.create_timer(1.0 / rate, self.tick)

    def on_groups(self, msg):
        d = json.loads(msg.data)
        self.frame = d["frame"]
        self.humans = {n: g["oriented"] for n, g in d["groups"].items() if n != self.robot_group}
        self.get_logger().info(f"Humans: {sorted(self.humans)} (robot: {self.robot_group})")

    def on_agents(self, msg):
        self.labels = {a["group"]: a["name"] for a in json.loads(msg.data)["agents"] if a["yaw_deg"] is not None}

    def fresh_tf(self, frames, now):
        for f in frames:  # labeler frame first; group frame if the labeler is not running
            try:
                tf = self.tf.lookup_transform(self.frame, f, Time())
            except Exception:
                continue
            if now - Time.from_msg(tf.header.stamp) <= self.max_age:
                return f, tf
        return None, None

    def tick(self):
        now = self.get_clock().now()
        humans = []
        for name, oriented in self.humans.items():
            label = self.labels.get(name)
            frame, tf = self.fresh_tf([label, name] if label else [name], now)
            if tf is None:
                continue  # hat out of view
            oriented = oriented or frame == label
            p, q = tf.transform.translation, tf.transform.rotation
            heading = math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
            humans.append({"name": name, "frame": frame, "x": round(p.x, 3), "y": round(p.y, 3), "z": round(p.z, 3),
                           "heading_deg": round(heading, 1) if oriented else None})
        visible = {h["name"]: h.pop("frame") for h in humans}
        self.humans_pub.publish(String(data=json.dumps({"frame": self.frame, "humans": humans})))

        # bodies only once their TF exists, else RViz drops the latched markers
        if self.smpl is not None and not visible.items() <= self.shown.items():
            self.shown.update(visible)
            self.publish_bodies()

    def publish_bodies(self):
        v, f = self.smpl
        points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in v[f.reshape(-1)]]
        array = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])  # drop bodies locked to a replaced frame
        for i, name in enumerate(sorted(self.shown.values())):
            mk = Marker(ns="smplx", id=i, type=Marker.TRIANGLE_LIST, action=Marker.ADD)
            mk.header.frame_id = name
            mk.frame_locked = True
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 1.0
            r, g, b = COLORS[i % len(COLORS)]
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = r, g, b, 0.6
            mk.points = points
            label = Marker(ns="names", id=i, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, text=name)
            label.header.frame_id = name
            label.frame_locked = True
            label.pose.position.z = 0.25
            label.pose.orientation.w = 1.0
            label.scale.z = 0.12
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            array.markers += [mk, label]
        self.bodies_pub.publish(array)


def main():
    rclpy.init()
    node = Humans()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
