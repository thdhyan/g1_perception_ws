"""PhaseSpace robot-head hat -> G1 head-centre frame and pose.

The robot's LED group (default dhyan-hat-3) sits on the G1 head. Its group TF
from phasespace_ros2 is the LED centroid (rotated once a reference layout is
captured); `offset` (metres, in the hat frame) moves that to the head centre.
When the mocap labeler (/mocap/agents) names the robot group, the head frame is
re-parented onto the labeler's frame, so it carries the labeler's yaw. It is
sent on /tf every tick: Jazzy's StaticTransformBroadcaster drops any later
transform for a child frame it has already sent, so a static one cannot re-parent.

Publishes:
  /tf                     <robot_group or labeler name> -> <head_frame> (offset, no rotation)
  /phasespace/robot_head  PoseStamped of <head_frame> in the PhaseSpace frame
  /phasespace/robot       MarkerArray (latched): head box + facing arrow, frame-locked to <head_frame>
"""

import json

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def rotate(q, v):
    """Rotate vector v by unit quaternion q (geometry_msgs Quaternion)."""
    u = (q.x, q.y, q.z)
    t = [2 * (u[1] * v[2] - u[2] * v[1]), 2 * (u[2] * v[0] - u[0] * v[2]), 2 * (u[0] * v[1] - u[1] * v[0])]
    return [v[i] + q.w * t[i] + (u[(i + 1) % 3] * t[(i + 2) % 3] - u[(i + 2) % 3] * t[(i + 1) % 3]) for i in range(3)]


class RobotHead(Node):
    def __init__(self):
        super().__init__("phasespace_robot_head")
        self.group = self.declare_parameter("robot_group", "dhyan-hat-3").value
        self.head = self.declare_parameter("head_frame", "ps_robot_head").value
        self.frame = self.declare_parameter("world_frame", "phasespace").value
        self.offset = self.declare_parameter("offset", [0.0, 0.0, 0.0]).value
        rate = self.declare_parameter("rate", 30.0).value
        self.max_age = Duration(seconds=self.declare_parameter("max_age", 0.5).value)

        self.tf_pub = TransformBroadcaster(self)
        self.parent = None
        self.set_parent(self.group)
        self.create_subscription(String, "/mocap/agents", self.on_agents, 10)

        self.pose_pub = self.create_publisher(PoseStamped, "/phasespace/robot_head", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/phasespace/robot", LATCHED)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.shown = False
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f"Robot head: {self.group} -> {self.head}, offset {list(self.offset)} m")

    def set_parent(self, parent):
        if parent == self.parent:
            return
        self.parent = parent
        self.get_logger().info(f"Robot head parent: {parent}")

    def on_agents(self, msg):
        for a in json.loads(msg.data)["agents"]:
            if a["group"] == self.group:
                self.set_parent(a["name"] if a["yaw_deg"] is not None else self.group)

    def tick(self):
        try:
            tf = self.tf.lookup_transform(self.frame, self.parent, Time())
        except Exception:
            return
        if self.get_clock().now() - Time.from_msg(tf.header.stamp) > self.max_age:
            return  # hat out of view
        t = TransformStamped()
        t.header.stamp = tf.header.stamp
        t.header.frame_id = self.parent
        t.child_frame_id = self.head
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in self.offset)
        t.transform.rotation.w = 1.0
        self.tf_pub.sendTransform(t)
        pose = PoseStamped()
        pose.header = tf.header
        p, q = tf.transform.translation, tf.transform.rotation
        d = rotate(q, [float(v) for v in self.offset])
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p.x + d[0], p.y + d[1], p.z + d[2]
        pose.pose.orientation = q
        self.pose_pub.publish(pose)
        if not self.shown:  # markers only once the frame exists, else RViz drops them
            self.shown = True
            self.publish_markers()

    def publish_markers(self):
        head = Marker(ns="robot_head", id=0, type=Marker.CUBE, action=Marker.ADD)
        head.scale.x, head.scale.y, head.scale.z = 0.15, 0.2, 0.2  # ~G1 head
        arrow = Marker(ns="robot_head", id=1, type=Marker.ARROW, action=Marker.ADD)
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.35, 0.03, 0.03  # along head +x
        label = Marker(ns="robot_head", id=2, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, text="G1 head")
        label.pose.position.z = 0.25
        label.scale.z = 0.12
        for mk in (head, arrow, label):
            mk.header.frame_id = self.head
            mk.frame_locked = True
            mk.pose.orientation.w = 1.0
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = 1.0, 0.15, 0.15, 0.8
        label.color.g = label.color.b = 1.0
        self.marker_pub.publish(MarkerArray(markers=[head, arrow, label]))


def main():
    rclpy.init()
    node = RobotHead()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
