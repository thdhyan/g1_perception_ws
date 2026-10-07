"""PhaseSpace robot-head hat -> G1 head-centre frame and pose.

The robot's LED group (default dhyan-hat-3) sits on the G1 head. Its group TF
from phasespace_ros2 is the LED centroid (rotated once a reference layout is
captured); `offset` (metres, in the hat frame) moves that to the head centre.

Publishes:
  /tf_static              <robot_group> -> <head_frame> (offset, no rotation)
  /phasespace/robot_head  PoseStamped of <head_frame> in the PhaseSpace frame
  /phasespace/robot       MarkerArray (latched): head box + facing arrow, frame-locked to <head_frame>
"""

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class RobotHead(Node):
    def __init__(self):
        super().__init__("phasespace_robot_head")
        self.group = self.declare_parameter("robot_group", "dhyan-hat-3").value
        self.head = self.declare_parameter("head_frame", "ps_robot_head").value
        self.frame = self.declare_parameter("world_frame", "phasespace").value
        offset = self.declare_parameter("offset", [0.0, 0.0, 0.0]).value
        rate = self.declare_parameter("rate", 30.0).value
        self.max_age = Duration(seconds=self.declare_parameter("max_age", 0.5).value)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.group
        t.child_frame_id = self.head
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in offset)
        t.transform.rotation.w = 1.0
        self.static_tf = StaticTransformBroadcaster(self)
        self.static_tf.sendTransform(t)

        self.pose_pub = self.create_publisher(PoseStamped, "/phasespace/robot_head", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/phasespace/robot", LATCHED)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.shown = False
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f"Robot head: {self.group} -> {self.head}, offset {list(offset)} m")

    def tick(self):
        try:
            tf = self.tf.lookup_transform(self.frame, self.head, Time())
        except Exception:
            return
        if self.get_clock().now() - Time.from_msg(tf.header.stamp) > self.max_age:
            return  # hat out of view
        pose = PoseStamped()
        pose.header = tf.header
        p = tf.transform.translation
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p.x, p.y, p.z
        pose.pose.orientation = tf.transform.rotation
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
