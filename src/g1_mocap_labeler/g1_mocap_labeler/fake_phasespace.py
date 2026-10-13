"""Fake phasespace_ros2 for testing without the lab.

Publishes the same interface as the real tracker: /phasespace/groups (latched JSON)
and TF phasespace -> <hat>, phasespace -> <hat>/led_<id> (hat TF unrotated, like a
hat with no tracker reference). Two human hats stand still; the robot hat
integrates the body-frame Twist on `cmd_topic` (goto_executor's dry-run output),
so the whole plan -> approve -> goto loop can run closed-loop on a laptop.
"""

import json
import math
import random

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
# 8 LEDs on a hat, asymmetric so the yaw fit is unambiguous (hat-local, x = facing)
LAYOUT = [(0.10, 0.0, 0.0), (0.07, 0.07, 0.02), (0.0, 0.09, 0.03), (-0.07, 0.07, 0.02),
          (-0.10, 0.0, 0.0), (-0.06, -0.06, 0.04), (0.0, -0.09, 0.03), (0.05, -0.05, 0.06)]


class FakePhaseSpace(Node):
    def __init__(self):
        super().__init__("fake_phasespace")
        self.frame = "phasespace"
        # name: [x, y, z, yaw_deg]
        self.hats = {
            "dhyan-hat": [1.5, 0.4, 1.70, 180.0],
            "dhyan-hat-2": [1.5, -0.6, 1.65, 150.0],
            "dhyan-hat-3": [-2.0, 0.0, 1.30, 0.0],
        }
        self.robot = self.declare_parameter("robot_group", "dhyan-hat-3").value
        self.noise = self.declare_parameter("noise", 0.003).value
        cmd_topic = self.declare_parameter("cmd_topic", "/intent/dry_run_cmd_vel").value
        self.cmd, self.dt = (0.0, 0.0, 0.0), 1.0 / 60.0
        self.tf_pub = TransformBroadcaster(self)
        groups = {n: {"leds": list(range(8 * i, 8 * i + 8)), "oriented": False} for i, n in enumerate(self.hats)}
        self.leds = {n: g["leds"] for n, g in groups.items()}
        pub = self.create_publisher(String, "/phasespace/groups", LATCHED)
        pub.publish(String(data=json.dumps({"frame": self.frame, "groups": groups})))
        self.create_subscription(Twist, cmd_topic, self.on_cmd, 10)
        self.create_timer(self.dt, self.tick)
        self.get_logger().info(f"fake PhaseSpace: {list(self.hats)}, robot {self.robot} follows {cmd_topic}")

    def on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.linear.y, msg.angular.z)

    def tick(self):
        r = self.hats[self.robot]
        vx, vy, wz = self.cmd
        yaw = math.radians(r[3])
        r[0] += (vx * math.cos(yaw) - vy * math.sin(yaw)) * self.dt
        r[1] += (vx * math.sin(yaw) + vy * math.cos(yaw)) * self.dt
        r[3] = math.degrees(yaw + wz * self.dt)
        now = self.get_clock().now().to_msg()
        tfs = []
        for name, (x, y, z, yaw_deg) in self.hats.items():
            c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
            tfs.append(self.tf(now, name, x, y, z))
            hidden = random.randrange(8)  # one LED occluded per frame
            for k, (lx, ly, lz) in enumerate(LAYOUT):
                if k == hidden:
                    continue
                n = lambda: random.gauss(0.0, self.noise)  # noqa: E731
                tfs.append(self.tf(now, f"{name}/led_{self.leds[name][k]}",
                                   x + c * lx - s * ly + n(), y + s * lx + c * ly + n(), z + lz + n()))
        self.tf_pub.sendTransform(tfs)

    def tf(self, stamp, child, x, y, z):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.frame
        t.child_frame_id = child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
        t.transform.rotation.w = 1.0
        return t


def main():
    rclpy.init()
    node = FakePhaseSpace()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
