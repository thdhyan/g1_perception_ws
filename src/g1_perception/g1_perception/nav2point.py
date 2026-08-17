#!/usr/bin/env python3
"""
Nav2Point navigation controller.

Ported from github.com/thdhyan/g1pilot

Handles waypoint navigation and human-aware approach:
- 60cm standoff: robot stops 60cm in front of detected/selected human, facing them
- Subscribes to /g1/selected_human for human targets (applies standoff affordance)
- Subscribes to /g1pilot/path for Nav2 planned paths (no standoff)
- Subscribes to /g1/nav_goal for direct point goals (converted to Path, no standoff)
- Publishes joy commands to /g1pilot/auto_joy -> loco_client -> robot SDK

The standoff computation handles two cases:
1. Human pose with valid yaw: stop 60cm in front using human's facing direction
2. Identity quaternion (no yaw info): stop 60cm between robot and human
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, Bool

def yaw_from_quat(x, y, z, w):
    """Convert quaternion to yaw angle (rotation about z-axis)."""
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)

class Nav2Point(Node):
    def __init__(self):
        super().__init__('nav2point')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('pos_kp', 0.8)
        self.declare_parameter('yaw_kp', 1.5)
        self.declare_parameter('waypoint_tolerance', 0.20)
        self.declare_parameter('goal_tolerance', 0.10)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('joy_topic', '/g1pilot/auto_joy')
        self.declare_parameter('path_topic', '/g1pilot/path')
        self.declare_parameter('auto_enable_topic', '/g1pilot/auto_enable')
        self.declare_parameter('vx_limit', 0.6)
        self.declare_parameter('vy_limit', 0.6)
        self.declare_parameter('wz_limit', 0.5)
        self.declare_parameter('standoff_distance', 0.60)
        self.declare_parameter('human_goal_topic', '/g1/selected_human')
        self.declare_parameter('nav_goal_topic', '/g1/nav_goal')

        self.rate = self.get_parameter('publish_rate').value
        self.pos_kp = self.get_parameter('pos_kp').value
        self.yaw_kp = self.get_parameter('yaw_kp').value
        self.wp_tol = self.get_parameter('waypoint_tolerance').value
        self.goal_tol = self.get_parameter('goal_tolerance').value
        self.frame_id = self.get_parameter('frame_id').value
        self.joy_topic = self.get_parameter('joy_topic').value
        self.path_topic = self.get_parameter('path_topic').value
        self.vx_lim = self.get_parameter('vx_limit').value
        self.vy_lim = self.get_parameter('vy_limit').value
        self.wz_lim = self.get_parameter('wz_limit').value
        self.auto_enable_topic = self.get_parameter('auto_enable_topic').value
        self.standoff_distance = self.get_parameter('standoff_distance').value
        self.human_goal_topic = self.get_parameter('human_goal_topic').value
        self.nav_goal_topic = self.get_parameter('nav_goal_topic').value

        qos = QoSProfile(depth=10)
        # Real G1 robot: use /unitree/slam_mapping/odom (confirmed via ros2 topic list).
        # g1pilot used /lidar_odometry/pose_fixed — override via odom_topic param if needed.
        odom_topic = self.get_parameter('odom_topic').value if self.has_parameter('odom_topic') else '/unitree/slam_mapping/odom'
        self.declare_parameter('odom_topic', odom_topic)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.cb_odom, qos)
        self.sub_auto_enable = self.create_subscription(Bool, self.auto_enable_topic, self.cb_auto_enable, qos)
        self.sub_path = self.create_subscription(Path, self.path_topic, self.cb_path, qos)
        self.sub_human_goal = self.create_subscription(PoseStamped, self.human_goal_topic, self.cb_human_goal, qos)
        self.sub_nav_goal = self.create_subscription(PoseStamped, self.nav_goal_topic, self.cb_nav_goal, qos)

        self.pub_joy = self.create_publisher(Joy, self.joy_topic, qos)
        self.pub_wp_marker = self.create_publisher(Marker, '/g1pilot/waypoint_marker', qos)
        self.pub_goal_marker = self.create_publisher(Marker, '/g1pilot/goal_marker', qos)
        self.pub_standoff_marker = self.create_publisher(Marker, '/g1/standoff_marker', qos)
        self.pub_auto_enable = self.create_publisher(Bool, self.auto_enable_topic, qos)

        self.timer = self.create_timer(1.0 / self.rate, self.loop)

        self.have_pose = False
        self.auto_enabled = False
        self.path = []
        self.path_frame = self.frame_id
        self.idx = 0
        self.x = self.y = self.yaw = 0.0
        self.logged_no_pose = False
        self.logged_no_path = False
        self.logged_end_path = False

    def cb_odom(self, msg: Odometry):
        """Update robot pose from odometry."""
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        qx, qy, qz, qw = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        self.yaw = yaw_from_quat(qx, qy, qz, qw)
        self.have_pose = True
        self.logged_no_pose = False

    def cb_auto_enable(self, msg: Bool):
        """Update automation enable state."""
        self.auto_enabled = msg.data

    def cb_path(self, msg: Path):
        """Receive Nav2 planned path (no standoff applied)."""
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.path_frame = msg.header.frame_id if msg.header.frame_id else self.frame_id
        self.idx = 0
        self.logged_no_path = False
        self.logged_end_path = False
        if self.path:
            self.publish_goal_marker(self.path[-1][0], self.path[-1][1])

    def cb_human_goal(self, msg: PoseStamped):
        """
        Receive human target pose and generate standoff waypoint.

        Computes a point 60cm in front of the human, then creates a synthetic
        single-waypoint path to that standoff location.
        """
        hx = float(msg.pose.position.x)
        hy = float(msg.pose.position.y)
        hqx = float(msg.pose.orientation.x)
        hqy = float(msg.pose.orientation.y)
        hqz = float(msg.pose.orientation.z)
        hqw = float(msg.pose.orientation.w)

        human_yaw = yaw_from_quat(hqx, hqy, hqz, hqw)

        # Check if quaternion is approximately identity (no meaningful yaw)
        is_identity = (abs(hqw - 1.0) < 0.1 and
                      abs(hqx) < 0.1 and abs(hqy) < 0.1 and abs(hqz) < 0.1)

        if is_identity:
            # No valid human yaw; compute standoff from robot to human
            bearing = math.atan2(hy - self.y, hx - self.x)
            standoff_x = hx - self.standoff_distance * math.cos(bearing)
            standoff_y = hy - self.standoff_distance * math.sin(bearing)
        else:
            # Use human's facing direction to compute standoff point
            standoff_x = hx - self.standoff_distance * math.cos(human_yaw)
            standoff_y = hy - self.standoff_distance * math.sin(human_yaw)

        # Create synthetic Path with single standoff waypoint
        path_msg = Path()
        path_msg.header.frame_id = msg.header.frame_id if msg.header.frame_id else self.frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()

        pose_stamped = PoseStamped()
        pose_stamped.header = path_msg.header
        pose_stamped.pose.position.x = standoff_x
        pose_stamped.pose.position.y = standoff_y
        pose_stamped.pose.orientation.w = 1.0

        path_msg.poses = [pose_stamped]

        # Publish synthetic path
        self.path = [(standoff_x, standoff_y)]
        self.path_frame = path_msg.header.frame_id
        self.idx = 0
        self.logged_no_path = False
        self.logged_end_path = False

        # Publish markers
        self.publish_goal_marker(standoff_x, standoff_y)
        self.publish_standoff_marker(standoff_x, standoff_y)

        # Enable automation
        enable_msg = Bool()
        enable_msg.data = True
        self.pub_auto_enable.publish(enable_msg)

        self.get_logger().info(
            f'Human selected @ ({hx:.2f}, {hy:.2f}), '
            f'standing off to ({standoff_x:.2f}, {standoff_y:.2f})'
        )

    def cb_nav_goal(self, msg: PoseStamped):
        """
        Receive direct point goal and convert to Path (no standoff).
        If frame_id is 'base_link', treat (x,y) as relative to current robot pose.
        """
        gx = float(msg.pose.position.x)
        gy = float(msg.pose.position.y)

        if msg.header.frame_id == 'base_link' and self.have_pose:
            # Rotate delta from robot body frame to odom frame, then offset
            import math as _math
            c = _math.cos(self.yaw)
            s = _math.sin(self.yaw)
            gx_odom = self.x + c * gx - s * gy
            gy_odom = self.y + s * gx + c * gy
            gx, gy = gx_odom, gy_odom
        elif msg.header.frame_id == 'base_link':
            # No odom yet — treat as odom-frame delta from origin
            pass

        # Create synthetic Path with single waypoint
        path_msg = Path()
        path_msg.header.frame_id = msg.header.frame_id if msg.header.frame_id else self.frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()

        pose_stamped = PoseStamped()
        pose_stamped.header = path_msg.header
        pose_stamped.pose.position.x = gx
        pose_stamped.pose.position.y = gy
        pose_stamped.pose.orientation.w = 1.0

        path_msg.poses = [pose_stamped]

        # Publish synthetic path
        self.path = [(gx, gy)]
        self.path_frame = path_msg.header.frame_id
        self.idx = 0
        self.logged_no_path = False
        self.logged_end_path = False

        # Publish goal marker
        self.publish_goal_marker(gx, gy)

        # Enable automation
        enable_msg = Bool()
        enable_msg.data = True
        self.pub_auto_enable.publish(enable_msg)

        self.get_logger().info(f'Nav goal received @ ({gx:.2f}, {gy:.2f})')

    def publish_goal_marker(self, gx, gy):
        """Publish goal position marker (green sphere)."""
        m = Marker()
        m.header.frame_id = self.path_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'g1pilot_goal'
        m.id = 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = gx
        m.pose.position.y = gy
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.12
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.9
        self.pub_goal_marker.publish(m)

    def publish_wp_marker(self, wx, wy):
        """Publish current waypoint marker (orange sphere)."""
        m = Marker()
        m.header.frame_id = self.path_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'g1pilot_wp'
        m.id = 2
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = wx
        m.pose.position.y = wy
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.10
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.6, 0.0, 0.9
        self.pub_wp_marker.publish(m)

    def publish_standoff_marker(self, sx, sy):
        """Publish standoff point marker (cyan sphere)."""
        m = Marker()
        m.header.frame_id = self.path_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'g1_standoff'
        m.id = 3
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = sx
        m.pose.position.y = sy
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.15
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 1.0, 0.9
        self.pub_standoff_marker.publish(m)

    def loop(self):
        """Main control loop: compute joy commands to navigate waypoints."""
        try:
            if len(self.path) == 0:
                if not self.logged_no_path:
                    self.get_logger().warn('No path available.')
                    self.logged_no_path = True
                return
            if not self.have_pose and self.auto_enabled:
                if not self.logged_no_pose:
                    self.get_logger().warn('No pose available.')
                    self.logged_no_pose = True
                return
            if self.idx >= len(self.path) and self.auto_enabled:
                if not self.logged_end_path:
                    self.get_logger().warn('Reached the end of the path.')
                    self.logged_end_path = True
                return

            self.logged_no_pose = False
            self.logged_no_path = False
            self.logged_end_path = False

            wx, wy = self.path[self.idx]
            dx = wx - self.x
            dy = wy - self.y
            dist_wp = math.hypot(dx, dy)

            if self.idx < len(self.path) - 1 and dist_wp <= self.wp_tol:
                self.idx += 1
                wx, wy = self.path[self.idx]
                dx = wx - self.x
                dy = wy - self.y
                dist_wp = math.hypot(dx, dy)

            self.publish_wp_marker(wx, wy)

            dist_goal = math.hypot(self.path[-1][0] - self.x, self.path[-1][1] - self.y)

            joy = Joy()
            joy.header.stamp = self.get_clock().now().to_msg()
            axes = [0.0] * 8
            buttons = [0] * 14

            if dist_goal <= self.goal_tol:
                joy.axes = axes
                joy.buttons = buttons
                self.pub_joy.publish(joy)
                self.path = []
                return

            desired_yaw = math.atan2(dy, dx)
            yaw_err = desired_yaw - self.yaw
            while yaw_err > math.pi: yaw_err -= 2 * math.pi
            while yaw_err < -math.pi: yaw_err += 2 * math.pi

            vx_w = max(-self.vx_lim, min(self.vx_lim, self.pos_kp * dx))
            vy_w = max(-self.vy_lim, min(self.vy_lim, self.pos_kp * dy))
            c = math.cos(-self.yaw)
            s = math.sin(-self.yaw)
            vx_b = c * vx_w - s * vy_w
            vy_b = s * vx_w + c * vy_w
            wz = max(-self.wz_lim, min(self.wz_lim, self.yaw_kp * yaw_err))

            ax1 = max(-1.0, min(1.0, -vx_b / self.vx_lim))
            ax0 = max(-1.0, min(1.0, -vy_b / self.vy_lim))
            ax3 = max(-1.0, min(1.0, -wz / self.wz_lim))

            axes[1] = ax1
            axes[0] = ax0
            axes[2] = ax3
            buttons[8] = 1

            joy.axes = axes
            joy.buttons = buttons
            self.pub_joy.publish(joy)

        except Exception as e:
            self.get_logger().error(f'Error in loop: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = Nav2Point()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
