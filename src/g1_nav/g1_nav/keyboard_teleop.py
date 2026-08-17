#!/usr/bin/env python3
"""Interactive keyboard teleoperation node for G1 humanoid in simulation / real robot.

Controls:
  W / S : Forward / Backward (+/- vx)
  A / D : Strafe Left / Right (+/- vy)
  Q / E : Turn Left / Right (+/- wz)
  SPACE : Stop / Hold Position
  Z / C : Decrease / Increase linear speed step
  U / O : Decrease / Increase angular speed step
  H     : Toggle Stand / Base Height command
  ESC   : Quit
"""

import os
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64

BANNER = """
===========================================================
  G1 Humanoid Keyboard Teleop (WASDQE)
===========================================================
  Movement:
      [W]  Forward              [Q]  Turn Left
  [A]     [D] Strafe Left/Right      [E]  Turn Right
      [S]  Backward

  [SPACE]  : Active Brake / Hold Position (0.0 vel)
  [Z] / [C]: Adjust Linear Speed  (-/+ 0.05 m/s)
  [U] / [O]: Adjust Turn Rate     (-/+ 0.10 rad/s)
  [H]      : Reset Stand Height   (0.74 m)
  [CTRL+C] : Quit
===========================================================
"""


class G1TeleopKeyboard(Node):
    def __init__(self):
        super().__init__("g1_teleop_keyboard")

        self.declare_parameter("cmd_vel_topic", "/g1/cmd_vel")
        self.declare_parameter("linear_speed", 0.30)
        self.declare_parameter("strafe_speed", 0.20)
        self.declare_parameter("yaw_rate", 0.50)
        self.declare_parameter("publish_rate", 20.0)

        cmd_topic = self.get_parameter("cmd_vel_topic").value
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.strafe_speed = float(self.get_parameter("strafe_speed").value)
        self.yaw_rate = float(self.get_parameter("yaw_rate").value)
        rate = float(self.get_parameter("publish_rate").value)

        self.pub_cmd_vel = self.create_publisher(Twist, cmd_topic, 10)
        self.pub_dual_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_height = self.create_publisher(Float64, "/g1/base_height", 10)

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

        self.timer = self.create_timer(1.0 / rate, self._timer_callback)
        self.get_logger().info(f"Teleop publishing on {cmd_topic} and /cmd_vel @ {rate} Hz")

    def _timer_callback(self):
        msg = Twist()
        msg.linear.x = float(self.vx)
        msg.linear.y = float(self.vy)
        msg.angular.z = float(self.wz)
        self.pub_cmd_vel.publish(msg)
        self.pub_dual_cmd.publish(msg)

    def set_vel(self, vx: float, vy: float, wz: float):
        self.vx = vx
        self.vy = vy
        self.wz = wz

    def stop(self):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main():
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = G1TeleopKeyboard()

    print(BANNER)
    print(f"Current: Lin={node.linear_speed:.2f} m/s | Strafe={node.strafe_speed:.2f} m/s | Yaw={node.yaw_rate:.2f} rad/s\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            key = get_key(settings)

            if not key:
                continue

            k = key.lower()
            if k == "w":
                node.set_vel(node.linear_speed, 0.0, 0.0)
                print(f"\r[CMD] Forward  (vx={node.vx:+.2f}, vy={node.vy:+.2f}, wz={node.wz:+.2f})", end="", flush=True)
            elif k == "s":
                node.set_vel(-node.linear_speed, 0.0, 0.0)
                print(f"\r[CMD] Backward (vx={node.vx:+.2f}, vy={node.vy:+.2f}, wz={node.wz:+.2f})", end="", flush=True)
            elif k == "a":
                node.set_vel(0.0, node.strafe_speed, 0.0)
                print(f"\r[CMD] Strafe L (vx={node.vx:+.2f}, vy={node.vy:+.2f}, wz={node.wz:+.2f})", end="", flush=True)
            elif k == "d":
                node.set_vel(0.0, -node.strafe_speed, 0.0)
                print(f"\r[CMD] Strafe R (vx={node.vx:+.2f}, vy={node.vy:+.2f}, wz={node.wz:+.2f})", end="", flush=True)
            elif k == "q":
                node.set_vel(0.0, 0.0, node.yaw_rate)
                print(f"\r[CMD] Turn L   (vx={node.vx:+.2f}, vy={node.vy:+.2f}, wz={node.wz:+.2f})", end="", flush=True)
            elif k == "e":
                node.set_vel(0.0, 0.0, -node.yaw_rate)
                print(f"\r[CMD] Turn R   (vx={node.vx:+.2f}, vy={node.vy:+.2f}, wz={node.wz:+.2f})", end="", flush=True)
            elif k == " ":
                node.stop()
                print("\r[CMD] STOP / HOLD POSITION (0.0 m/s)                             ", end="", flush=True)
            elif k == "c":
                node.linear_speed = min(0.8, node.linear_speed + 0.05)
                node.strafe_speed = min(0.5, node.strafe_speed + 0.05)
                print(f"\r[SPEED] Lin={node.linear_speed:.2f} m/s | Strafe={node.strafe_speed:.2f} m/s          ", end="", flush=True)
            elif k == "z":
                node.linear_speed = max(0.05, node.linear_speed - 0.05)
                node.strafe_speed = max(0.05, node.strafe_speed - 0.05)
                print(f"\r[SPEED] Lin={node.linear_speed:.2f} m/s | Strafe={node.strafe_speed:.2f} m/s          ", end="", flush=True)
            elif k == "o":
                node.yaw_rate = min(1.2, node.yaw_rate + 0.10)
                print(f"\r[SPEED] Yaw={node.yaw_rate:.2f} rad/s                                    ", end="", flush=True)
            elif k == "u":
                node.yaw_rate = max(0.1, node.yaw_rate - 0.10)
                print(f"\r[SPEED] Yaw={node.yaw_rate:.2f} rad/s                                    ", end="", flush=True)
            elif k == "h":
                h_msg = Float64()
                h_msg.data = 0.74
                node.pub_height.publish(h_msg)
                print("\r[CMD] Reset Stance Height (0.74m)                                ", end="", flush=True)
            elif key == "\x03" or key == "\x1b":  # Ctrl+C or ESC
                break

    except Exception as e:
        print(f"\nTeleop error: {e}")
    finally:
        node.stop()
        msg = Twist()
        node.pub_cmd_vel.publish(msg)
        node.pub_dual_cmd.publish(msg)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()
        print("\nTeleop stopped.")


if __name__ == "__main__":
    main()
