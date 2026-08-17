#!/usr/bin/env python3
"""G1 SDK locomotion + nav2point controller launch.

Mirrors the structure of github.com/thdhyan/g1pilot/launch/navigation_launcher.launch.py
but uses nodes ported into this workspace (g1_perception package).

Nodes launched:
  loco_client  — connects to robot via unitree_sdk2py, handles joystick +
                 emergency stop + balancing + gripper control.
  nav2point    — path-following controller: converts Nav2 paths / human poses
                 into joy commands → loco_client → robot SDK.
                 Includes 60 cm standoff affordance for human targets.

Prerequisites:
  export G1_INTERFACE=<your_ethernet_interface>   # e.g. eno2, eth0
  pip install unitree_sdk2py                       # Unitree's Python SDK

Launch args:
  interface        (default: $G1_INTERFACE env var)
  use_robot        (default: true)  — false = dry-run, no SDK connection
  arm_controlled   (default: both)  — left | right | both
  enable_arm_ui    (default: true)
  standoff_distance (default: 0.60) — metres to stop in front of human

Topic flow:
  /g1/selected_human  (PoseStamped) ──► nav2point ──► /g1pilot/auto_joy
  /g1pilot/path       (Path)        ──► nav2point ──► /g1pilot/auto_joy
  /g1pilot/auto_joy   (Joy)         ──► loco_client ──► unitree_sdk2py ──► robot
  /g1pilot/joy        (Joy)         ──► loco_client  (manual joystick override)
"""

import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    if not os.environ.get("G1_INTERFACE"):
        sys.exit(
            "ERROR: G1_INTERFACE environment variable is not set.\n"
            "Set it to your ethernet interface, e.g.:  export G1_INTERFACE=eno2"
        )

    interface = LaunchConfiguration("interface")
    use_robot = LaunchConfiguration("use_robot")
    arm_controlled = LaunchConfiguration("arm_controlled")
    enable_arm_ui = LaunchConfiguration("enable_arm_ui")
    standoff_distance = LaunchConfiguration("standoff_distance")

    return LaunchDescription([
        DeclareLaunchArgument(
            "interface",
            default_value=EnvironmentVariable("G1_INTERFACE"),
            description="Ethernet interface connected to G1 (e.g. eno2)",
        ),
        DeclareLaunchArgument(
            "use_robot",
            default_value="true",
            description="false = dry-run without SDK connection",
        ),
        DeclareLaunchArgument(
            "arm_controlled",
            default_value="both",
            description="Which arms to control: left | right | both",
        ),
        DeclareLaunchArgument(
            "enable_arm_ui",
            default_value="true",
            description="Enable arm joystick UI",
        ),
        DeclareLaunchArgument(
            "standoff_distance",
            default_value="0.60",
            description="Stop this many metres in front of the selected human (metres)",
        ),

        # loco_client: SDK bridge — joystick → robot locomotion
        Node(
            package="g1_perception",
            executable="loco_client",
            name="loco_client",
            parameters=[{
                "interface": interface,
                "use_robot": ParameterValue(use_robot, value_type=bool),
                "arm_controlled": arm_controlled,
                "enable_arm_ui": ParameterValue(enable_arm_ui, value_type=bool),
            }],
            output="screen",
        ),

        # nav2point: path follower with 60 cm human standoff
        Node(
            package="g1_perception",
            executable="nav2point",
            name="nav2point",
            parameters=[{
                "interface": interface,
                "use_robot": ParameterValue(use_robot, value_type=bool),
                "standoff_distance": ParameterValue(standoff_distance, value_type=float),
                "joy_topic": "/g1pilot/auto_joy",
                "path_topic": "/g1pilot/path",
                "auto_enable_topic": "/g1pilot/auto_enable",
            }],
            output="screen",
        ),
    ])
