#!/usr/bin/env python3
"""Launch the G1 WBC node (sim-agnostic).

Works with any sim/robot that provides:
  /joint_states (sensor_msgs/JointState)
  /imu/data     (sensor_msgs/Imu)
  /g1/cmd_vel   (geometry_msgs/Twist)

And publishes one position target per joint:
  /g1/joint/<joint_name> (std_msgs/Float64)  — 29 joints

For Gazebo: each is bridged in gz_bridge.yaml to /model/g1/<joint_name>
(gz.msgs.Double), consumed by a per-joint gz::sim::systems::JointPositionController.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Auto-locate policy files
# __file__ = .../src/g1_wbc/launch/wbc.launch.py  ->  parents[4] == .../thesis
_G1_SIM_BASE = Path(__file__).resolve().parents[4] / "G1_sim"
_POLICY_DIR = _G1_SIM_BASE / "assets" / "policy"
_DEFAULT_BALANCE = str(_POLICY_DIR / "GR00T-WholeBodyControl-Balance.onnx")
_DEFAULT_WALK = str(_POLICY_DIR / "GR00T-WholeBodyControl-Walk.onnx")


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "balance_policy_path",
            default_value=_DEFAULT_BALANCE,
            description="Path to GR00T Balance ONNX policy",
        ),
        DeclareLaunchArgument(
            "walk_policy_path",
            default_value=_DEFAULT_WALK,
            description="Path to GR00T Walk ONNX policy",
        ),
        DeclareLaunchArgument(
            "control_hz",
            default_value="50.0",
            description="WBC control rate in Hz",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulated clock",
        ),
        DeclareLaunchArgument(
            "joint_topic_prefix",
            default_value="/g1/joint",
            description="Per-joint target topic prefix (one Float64 on <prefix>/<joint_name>)",
        ),
        Node(
            package="g1_wbc",
            executable="wbc_node",
            name="g1_wbc_node",
            output="screen",
            parameters=[{
                "balance_policy_path": LaunchConfiguration("balance_policy_path"),
                "walk_policy_path": LaunchConfiguration("walk_policy_path"),
                "control_hz": LaunchConfiguration("control_hz"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "joint_topic_prefix": LaunchConfiguration("joint_topic_prefix"),
            }],
        ),
    ])
