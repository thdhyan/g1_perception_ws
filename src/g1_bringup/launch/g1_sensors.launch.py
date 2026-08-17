#!/usr/bin/env python3
"""
Robot-side sensor & description launch file for Unitree G1.

Launches:
  - livox_ros_driver2: Mid-360 LiDAR driver (publishing /livox/lidar)
  - lowstate_to_jointstate: Convert robot joint state encoders to /joint_states
  - g1_description: Robot State Publisher (publishing /robot_description and robot TFs)

Does NOT launch:
  - SLAM or navigation
  - Laptop static TFs or fake joint state publishers
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_g1_desc = get_package_share_directory("g1_description")
    urdf_path = os.path.join(pkg_g1_desc, "urdf", "g1_29dof.urdf")
    with open(urdf_path, "r") as f:
        robot_desc = f.read()

    lidar_config_path = (
        "/home/unitree/Projects/ros2_ws/src/livox_ros_driver2/config/MID360_config.json"
    )

    lidar_enabled = LaunchConfiguration("lidar", default="true")
    joint_states_enabled = LaunchConfiguration("joint_states", default="true")

    livox_ros2_params = [{
        "xfer_format": 0,
        "multi_topic": 0,
        "data_src": 0,
        "publish_freq": 10.0,
        "output_data_type": 0,
        "frame_id": "mid360_link",
        "user_config_path": lidar_config_path,
    }]

    nodes = [
        # 1. Robot State Publisher (publishes /robot_description and kinematic TF tree on robot side)
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_desc, "use_sim_time": False}],
        ),
        # 2. Livox Mid-360 LiDAR driver (launches directly with frame_id=mid360_link)
        Node(
            package="livox_ros_driver2",
            executable="livox_ros_driver2_node",
            name="livox_lidar_publisher",
            output="screen",
            parameters=livox_ros2_params,
            condition=IfCondition(lidar_enabled),
        ),
        # 3. Unitree Low-State to JointState converter
        Node(
            package="lowstate_to_jointstate",
            executable="lowstate_to_jointstate_node",
            name="lowstate_to_jointstate_node",
            output="screen",
            condition=IfCondition(joint_states_enabled),
        ),
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            "lidar",
            default_value="true",
            description="Whether to launch Livox Mid-360 LiDAR driver",
        ),
        DeclareLaunchArgument(
            "joint_states",
            default_value="true",
            description="Whether to launch lowstate_to_jointstate converter",
        ),
        *nodes,
    ])
