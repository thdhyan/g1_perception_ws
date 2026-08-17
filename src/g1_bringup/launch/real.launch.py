#!/usr/bin/env python3
"""
Real robot sensor launch for Unitree G1.

WARNING: This launch file is intended to run ON THE ROBOT (Foxy), not the laptop (Jazzy).
The robot runs ros2_ws with Foxy. When porting to the robot, update syntax as needed.

Launches:
  - livox_ros_driver2: Mid-360 LiDAR driver
  - realsense2_camera: D435 RGBD camera
  - lowstate_to_jointstate: Convert robot joint states
  - g1_description: TF broadcaster for kinematic tree
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for real robot."""

    # Declare launch arguments
    lidar_enabled = LaunchConfiguration("lidar", default="true")
    camera_enabled = LaunchConfiguration("camera", default="true")
    joint_states_enabled = LaunchConfiguration("joint_states", default="true")

    nodes = []

    # Livox Mid-360 LiDAR driver
    # Config file location: /home/unitree/Projects/ros2_ws/src/livox_ros_driver2/config/MID360_config.json
    # NOTE: On robot, adjust config path to actual robot filesystem
    lidar_config_path = (
        "/home/unitree/Projects/ros2_ws/src/livox_ros_driver2/config/MID360_config.json"
    )
    nodes.append(
        Node(
            package="livox_ros_driver2",
            executable="livox_ros_driver2_node",
            output="screen",
            parameters=[
                {"config_file_path": lidar_config_path},
            ],
            condition="IfCondition" in dir() and IfCondition(lidar_enabled),
        )
    )

    # RealSense D435 camera via realsense2_camera
    # Uses rs_launch.py from realsense2_camera package
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        launch_arguments=[
            ("rgb_camera.profile", "640x480x30"),
            ("depth_module.profile", "640x480x30"),
            ("pointcloud.enable", "true"),
        ],
        condition="IfCondition" in dir() and IfCondition(camera_enabled),
    )

    # lowstate_to_jointstate: convert Unitree robot state to JointState
    nodes.append(
        Node(
            package="lowstate_to_jointstate",
            executable="lowstate_to_jointstate_node",
            output="screen",
            condition="IfCondition" in dir() and IfCondition(joint_states_enabled),
        )
    )

    # g1_description: broadcast kinematic tree (TF and robot_description)
    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("g1_description"), "launch", "description.launch.py"]
            )
        ),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "lidar",
            default_value="true",
            description="Launch Livox Mid-360 LiDAR driver",
        ),
        DeclareLaunchArgument(
            "camera",
            default_value="true",
            description="Launch RealSense D435 camera",
        ),
        DeclareLaunchArgument(
            "joint_states",
            default_value="true",
            description="Launch lowstate_to_jointstate converter",
        ),
        *nodes,
        realsense_launch,
        description_launch,
    ])
