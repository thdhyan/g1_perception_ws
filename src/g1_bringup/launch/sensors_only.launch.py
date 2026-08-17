#!/usr/bin/env python3
"""
Lightweight sensor-only launch for testing LiDAR + camera without full stack.

Useful for:
  - Recording rosbags with sensor data only
  - Testing individual sensor drivers
  - Debugging perception algorithms without motor/localization nodes

Launches:
  - livox_ros_driver2: Mid-360 LiDAR
  - realsense2_camera: D435 RGBD camera

Does NOT launch:
  - TF tree, robot_state_publisher, or kinematic state
  - Robot control or motor interfaces
  - Navigation or localization
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for sensor-only mode."""

    lidar_enabled = LaunchConfiguration("lidar", default="true")
    camera_enabled = LaunchConfiguration("camera", default="true")

    nodes = []

    pkg_g1_desc = get_package_share_directory("g1_description")
    urdf_path = os.path.join(pkg_g1_desc, "urdf", "g1_29dof.urdf")
    with open(urdf_path, "r") as f:
        robot_desc = f.read()

    # Robot State Publisher (publishes /robot_description and URDF kinematic model)
    nodes.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_desc, "use_sim_time": False}],
        )
    )

    # Livox Mid-360 LiDAR driver
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
        )
    )

    # RealSense D435 camera
    camera_launch = IncludeLaunchDescription(
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
        *nodes,
        camera_launch,
    ])
