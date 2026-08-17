#!/usr/bin/env python3
"""
SLAM Mapping with G1 Mid-360 LiDAR: PointCloud2 -> LaserScan -> SLAM pipeline.

Launches an async SLAM system for real-time occupancy grid mapping on Unitree G1.

Nodes launched:
  1. lidar_bridge                    - republish LiDAR to /livox/mid360/points
  2. pointcloud_to_laserscan_node    - convert PointCloud2 to LaserScan (/scan)
  3. async_slam_toolbox_node         - SLAM (slam_toolbox async alogrithm)

Pre-requisites:
  sudo apt install ros-humble-slam-toolbox ros-humble-pointcloud-to-laserscan

SLAM Behavior:
  - Subscribes to /scan (LaserScan) from pointcloud_to_laserscan_node
  - Publishes occupancy grid to /map
  - Publishes TF: map -> odom (SLAM corrects odometry drift)
  - The odometry source (odom -> base_link) must be published by unitree_ros2 driver

Saving the Map:
  After running this launch file and exploring the environment:
    ros2 run nav2_map_server map_saver_cli -f ~/maps/my_room
  This saves two files:
    ~/maps/my_room.pgm   (occupancy grid image)
    ~/maps/my_room.yaml  (metadata: resolution, origin, threshold values)

Using Saved Map with Nav2:
  ros2 launch nav2_bringup navigation_launch.py \
    map:=~/maps/my_room.yaml \
    use_sim_time:=false

G1 Odometry TF:
  The unitree_ros2 driver publishes odometry as TF tree:
    odom -> base_link (robot base, located at G1's center)
  The SLAM node will automatically generate:
    map -> odom (SLAM-corrected frame)
  Together they form:
    map -> odom -> base_link
  This allows Nav2 and other tools to know the robot's position in the global map.

Args:
  source                (default: real)      - "real" or "sim" for lidar_bridge
  slam_params_file      (default: auto)      - path to slam_toolbox YAML config
  pc2scan_params_file   (default: auto)      - path to pointcloud_to_laserscan YAML config
  use_sim_time          (default: false)     - true for rosbag playback, false for real-time
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ── declare args ──────────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument(
            "source",
            default_value="real",
            description="LiDAR source: 'real' or 'sim'"
        ),
        DeclareLaunchArgument(
            "slam_params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("g1_perception"),
                "config",
                "slam_toolbox_params.yaml"
            ]),
            description="Path to slam_toolbox parameters YAML"
        ),
        DeclareLaunchArgument(
            "pc2scan_params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("g1_perception"),
                "config",
                "pc2scan_params.yaml"
            ]),
            description="Path to pointcloud_to_laserscan parameters YAML"
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation time (true for rosbag, false for real-time)"
        ),
    ]

    # ── substitutions ─────────────────────────────────────────────────────────
    source = LaunchConfiguration("source")
    slam_params_file = LaunchConfiguration("slam_params_file")
    pc2scan_params_file = LaunchConfiguration("pc2scan_params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # ── nodes ─────────────────────────────────────────────────────────────────

    # 1. Republish G1 Mid-360 LiDAR as PointCloud2
    lidar_bridge = Node(
        package="g1_perception",
        executable="lidar_bridge",
        name="lidar_bridge",
        parameters=[{"source": source}],
        output="screen",
    )

    # 2. Convert PointCloud2 to LaserScan
    #    Subscribes: /livox/mid360/points
    #    Publishes: /scan
    pointcloud_to_laserscan = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        remappings=[("cloud_in", "/livox/mid360/points")],
        parameters=[pc2scan_params_file],
        output="screen",
    )

    # 3. SLAM: async_slam_toolbox_node
    #    Subscribes: /scan (LaserScan)
    #    Publishes: /map (nav_msgs/OccupancyGrid)
    #    TF output: map -> odom
    slam_toolbox = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        parameters=[
            slam_params_file,
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )

    return LaunchDescription(args + [
        lidar_bridge,
        pointcloud_to_laserscan,
        slam_toolbox,
    ])
