#!/usr/bin/env python3
"""Full G1 perception + navigation stack launch.

Launches:
  1. lidar_bridge        - republish LiDAR to /livox/mid360/points
  2. livox_detection_node - native Livox CustomMsg -> CenterPoint detections
  3. centerpoint_node    - PointCloud2 path (fallback / dual-mode)
  4. human_selector_node - detect humans, CLI prompt, publish selected pose
  5. nav_goal_node       - send Nav2 NavigateToPose goals

Args:
  source          (default: real)  - "real" or "sim" for lidar_bridge
  use_livox_native (default: true) - use livox_detection_node (CustomMsg path)
                                     set false to use centerpoint_node (PC2 path)
  device          (default: cuda)
  checkpoint      (default: pt/livox_model_1.pt)
  score_threshold (default: 0.4)
  map_frame       (default: map)

Nav2 must be launched separately:
  ros2 launch nav2_bringup navigation_launch.py use_sim_time:=false map:=<map.yaml>
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── args ──────────────────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("source", default_value="real",
                              description="LiDAR source: real or sim"),
        DeclareLaunchArgument("use_livox_native", default_value="true",
                              description="true=CustomMsg path, false=PC2 path"),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("checkpoint", default_value="pt/livox_model_1.pt"),
        DeclareLaunchArgument("score_threshold", default_value="0.4"),
        DeclareLaunchArgument("map_frame", default_value="map"),
        DeclareLaunchArgument("detection_topic", default_value="/g1/detections/centerpoint"),
    ]

    source = LaunchConfiguration("source")
    use_livox_native = LaunchConfiguration("use_livox_native")
    device = LaunchConfiguration("device")
    checkpoint = LaunchConfiguration("checkpoint")
    score_threshold = LaunchConfiguration("score_threshold")
    map_frame = LaunchConfiguration("map_frame")
    detection_topic = LaunchConfiguration("detection_topic")

    # ── nodes ─────────────────────────────────────────────────────────────────
    lidar_bridge = Node(
        package="g1_perception",
        executable="lidar_bridge",
        name="g1_lidar_bridge",
        parameters=[{"source": source}],
        output="screen",
    )

    # Native Livox CustomMsg path (livox_ros_driver2 required)
    livox_detection = Node(
        package="g1_perception",
        executable="livox_detection_node",
        name="g1_livox_detection",
        parameters=[{
            "checkpoint_path": checkpoint,
            "device": device,
            "score_threshold": score_threshold,
        }],
        output="screen",
        condition=IfCondition(use_livox_native),
    )

    # PC2 path (no livox_ros_driver2 needed)
    centerpoint = Node(
        package="g1_perception",
        executable="centerpoint_node",
        name="g1_centerpoint_detection",
        parameters=[{
            "checkpoint_path": checkpoint,
            "device": device,
            "score_threshold": score_threshold,
        }],
        output="screen",
        condition=UnlessCondition(use_livox_native),
    )

    human_selector = Node(
        package="g1_perception",
        executable="human_selector_node",
        name="g1_human_selector",
        parameters=[{
            "detection_topic": detection_topic,
            "min_score": score_threshold,
        }],
        output="screen",
    )

    nav_goal = Node(
        package="g1_perception",
        executable="nav_goal_node",
        name="g1_nav_goal",
        parameters=[{"map_frame": map_frame}],
        output="screen",
    )

    return LaunchDescription(args + [
        lidar_bridge,
        livox_detection,
        centerpoint,
        human_selector,
        nav_goal,
    ])
