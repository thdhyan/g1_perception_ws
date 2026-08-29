#!/usr/bin/env python3
"""Full G1 perception + navigation stack launch.

Launches:
  1. lidar_bridge        - republish LiDAR to /livox/mid360/points
  2. livox_detection_node - VoxelNeXt 3D detection (livox_detection package)
  3. human_selector_node - detect humans, CLI prompt, publish selected pose
  4. nav_goal_node       - send Nav2 NavigateToPose goals

Args:
  source          (default: real)  - "real" or "sim" for lidar_bridge
  use_livox_native (default: true) - kept for legacy invocations; ignored
                                     (detection always runs via livox_detection)
  device          (default: cuda)
  checkpoint      (default: pt/voxelnext_nuscenes.pth)
  score_threshold (default: 0.4)
  map_frame       (default: map)

Nav2 must be launched separately:
  ros2 launch nav2_bringup navigation_launch.py use_sim_time:=false map:=<map.yaml>
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── args ──────────────────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument("source", default_value="real",
                              description="LiDAR source: real or sim"),
        DeclareLaunchArgument("use_livox_native", default_value="true",
                              description="legacy flag — ignored (detection "
                                          "always runs via the livox_detection package)"),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("checkpoint",
                              default_value=os.path.expanduser(
                                  "~/Projects/thesis/g1_perception_ws/pt/voxelnext_nuscenes.pth")),
        DeclareLaunchArgument("score_threshold", default_value="0.4"),
        DeclareLaunchArgument("map_frame", default_value="map"),
        DeclareLaunchArgument("detection_topic", default_value="/g1/detections/livox"),
    ]

    source = LaunchConfiguration("source")
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

    # VoxelNeXt detection (PointCloud2 + Optional CustomMsg handled by the node)
    detection = Node(
        package="livox_detection",
        executable="livox_detection_node",
        name="g1_livox_detection",
        parameters=[{
            "algorithm": "voxelnext",
            "checkpoint_path": checkpoint,
            "device": device,
            "score_threshold": score_threshold,
            "input_topic": "/livox/mid360/points",
            "class_filter": "pedestrian",
            "target_frame": "pelvis",
        }],
        output="screen",
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
        detection,
        human_selector,
        nav_goal,
    ])
