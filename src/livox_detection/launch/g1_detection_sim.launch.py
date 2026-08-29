#!/usr/bin/env python3
"""Launch VoxelNeXt 3D Human Detection for G1 Simulation."""

import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    ws_base = Path(__file__).resolve().parents[4]
    default_ckpt = str(ws_base / "G1_sim" / "detection" / "pt" / "livox_model_1.pt")

    return LaunchDescription([
        DeclareLaunchArgument(
            "algorithm",
            default_value="voxelnext",
            description="Detection algorithm (only 'voxelnext' is supported)",
        ),
        DeclareLaunchArgument(
            "checkpoint_path",
            default_value=default_ckpt,
            description="Path to VoxelNeXt checkpoint (.pth)",
        ),
        DeclareLaunchArgument(
            "input_topic",
            default_value="/livox/mid360/points",
            description="Input PointCloud2 topic",
        ),
        DeclareLaunchArgument(
            "target_frame",
            default_value="pelvis",
            description="Coordinate frame to publish bounding boxes in",
        ),
        DeclareLaunchArgument(
            "score_threshold",
            default_value="0.15",
            description="Detection score threshold",
        ),
        DeclareLaunchArgument(
            "accumulate_frames",
            default_value="4",
            description="Sweeps to accumulate for dense solid-state LiDAR point clouds",
        ),
        DeclareLaunchArgument(
            "max_hz",
            default_value="5.0",
            description="Inference rate in Hz",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cuda",
            description="Torch inference device ('cuda' or 'cpu')",
        ),
        Node(
            package="livox_detection",
            executable="livox_detection_node",
            name="g1_livox_detection",
            output="screen",
            parameters=[{
                "algorithm": LaunchConfiguration("algorithm"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "input_topic": LaunchConfiguration("input_topic"),
                "target_frame": LaunchConfiguration("target_frame"),
                "score_threshold": LaunchConfiguration("score_threshold"),
                "accumulate_frames": LaunchConfiguration("accumulate_frames"),
                "max_hz": LaunchConfiguration("max_hz"),
                "device": LaunchConfiguration("device"),
            }],
        ),
    ])
