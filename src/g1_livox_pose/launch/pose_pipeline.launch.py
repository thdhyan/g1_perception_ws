#!/usr/bin/env python3
"""Launch the g1_livox_pose pipeline: pose estimation + sequence assembly."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    backend = LaunchConfiguration("backend")
    cloud_topic = LaunchConfiguration("cloud_topic")
    detections_topic = LaunchConfiguration("detections_topic")
    target_frame = LaunchConfiguration("target_frame")
    sequence_frame = LaunchConfiguration("sequence_frame")
    log_path = LaunchConfiguration("log_path")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("backend", default_value="debug"),
        DeclareLaunchArgument("cloud_topic", default_value="/livox/collected_points"),
        DeclareLaunchArgument(
            "detections_topic", default_value="/g1/sorted_humans"
        ),
        DeclareLaunchArgument("target_frame", default_value="pelvis"),
        DeclareLaunchArgument("sequence_frame", default_value="odom"),
        DeclareLaunchArgument("log_path", default_value=""),

        Node(
            package="g1_livox_pose",
            executable="human_pose_node",
            name="g1_human_pose",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "backend": backend,
                "input_cloud_topic": cloud_topic,
                "input_detections_topic": detections_topic,
                "target_frame": target_frame,
            }],
        ),

        Node(
            package="g1_livox_pose",
            executable="pose_sequence_assembler_node",
            name="g1_pose_sequence_assembler",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "sequence_frame": sequence_frame,
                "log_path": log_path,
            }],
        ),
    ])
