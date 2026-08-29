#!/usr/bin/env python3
"""
Full G1 real-robot bringup — wires all packages into one launch.

PREREQUISITE: Start robot_bridge.py in a separate terminal FIRST:
  export PATH=/usr/bin:/bin:$PATH
  export CYCLONEDDS_HOME=/home/thakk100/cyclonedds/install
  cd src/g1_control/g1_control && python3 robot_bridge.py

Usage (real robot):
  ros2 launch g1_bringup full_real.launch.py slam:=true device:=cuda

Usage (sim):
  ros2 launch g1_bringup full_real.launch.py source:=sim use_sim_time:=true

Human selection: human_selector_node prints a numbered CLI menu when
detections arrive — type the number and Enter to select a target human.
The robot then walks to 60 cm in front of the selected human.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description wiring all G1 packages."""

    # ── Declare launch arguments ──────────────────────────────────────────────
    source_arg = DeclareLaunchArgument(
        "source",
        default_value="real",
        description='Sensor source: "real" (robot) or "sim" (Gazebo)',
    )

    slam_arg = DeclareLaunchArgument(
        "slam",
        default_value="true",
        description="If true, launch SLAM; if false, launch navigation with map",
    )

    map_arg = DeclareLaunchArgument(
        "map",
        default_value="",
        description="Path to Nav2 .yaml map file (used when slam:=false)",
    )

    device_arg = DeclareLaunchArgument(
        "device",
        default_value="cuda",
        description="Detection inference device (cuda or cpu)",
    )

    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint",
        default_value=os.path.expanduser("~/Projects/thesis/g1_perception_ws/pt/voxelnext_nuscenes.pth"),
        description="Path to VoxelNeXt checkpoint",
    )

    score_threshold_arg = DeclareLaunchArgument(
        "score_threshold",
        default_value="0.4",
        description="Detection confidence threshold [0, 1]",
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulated time (set true when source:=sim)",
    )

    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo in headless mode (when source:=sim)",
    )

    standoff_distance_arg = DeclareLaunchArgument(
        "standoff_distance",
        default_value="0.6",
        description="Standoff distance from human (meters)",
    )

    map_frame_arg = DeclareLaunchArgument(
        "map_frame",
        default_value="map",
        description="Map frame ID for nav goals",
    )

    socket_path_arg = DeclareLaunchArgument(
        "socket_path",
        default_value="/tmp/g1_robot_bridge.sock",
        description="Path to robot_bridge Unix socket",
    )

    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="false",
        description="Launch RViz2 visualization (when source:=sim)",
    )

    # ── Launch configurations ─────────────────────────────────────────────────
    source = LaunchConfiguration("source")
    slam = LaunchConfiguration("slam")
    map_path = LaunchConfiguration("map")
    device = LaunchConfiguration("device")
    checkpoint = LaunchConfiguration("checkpoint")
    score_threshold = LaunchConfiguration("score_threshold")
    use_sim_time = LaunchConfiguration("use_sim_time")
    headless = LaunchConfiguration("headless")
    standoff_distance = LaunchConfiguration("standoff_distance")
    map_frame = LaunchConfiguration("map_frame")
    socket_path = LaunchConfiguration("socket_path")
    rviz = LaunchConfiguration("rviz")

    # ── Launch descriptions ───────────────────────────────────────────────────

    # 1. Real sensors (source == "real")
    real_sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("g1_bringup"), "/launch/real.launch.py"]
        ),
        condition=IfCondition(
            PythonExpression(["'", source, "' == 'real'"])
        ),
    )

    # 2. Simulation bringup (source == "sim")
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("g1_bringup"), "/launch/sim.launch.py"]
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "headless": headless,
            "rviz": rviz,
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", source, "' == 'sim'"])
        ),
    )

    # 3. Livox detection
    livox_detection_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("livox_detection"), "/launch/livox_detection.launch.py"]
        ),
        launch_arguments={
            "checkpoint_path": checkpoint,
            "score_threshold": score_threshold,
            "device": device,
        }.items(),
    )

    # 4. Human selector node (from g1_perception)
    human_selector_node = Node(
        package="g1_perception",
        executable="human_selector_node",
        name="g1_human_selector",
        output="screen",
        parameters=[
            {"detection_topic": "/g1/detections/livox"},
            {"min_score": 0.3},
        ],
    )

    # 5. SLAM (slam == "true")
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("g1_nav"), "/launch/slam.launch.py"]
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
        condition=IfCondition(slam),
    )

    # 6. Navigation with map (slam == "false")
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("g1_nav"), "/launch/navigation.launch.py"]
        ),
        launch_arguments={
            "map": map_path,
            "use_sim_time": use_sim_time,
        }.items(),
        condition=UnlessCondition(slam),
    )

    # 7. G1 control (cmd_vel_bridge + human_follower)
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("g1_control"), "/launch/control.launch.py"]
        ),
        launch_arguments={
            "socket_path": socket_path,
            "standoff_distance": standoff_distance,
            "map_frame": map_frame,
        }.items(),
    )

    # ── Return launch description ─────────────────────────────────────────────
    return LaunchDescription(
        [
            # Arguments
            source_arg,
            slam_arg,
            map_arg,
            device_arg,
            checkpoint_arg,
            score_threshold_arg,
            use_sim_time_arg,
            headless_arg,
            standoff_distance_arg,
            map_frame_arg,
            socket_path_arg,
            rviz_arg,
            # Launches and nodes
            real_sensors_launch,
            sim_launch,
            livox_detection_launch,
            human_selector_node,
            slam_launch,
            navigation_launch,
            control_launch,
        ]
    )
