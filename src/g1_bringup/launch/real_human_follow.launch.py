#!/usr/bin/env python3
"""Unified Real Robot Human Detection, Follow & Greet Launch File.

Launches the complete laptop-side perception, 3D detection, human selection,
and locomotion/greeting pipeline for the Unitree G1 humanoid on a real robot.

Supported 3D Detection Backends:
  - 'voxelnext'   : Fully-sparse anchor-free 3D detection (best for diverse human shapes)
  - 'centerpoint' : Anchor-free CenterPoint model (livox_model_1.pt)
  - 'pointpillar' : Voxelization + Pillar SSD detector

Prerequisites:
  1. Real robot sensors running:
       ./scripts/start_robot_sensors.sh
  2. Locomotion bridge running (separate terminal):
       source setup_g1_env.sh && python3 src/g1_control/g1_control/robot_bridge.py

Usage:
  # 1. Run with VoxelNeXt (default):
  ros2 launch g1_bringup real_human_follow.launch.py algorithm:=voxelnext

  # 2. Run with CenterPoint:
  ros2 launch g1_bringup real_human_follow.launch.py algorithm:=centerpoint

  # 3. Run with PointPillars:
  ros2 launch g1_bringup real_human_follow.launch.py algorithm:=pointpillar
"""

import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_livox_det = get_package_share_directory("livox_detection")
    pkg_g1_desc = get_package_share_directory("g1_description")
    
    default_rviz_config = os.path.join(pkg_livox_det, "config", "livox_snapshot_viz.rviz")
    urdf_path = os.path.join(pkg_g1_desc, "urdf", "g1_29dof.urdf")
    with open(urdf_path, "r") as f:
        robot_desc = f.read()

    # Workspace root resolution for VoxelNeXt config/model paths
    ws_root = Path(__file__).resolve().parents[4]
    voxelnext_cfg_default = str(
        ws_root / "VoxelNeXt" / "tools" / "cfgs" / "nuscenes_models" / "cbgs_voxel0075_voxelnext.yaml"
    )
    voxelnext_dir_default = str(ws_root / "VoxelNeXt")
    voxelnext_ckpt_default = str(ws_root / "pt" / "voxelnext_nuscenes.pth")
    centerpoint_ckpt_default = "/home/thakk100/Projects/Thesis/livox_detection/pt/livox_model_1.pt"

    return LaunchDescription([
        # ── Launch Arguments ──────────────────────────────────────────────────
        DeclareLaunchArgument(
            "algorithm",
            default_value="voxelnext",
            description="Detection algorithm backend: 'voxelnext', 'centerpoint', or 'pointpillar'",
        ),
        DeclareLaunchArgument(
            "checkpoint_path",
            default_value=voxelnext_ckpt_default,
            description="Path to model checkpoint (.pth or .pt file)",
        ),
        DeclareLaunchArgument(
            "voxelnext_cfg",
            default_value=voxelnext_cfg_default,
            description="Path to VoxelNeXt YAML configuration",
        ),
        DeclareLaunchArgument(
            "voxelnext_dir",
            default_value=voxelnext_dir_default,
            description="Path to VoxelNeXt source repository directory",
        ),
        DeclareLaunchArgument(
            "score_threshold",
            default_value="0.15",
            description="Confidence threshold for 3D human detection",
        ),
        DeclareLaunchArgument(
            "collect_frames",
            default_value="3",
            description="Number of Livox LiDAR frames to accumulate in Pass 1 snapshot",
        ),
        DeclareLaunchArgument(
            "collect_duration_sec",
            default_value="0.1",
            description="Duration in seconds to accumulate point clouds in Pass 1",
        ),
        DeclareLaunchArgument(
            "input_topic",
            default_value="/livox/lidar",
            description="Input LiDAR topic from real robot (PointCloud2 or CustomMsg)",
        ),
        DeclareLaunchArgument(
            "target_frame",
            default_value="pelvis",
            description="Target coordinate frame for human detections (pelvis root)",
        ),
        DeclareLaunchArgument(
            "standoff_distance",
            default_value="0.80",
            description="Standoff distance in front of target human in meters (default 0.80m)",
        ),
        DeclareLaunchArgument(
            "greeting_action",
            default_value="shake_hand",
            description=("Gesture on arrival: 'shake_hand', 'low_wave', 'high_wave', "
                         "'wave_and_shake', 'none', or any G1ArmActionClient action name "
                         "(clap, hug, heart, right_heart, high_five, hands_up, reject, "
                         "right_hand_up, x-ray, two-hand_kiss, left_kiss, right_kiss)"),
        ),
        DeclareLaunchArgument(
            "linear_speed",
            default_value="0.20",
            description="Approach walking speed in m/s (safe default 0.20 m/s)",
        ),
        DeclareLaunchArgument(
            "auto_execute",
            default_value="true",
            description="Whether to auto-execute approach locomotion upon target human selection",
        ),
        DeclareLaunchArgument(
            "auto_greet",
            default_value="true",
            description="Whether to auto-execute greeting gesture upon arrival at standoff",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="true",
            description="Whether to launch RViz visualization",
        ),
        DeclareLaunchArgument(
            "joint_state_publisher",
            default_value="true",
            description="Whether to launch dummy joint_state_publisher when robot /joint_states is offline",
        ),

        # ── 1. Robot State Publisher & TF ─────────────────────────────────────
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_desc, "use_sim_time": False}],
        ),
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            name="joint_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_desc,
                "use_sim_time": False,
                "rate": 30,
            }],
            condition=IfCondition(LaunchConfiguration("joint_state_publisher")),
        ),

        # ── 1b. Sensor frame binding ─────────────────────────────────────────
        # The Livox driver publishes clouds in 'livox_frame'; the URDF chain
        # ends at 'mid360_link'. Without this identity link the tree is broken
        # between the two and detections cannot be transformed into 'pelvis' --
        # inference still runs, so the symptom is "detections but no TF".
        # sim.launch.py has always published this; the real-robot launch did
        # not. Publish it here rather than on the robot: the robot is on Foxy
        # and the laptop on Jazzy, and TF/String messages do not survive that
        # CDR-encoding gap (see the serdata.cpp:308 errors).
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_mid360_to_livox",
            arguments=["--frame-id", "mid360_link", "--child-frame-id", "livox_frame"],
            parameters=[{"use_sim_time": False}],
            output="log",
        ),

        # ── 2. Snapshot 3D Detection Pipeline Node ────────────────────────────
        Node(
            package="livox_detection",
            executable="livox_snapshot_pipeline_node",
            name="livox_snapshot_pipeline_node",
            output="screen",
            parameters=[{
                "algorithm": LaunchConfiguration("algorithm"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "voxelnext_cfg": LaunchConfiguration("voxelnext_cfg"),
                "voxelnext_dir": LaunchConfiguration("voxelnext_dir"),
                "input_topic": LaunchConfiguration("input_topic"),
                "target_frame": LaunchConfiguration("target_frame"),
                "score_threshold": LaunchConfiguration("score_threshold"),
                "collect_frames": LaunchConfiguration("collect_frames"),
                "collect_duration_sec": LaunchConfiguration("collect_duration_sec"),
                "auto_start": True,
                "offset_ground": 1.33,
                "enable_cli_input": False,
            }],
        ),

        # ── 3. Human Follow & Greeting Controller ────────────────────────────
        Node(
            package="g1_arm_control",
            executable="human_follow_and_greet_node",
            name="human_follow_and_greet_node",
            output="screen",
            parameters=[{
                "standoff_distance": LaunchConfiguration("standoff_distance"),
                "greeting_action": LaunchConfiguration("greeting_action"),
                "linear_speed": LaunchConfiguration("linear_speed"),
                "auto_execute": LaunchConfiguration("auto_execute"),
                "auto_greet": LaunchConfiguration("auto_greet"),
            }],
        ),

        # ── 4. RViz Visualization ─────────────────────────────────────────────
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_real_human_follow",
            arguments=["-d", default_rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
