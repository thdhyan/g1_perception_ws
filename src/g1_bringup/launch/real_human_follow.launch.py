#!/usr/bin/env python3
"""Unified Real Robot Human Detection, Follow & Greet Launch File.

Launches the complete laptop-side perception, 3D detection, human selection,
and locomotion/greeting pipeline for the Unitree G1 humanoid on a real robot.

3D detection backend: 'voxelnext' — fully-sparse anchor-free 3D
detection (only supported backend).

Prerequisites:
  1. Real robot sensors running:
       ./scripts/start_robot_sensors.sh
  2. Locomotion bridge running (separate terminal):
       source setup_g1_env.sh && python3 src/g1_control/g1_control/robot_bridge.py

Usage:
  # 1. Run with VoxelNeXt (default):
  ros2 launch g1_bringup real_human_follow.launch.py algorithm:=voxelnext

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
    # Walk up to the workspace root rather than counting parents: this file is
    # reached through install/ or, under --symlink-install, through src/, and the
    # two are at different depths. A fixed parents[N] silently lands outside the
    # workspace, pcdet then fails to import, and the detector loads with an
    # error (node publishes empty detections).
    ws_root = next(
        (p for p in Path(__file__).resolve().parents if (p / "VoxelNeXt").is_dir()),
        Path(__file__).resolve().parents[4],
    )
    voxelnext_cfg_default = str(
        ws_root / "VoxelNeXt" / "tools" / "cfgs" / "nuscenes_models" / "cbgs_voxel0075_voxelnext.yaml"
    )
    voxelnext_dir_default = str(ws_root / "VoxelNeXt")
    voxelnext_ckpt_default = str(ws_root / "pt" / "voxelnext_nuscenes.pth")

    return LaunchDescription([
        # ── Launch Arguments ──────────────────────────────────────────────────
        DeclareLaunchArgument(
            "algorithm",
            default_value="voxelnext",
            description="Detection algorithm backend (only 'voxelnext' is supported)",
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
            "offset_ground",
            default_value="-0.3",
            description=("Z-shift (m) applied before inference and undone on the output "
                         "boxes -- whatever puts the ground where the nuScenes-trained "
                         "model expects it, not the sensor height. Measured on the real "
                         "robot (ground at z=-1.27 in mid360_link): a person at 4.3m "
                         "scores 0.28 at -0.3 and only 0.14 at the old sim value of 1.33"),
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
            default_value="1.50",
            description="Where to stop in front of the target human, in metres",
        ),
        DeclareLaunchArgument(
            "greeting_action",
            default_value="low_wave",
            description=("Gesture on arrival: 'shake_hand', 'low_wave', 'high_wave', "
                         "'wave_and_shake', 'none', or any G1ArmActionClient action name "
                         "(clap, hug, heart, right_heart, high_five, hands_up, reject, "
                         "right_hand_up, x-ray, two-hand_kiss, left_kiss, right_kiss)"),
        ),
        DeclareLaunchArgument(
            "linear_speed",
            default_value="0.90",
            description="Approach walking speed in m/s, used for the whole walk-up",
        ),
        DeclareLaunchArgument(
            "yaw_rate",
            default_value="0.25",
            description="Turning speed in rad/s during approach rotation to face target",
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
            default_value="false",
            description=("Whether to launch the dummy joint_state_publisher. Only meaningful "
                         "with publish_tf:=true; against the real robot it publishes the URDF "
                         "zero pose over the robot's real encoder angles"),
        ),
        DeclareLaunchArgument(
            "publish_tf",
            default_value="false",
            description=("Whether the LAPTOP publishes the robot's TF tree. False against the "
                         "real robot, which computes its own -- see the comment below. True "
                         "only for bag playback or a robot whose sensor launch is not running"),
        ),

        # ── 1. Laptop-side TF -- OFF by default against the real robot ────────
        #
        # The robot runs its own robot_state_publisher fed by lowstate_to_jointstate,
        # so it already publishes /tf, /tf_static and /robot_description from real
        # encoder angles. Running these here as well is actively harmful: the dummy
        # joint_state_publisher emits the URDF zero pose, two authorities on the same
        # frames produce TF_OLD_DATA storms, and the robot (Foxy) cannot deserialise
        # what the laptop (Jazzy) publishes, so it drowns in
        # 'invalid data size ... serdata.cpp:308'. Jazzy reads Foxy fine, which is why
        # simply consuming the robot's TF works.
        #
        # publish_tf:=true is for bag playback or a robot without its sensor launch up.
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_desc, "use_sim_time": False}],
            condition=IfCondition(LaunchConfiguration("publish_tf")),
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
        # For sources that publish clouds in 'livox_frame' while the URDF chain
        # ends at 'mid360_link': without the identity link the tree is split and
        # detections cannot reach 'pelvis' (inference still runs, so the symptom
        # is "detections but no TF").
        #
        # The real robot does NOT need this -- its Livox driver is configured with
        # frame_id=mid360_link. Bags recorded before that, and sim, still do.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_mid360_to_livox",
            arguments=["--frame-id", "mid360_link", "--child-frame-id", "livox_frame"],
            parameters=[{"use_sim_time": False}],
            output="log",
            condition=IfCondition(LaunchConfiguration("publish_tf")),
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
                "offset_ground": LaunchConfiguration("offset_ground"),
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
                "yaw_rate": LaunchConfiguration("yaw_rate"),
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
