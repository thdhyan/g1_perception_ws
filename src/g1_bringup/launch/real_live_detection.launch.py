#!/usr/bin/env python3
"""Real Robot LIVE 3D Human Detection, with an operator-confirmed approach.

Live continuous detection (not the 2-pass snapshot pipeline) plus
human_follow_and_greet_node, which walks to a standoff in front of the human
published on /g1/selected_human.

Nothing moves on its own. auto_execute defaults to false, so selecting a target
only ARMS the approach; the node then waits on /g1/approach_selected, which is
[Y] in the keyboard console. Pass follow:=false for a perception-only stack that
cannot move the robot at all.

Uses the continuous detector (livox_detection_node, sliding accumulation +
inference every frame) rather than the 2-pass snapshot pipeline.

Supported backends: 'voxelnext' (only).

All TF is computed ON THE ROBOT (real encoders -> /joint_states -> /tf); this
launch publishes no TF of its own. See the comment in section 1 below.

Prerequisites:
  1. Robot sensors + joint states running:
       ./scripts/start_robot_sensors.sh "src/g1_sensors.launch.py joint_states:=true"
     (the robot's copy of g1_sensors.launch.py defaults joint_states to false)
  2. Robot and laptop clocks in sync -- the robot's chrony has no reachable
     upstream and drifts minutes behind, which shows up as TF_OLD_DATA warnings.
  3. source setup_g1_env.sh  (ROS_DOMAIN_ID / CycloneDDS unicast to the robot)

Usage:
  ros2 launch g1_bringup real_live_detection.launch.py algorithm:=voxelnext
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

    default_rviz_config = os.path.join(pkg_livox_det, "config", "livox_human_viz.rviz")

    # Walk up to the workspace root rather than counting parents: this file is
    # reached through install/ or, under --symlink-install, through src/, and the
    # two are at different depths. A fixed parents[N] silently lands on
    # ~/Projects/thesis instead, so VoxelNeXt is never found and the node logs
    # a backend-load error (empty detections) ("No module named 'pcdet'").
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
        DeclareLaunchArgument(
            "algorithm",
            default_value="voxelnext",
            description="Detection backend (only 'voxelnext' is supported)",
        ),
        DeclareLaunchArgument(
            "checkpoint_path",
            default_value=voxelnext_ckpt_default,
            description="Model checkpoint (VoxelNeXt: pt/voxelnext_nuscenes.pth)",
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
            description=("Detection confidence threshold. Measured against a person at "
                         "4.3m on the real Mid-360: pedestrian peaks around 0.28, so 0.10 "
                         "is needlessly noisy and 0.30 rejects real people"),
        ),
        DeclareLaunchArgument(
            "class_filter",
            default_value="pedestrian",
            description="Comma-separated class whitelist; empty string keeps every class",
        ),
        DeclareLaunchArgument(
            "max_hz",
            default_value="5.0",
            description=("Inference rate cap (Hz). VoxelNeXt measures ~4.0-4.2Hz achieved "
                         "against a 5Hz cap on this GPU"),
        ),
        DeclareLaunchArgument(
            "accumulate_frames",
            default_value="10",
            description=("Sweeps accumulated per inference. 10 x ~20k points is what makes a "
                         "person at 4m dense enough to score; 2 sweeps scores near zero"),
        ),
        DeclareLaunchArgument(
            "max_distance",
            default_value="25.0",
            description="Discard detections farther than this 2D distance (m) from the sensor",
        ),
        DeclareLaunchArgument(
            "offset_ground",
            default_value="-0.3",
            description=("Z-shift (m) applied before inference and undone on the output boxes. "
                         "NOT the sensor height -- it is whatever puts the ground where the "
                         "nuScenes-trained model expects it. Measured on the real robot "
                         "(ground sits at z=-1.27 in mid360_link): a person at 4.3m scores "
                         "0.28 at -0.3, 0.24 at -0.45, 0.22 at -0.6, and only 0.14 at the old "
                         "sim default of 1.33"),
        ),
        DeclareLaunchArgument(
            "input_topic",
            default_value="/livox/lidar",
            description="Livox PointCloud2 topic published by the robot",
        ),
        DeclareLaunchArgument(
            "target_frame",
            default_value="pelvis",
            description="TF frame the detections are transformed into",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="true",
            description="Whether to launch RViz",
        ),
        DeclareLaunchArgument(
            "follow",
            default_value="true",
            description=("Run human_follow_and_greet_node, which walks to a standoff in "
                         "front of whichever human is published on /g1/selected_human and "
                         "owns the /g1/approach_selected service that [Y] calls. Safe to "
                         "leave on because auto_execute defaults to false: the node arms "
                         "the approach and waits. follow:=false makes this launch "
                         "perception-only, unable to move the robot at all"),
        ),
        DeclareLaunchArgument(
            "standoff_distance",
            default_value="1.50",
            description="Where to stop in front of the target human, in metres",
        ),
        DeclareLaunchArgument(
            "greeting_action",
            default_value="low_wave",
            description="Gesture on arrival; any bridge action name, or 'none'",
        ),
        DeclareLaunchArgument(
            "linear_speed",
            default_value="0.30",
            description="Approach walking speed (m/s)",
        ),
        DeclareLaunchArgument(
            "auto_execute",
            default_value="false",
            description=("Walk as soon as a human is selected. False means selecting only "
                         "ARMS the approach and the operator confirms it with [Y] in the "
                         "keyboard console (/g1/approach_selected) -- keep it false while "
                         "detections still jump between frames"),
        ),
        DeclareLaunchArgument(
            "auto_greet",
            default_value="true",
            description="Whether to greet automatically once the standoff is reached",
        ),

        # ── 1. NO TF published from the laptop ────────────────────────────────
        # TF comes entirely from the robot: lowstate_to_jointstate_node turns the
        # Unitree /lowstate encoders into /joint_states, and the robot's own
        # robot_state_publisher turns that into /tf. Start it with:
        #   ./scripts/start_robot_sensors.sh "src/g1_sensors.launch.py joint_states:=true"
        #
        # Running robot_state_publisher / joint_state_publisher here as well is
        # actively harmful, not merely redundant:
        #   - a laptop joint_state_publisher publishes the URDF *zero* pose, not
        #     the robot's real joints, so every detection lands in a wrong pelvis;
        #   - two authorities on the same frames produce TF_OLD_DATA storms;
        #   - the robot is Foxy and the laptop Jazzy. Foxy's rmw_cyclonedds cannot
        #     deserialise Jazzy's XCDR2, so anything the laptop publishes on a
        #     topic the robot subscribes to (/joint_states, /tf) floods the robot
        #     with 'invalid data size ... serdata.cpp:308'. The reverse direction
        #     is fine -- Jazzy reads Foxy's XCDR1, which is why /livox/lidar and
        #     the robot's /tf are readable here.
        #
        # No mid360_link->livox_frame shim either: the robot's Livox driver is
        # configured with frame_id=mid360_link, so clouds already arrive on a
        # frame the URDF chain contains.

        # ── 2. Live 3D Detector ───────────────────────────────────────────────
        Node(
            package="livox_detection",
            executable="livox_detection_node",
            name="livox_detection_node",
            output="screen",
            parameters=[{
                "algorithm": LaunchConfiguration("algorithm"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "voxelnext_cfg": LaunchConfiguration("voxelnext_cfg"),
                "voxelnext_dir": LaunchConfiguration("voxelnext_dir"),
                "input_topic": LaunchConfiguration("input_topic"),
                "target_frame": LaunchConfiguration("target_frame"),
                "score_threshold": LaunchConfiguration("score_threshold"),
                "class_filter": LaunchConfiguration("class_filter"),
                "max_hz": LaunchConfiguration("max_hz"),
                "accumulate_frames": LaunchConfiguration("accumulate_frames"),
                "max_distance": LaunchConfiguration("max_distance"),
                "offset_ground": LaunchConfiguration("offset_ground"),
                "device": "cuda",
            }],
        ),

        # ── 2b. Optional: walk to the selected human ─────────────────────────
        # Consumes /g1/selected_human (PoseStamped in pelvis), computes a standoff
        # goal along the robot->human vector, and drives there through
        # robot_bridge's socket. With auto_execute false it only arms the plan and
        # waits for /g1/approach_selected, which is [Y] in the keyboard console.
        Node(
            package="g1_arm_control",
            executable="human_follow_and_greet_node",
            name="human_follow_and_greet_node",
            output="screen",
            condition=IfCondition(LaunchConfiguration("follow")),
            parameters=[{
                "standoff_distance": LaunchConfiguration("standoff_distance"),
                "greeting_action": LaunchConfiguration("greeting_action"),
                "linear_speed": LaunchConfiguration("linear_speed"),
                "auto_execute": LaunchConfiguration("auto_execute"),
                "auto_greet": LaunchConfiguration("auto_greet"),
            }],
        ),

        # ── 3. RViz ───────────────────────────────────────────────────────────
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_real_live_detection",
            arguments=["-d", default_rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
