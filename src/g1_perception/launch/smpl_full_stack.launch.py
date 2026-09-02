#!/usr/bin/env python3
"""smpl_full_stack.launch.py — full live SMPL body-tracking stack on this laptop.

Everything runs locally (Humble). Robot only needs its sensor drivers up
(scripts/robot_sensors_remote.sh start) — no ROS2 nodes run on the robot itself.

Launches:
  1. lidar_bridge      - republish LiDAR to /livox/mid360/points
  2. livox_detection   - VoxelNeXt 3D detection -> /g1/detections/livox
  3. smpl_hmr_node     - LiDAR-HMR (beta/theta) + beta-lookup tracker
                         -> /g1/smpl/{mesh,joints,skeleton,tracks}
  4. rosbag2 recording - ros2 bag record -a (optional, on by default)

Usage:
    ros2 launch g1_perception smpl_full_stack.launch.py
    ros2 launch g1_perception smpl_full_stack.launch.py record:=false
    ros2 launch g1_perception smpl_full_stack.launch.py source:=sim
"""

import os
import time
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

_WS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')
)


def generate_launch_description():
    args = [
        DeclareLaunchArgument('source', default_value='real',
                              description='LiDAR source: real or sim'),
        DeclareLaunchArgument('device', default_value='cuda'),
        DeclareLaunchArgument('checkpoint',
                              default_value=os.path.join(_WS_ROOT, 'pt', 'voxelnext_nuscenes.pth'),
                              description='VoxelNeXt detection checkpoint'),
        DeclareLaunchArgument('score_threshold', default_value='0.4'),
        DeclareLaunchArgument('detection_topic', default_value='/g1/detections/livox'),
        DeclareLaunchArgument('smpl_checkpoint', default_value='humanm3',
                              description='LiDAR-HMR checkpoint tag'),
        DeclareLaunchArgument('beta_cos_thresh', default_value='0.85',
                              description='Cosine-sim threshold for the beta lookup tracker'),
        DeclareLaunchArgument('max_range', default_value='6.0'),
        DeclareLaunchArgument('show_boxes', default_value='false'),
        DeclareLaunchArgument('record', default_value='true',
                              description='Also start ros2 bag record -a'),
        DeclareLaunchArgument('bag_out', default_value=os.path.join(_WS_ROOT, 'bags')),
    ]

    source          = LaunchConfiguration('source')
    device          = LaunchConfiguration('device')
    checkpoint      = LaunchConfiguration('checkpoint')
    score_threshold = LaunchConfiguration('score_threshold')
    detection_topic = LaunchConfiguration('detection_topic')
    smpl_checkpoint = LaunchConfiguration('smpl_checkpoint')
    beta_cos_thresh = LaunchConfiguration('beta_cos_thresh')
    max_range       = LaunchConfiguration('max_range')
    show_boxes      = LaunchConfiguration('show_boxes')
    bag_out         = LaunchConfiguration('bag_out')

    # ── 1. LiDAR bridge ──────────────────────────────────────────────────────
    lidar_bridge = Node(
        package='g1_perception',
        executable='lidar_bridge',
        name='g1_lidar_bridge',
        parameters=[{'source': source}],
        output='screen',
    )

    # ── 2. VoxelNeXt detection ───────────────────────────────────────────────
    detection = Node(
        package='livox_detection',
        executable='livox_detection_node',
        name='g1_livox_detection',
        parameters=[{
            'algorithm': 'voxelnext',
            'checkpoint_path': checkpoint,
            'device': device,
            'score_threshold': score_threshold,
            'input_topic': '/livox/mid360/points',
            'class_filter': 'pedestrian',
            'target_frame': 'pelvis',
        }],
        output='screen',
    )

    # ── 3. LiDAR-HMR + beta-lookup tracker ───────────────────────────────────
    smpl_hmr = Node(
        package='g1_perception',
        executable='smpl_hmr_node',
        name='smpl_hmr_node',
        output='screen',
        parameters=[{
            'checkpoint':       smpl_checkpoint,
            'config_path':      ['configs/mesh/', smpl_checkpoint, '.yaml'],
            'device':           device,
            'min_score':        score_threshold,
            'max_range':        max_range,
            'beta_cos_thresh':  beta_cos_thresh,
            'detection_topic':  detection_topic,
            'cloud_topic':      '/livox/mid360/points',
            'show_mesh':        True,
            'show_skeleton':    True,
            'show_boxes':       show_boxes,
        }],
    )

    # ── 4. rosbag2 recording (everything, dynamically discovered) ───────────
    bag_name = 'g1_smpl_' + time.strftime('%Y%m%d_%H%M%S')  # evaluated once, at launch time
    bag_path = PathJoinSubstitution([bag_out, bag_name])
    record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-a', '-o', bag_path, '--storage', 'sqlite3'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('record')),
    )

    return LaunchDescription(args + [
        lidar_bridge,
        detection,
        smpl_hmr,
        record,
    ])
