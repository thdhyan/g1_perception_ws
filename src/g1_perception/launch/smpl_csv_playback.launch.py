#!/usr/bin/env python3
"""smpl_csv_playback.launch.py — replay a recorded LiDAR CSV through detection
+ SMPL-HMR, no robot needed.

Minimal stack (just detection + HMR, per request — no robot_state_publisher,
no joint_state_publisher, no distance sorter):
  1. livox_csv_player_node - streams CSV points -> /livox/lidar (10Hz, loops)
  2. livox_detection_node  - VoxelNeXt -> /g1/detections/livox
  3. smpl_hmr_node         - LiDAR-HMR + beta-lookup tracker -> /g1/smpl/*
  4. rviz2                 - visualize cloud + detections + SMPL mesh

Usage:
    ros2 launch g1_perception smpl_csv_playback.launch.py
    ros2 launch g1_perception smpl_csv_playback.launch.py csv_path:=/path/to/other.csv
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_WS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')
)
_DEFAULT_CSV = os.path.expanduser('~/Downloads/2026-07-29_17-21-48_points.csv')
_RVIZ_CFG = os.path.join(_WS_ROOT, 'src', 'g1_perception', 'config', 'smpl_hmr.rviz')


def generate_launch_description():
    args = [
        DeclareLaunchArgument('csv_path', default_value=_DEFAULT_CSV),
        DeclareLaunchArgument('rate_hz', default_value='10.0'),
        DeclareLaunchArgument('loop', default_value='true'),
        DeclareLaunchArgument('playback_speed', default_value='1.0'),
        DeclareLaunchArgument('checkpoint_path',
                              default_value=os.path.join(_WS_ROOT, 'pt', 'voxelnext_nuscenes.pth')),
        DeclareLaunchArgument('score_threshold', default_value='0.25'),
        DeclareLaunchArgument('device', default_value='cuda'),
        DeclareLaunchArgument('smpl_checkpoint', default_value='humanm3'),
        DeclareLaunchArgument('beta_cos_thresh', default_value='0.85'),
        DeclareLaunchArgument('max_range', default_value='6.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
    ]

    csv_path        = LaunchConfiguration('csv_path')
    rate_hz         = LaunchConfiguration('rate_hz')
    loop            = LaunchConfiguration('loop')
    playback_speed  = LaunchConfiguration('playback_speed')
    checkpoint_path = LaunchConfiguration('checkpoint_path')
    score_threshold = LaunchConfiguration('score_threshold')
    device          = LaunchConfiguration('device')
    smpl_checkpoint = LaunchConfiguration('smpl_checkpoint')
    beta_cos_thresh = LaunchConfiguration('beta_cos_thresh')
    max_range       = LaunchConfiguration('max_range')

    # ── 1. CSV playback -> /livox/lidar ─────────────────────────────────────
    csv_player = Node(
        package='livox_detection',
        executable='livox_csv_player_node',
        name='livox_csv_player_node',
        output='screen',
        parameters=[{
            'csv_path': csv_path,
            'topic': '/livox/lidar',
            'frame_id': 'mid360_link',
            'rate_hz': rate_hz,
            'time_window_ms': 100.0,
            'filter_zeros': True,
            'loop': loop,
            'playback_speed': playback_speed,
            'publish_tf': True,
        }],
    )

    # ── 2. VoxelNeXt detection ───────────────────────────────────────────────
    detection = Node(
        package='livox_detection',
        executable='livox_detection_node',
        name='livox_detection_node',
        output='screen',
        parameters=[{
            'algorithm': 'voxelnext',
            'checkpoint_path': checkpoint_path,
            'target_frame': 'pelvis',
            'score_threshold': score_threshold,
            'accumulate_frames': 2,
            'max_hz': 10.0,
            'input_topic': '/livox/lidar',
            'device': device,
        }],
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
            'detection_topic':  '/g1/detections/livox',
            'cloud_topic':      '/livox/lidar',
            'show_mesh':        True,
            'show_skeleton':    True,
            'show_boxes':       False,
        }],
    )

    # ── 4. RViz2 ──────────────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_smpl_csv_playback',
        arguments=['-d', _RVIZ_CFG] if os.path.exists(_RVIZ_CFG) else [],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='log',
    )

    return LaunchDescription(args + [
        csv_player,
        detection,
        smpl_hmr,
        rviz,
    ])
