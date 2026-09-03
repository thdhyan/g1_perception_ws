#!/usr/bin/env python3
"""
smpl_demo.launch.py — LiDAR + detection + HMR + ReID server, no locomotion control.

For demo with real robot connected via DDS (ROS_DOMAIN_ID=42).
Robot runs livox_ros_driver2 (Foxy); this laptop runs everything else (Humble).

Stack:
  1. lidar_bridge      /livox/lidar → /livox/mid360/points (frame rewrite)
  2. livox_detection   VoxelNeXt → /g1/detections/livox
  3. smpl_hmr_node     LiDAR-HMR humanm3 → /g1/smpl/{mesh,joints,skeleton,tracks}
  4. reid_server_node  live β lookup table → /g1/reid/{table,matches}
  5. rviz2             visualise cloud + mesh + skeleton (optional)

No locomotion, no navigation, no rosbag recording by default.

Usage:
    ros2 launch g1_perception smpl_demo.launch.py

    # without RViz (headless):
    ros2 launch g1_perception smpl_demo.launch.py rviz:=false

    # tune detection sensitivity:
    ros2 launch g1_perception smpl_demo.launch.py score_threshold:=0.3 max_range:=4.0

    # enable auto-enroll in ReID table:
    ros2 launch g1_perception smpl_demo.launch.py auto_enroll:=true

    # record bag:
    ros2 launch g1_perception smpl_demo.launch.py record:=true
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
_RVIZ_CFG = os.path.join(_WS_ROOT, 'src', 'g1_perception', 'config', 'smpl_hmr.rviz')

# Storage SSD preferred path; falls back to ~/bags if SSD not mounted.
_STORAGE_BAG = '/Storage/data/thesis/g1_point_clouds'
_LOCAL_BAG   = os.path.expanduser('~/bags')

def _bag_default() -> str:
    """Return bag root: /Storage SSD if mounted, else ~/bags."""
    if os.path.ismount('/Storage'):
        os.makedirs(_STORAGE_BAG, exist_ok=True)
        return _STORAGE_BAG
    os.makedirs(_LOCAL_BAG, exist_ok=True)
    return _LOCAL_BAG


def generate_launch_description():
    args = [
        # ── source / device ──────────────────────────────────────────────────
        DeclareLaunchArgument('source',      default_value='real',
                              description='LiDAR source: real | sim | unitree_slam'),
        DeclareLaunchArgument('device',      default_value='cuda'),

        # ── VoxelNeXt detection ──────────────────────────────────────────────
        DeclareLaunchArgument('checkpoint',
                              default_value=os.path.join(_WS_ROOT, 'pt', 'voxelnext_nuscenes.pth')),
        DeclareLaunchArgument('score_threshold', default_value='0.35'),
        DeclareLaunchArgument('max_range',        default_value='4.0',
                              description='Ignore detections beyond this radius (m)'),
        DeclareLaunchArgument('show_boxes',       default_value='false'),

        # ── LiDAR-HMR ────────────────────────────────────────────────────────
        DeclareLaunchArgument('smpl_checkpoint',  default_value='humanm3'),
        DeclareLaunchArgument('beta_cos_thresh',  default_value='0.85'),
        DeclareLaunchArgument('beta_ema_alpha',   default_value='0.15'),
        DeclareLaunchArgument('beta_max_table',   default_value='30'),
        DeclareLaunchArgument('beta_debug_sims',  default_value='false'),
        DeclareLaunchArgument('show_mesh',        default_value='true'),
        DeclareLaunchArgument('show_skeleton',    default_value='true'),
        DeclareLaunchArgument('sync_slop',        default_value='0.15'),
        DeclareLaunchArgument('marker_lifetime',  default_value='0.5'),

        # ── ReID server ───────────────────────────────────────────────────────
        DeclareLaunchArgument('reid_cos_thresh',       default_value='0.85'),
        DeclareLaunchArgument('reid_ema_alpha',        default_value='0.15'),
        DeclareLaunchArgument('reid_delta_thresh',     default_value='0.25'),
        DeclareLaunchArgument('reid_max_table',        default_value='30'),
        DeclareLaunchArgument('auto_enroll',           default_value='false'),
        DeclareLaunchArgument('min_stable_frames',     default_value='4'),
        DeclareLaunchArgument('reid_publish_rate_hz',  default_value='10.0'),

        # ── optional extras ───────────────────────────────────────────────────
        DeclareLaunchArgument('rviz',   default_value='true'),
        DeclareLaunchArgument('record', default_value='false'),
        DeclareLaunchArgument('bag_out', default_value=_bag_default(),
                              description='Bag root dir; auto-selects /Storage if mounted'),
    ]

    source          = LaunchConfiguration('source')
    device          = LaunchConfiguration('device')
    checkpoint      = LaunchConfiguration('checkpoint')
    score_threshold = LaunchConfiguration('score_threshold')
    max_range       = LaunchConfiguration('max_range')
    show_boxes      = LaunchConfiguration('show_boxes')
    smpl_checkpoint = LaunchConfiguration('smpl_checkpoint')
    beta_cos_thresh = LaunchConfiguration('beta_cos_thresh')
    beta_ema_alpha  = LaunchConfiguration('beta_ema_alpha')
    beta_max_table  = LaunchConfiguration('beta_max_table')
    beta_debug_sims = LaunchConfiguration('beta_debug_sims')
    show_mesh       = LaunchConfiguration('show_mesh')
    show_skeleton   = LaunchConfiguration('show_skeleton')
    sync_slop       = LaunchConfiguration('sync_slop')
    marker_lifetime = LaunchConfiguration('marker_lifetime')
    bag_out         = LaunchConfiguration('bag_out')

    # ── 1. LiDAR bridge (/livox/lidar → /livox/mid360/points, in pelvis frame) ─
    # target_frame=pelvis applies TF transform so inverted Mid-360 (mounted
    # upside-down on G1) is correctly oriented for detection + HMR.
    lidar_bridge = Node(
        package='g1_perception',
        executable='lidar_bridge',
        name='g1_lidar_bridge',
        parameters=[{'source': source, 'target_frame': 'pelvis'}],
        output='screen',
    )

    # ── 2. VoxelNeXt detection ─────────────────────────────────────────────────
    detection = Node(
        package='livox_detection',
        executable='livox_detection_node',
        name='g1_livox_detection',
        output='screen',
        parameters=[{
            'algorithm':        'voxelnext',
            'checkpoint_path':  checkpoint,
            'device':           device,
            'score_threshold':  score_threshold,
            'accumulate_frames': 2,
            'max_hz':           10.0,
            'input_topic':      '/livox/mid360/points',
            'class_filter':     'pedestrian',
            'target_frame':     'pelvis',
        }],
    )

    # ── 3. LiDAR-HMR + β-tracker ───────────────────────────────────────────────
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
            'beta_ema_alpha':   beta_ema_alpha,
            'beta_max_table':   beta_max_table,
            'beta_debug_sims':  beta_debug_sims,
            'sync_slop':        sync_slop,
            'marker_lifetime':  marker_lifetime,
            'detection_topic':  '/g1/detections/livox',
            'cloud_topic':      '/livox/mid360/points',
            'show_mesh':        show_mesh,
            'show_skeleton':    show_skeleton,
            'show_boxes':       show_boxes,
        }],
    )

    # ── 4. ReID server (live β lookup table) ───────────────────────────────────
    reid_server = Node(
        package='g1_perception',
        executable='reid_server_node',
        name='reid_server_node',
        output='screen',
        parameters=[{
            'cos_thresh':        LaunchConfiguration('reid_cos_thresh'),
            'ema_alpha':         LaunchConfiguration('reid_ema_alpha'),
            'delta_thresh':      LaunchConfiguration('reid_delta_thresh'),
            'max_table_size':    LaunchConfiguration('reid_max_table'),
            'auto_enroll':       LaunchConfiguration('auto_enroll'),
            'min_stable_frames': LaunchConfiguration('min_stable_frames'),
            'publish_rate_hz':   LaunchConfiguration('reid_publish_rate_hz'),
        }],
    )

    # ── 5. RViz2 ───────────────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', _RVIZ_CFG] if os.path.exists(_RVIZ_CFG) else [],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='log',
    )

    # ── 6. Optional rosbag ─────────────────────────────────────────────────────
    bag_name = 'g1_demo_' + time.strftime('%Y%m%d_%H%M%S')
    bag_path = PathJoinSubstitution([bag_out, bag_name])
    record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record',
             '/livox/mid360/points',
             '/g1/detections/livox',
             '/g1/smpl/tracks',
             '/g1/reid/table',
             '/g1/reid/matches',
             '-o', bag_path, '--storage', 'sqlite3'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('record')),
    )

    return LaunchDescription(args + [
        lidar_bridge,
        detection,
        smpl_hmr,
        reid_server,
        rviz,
        record,
    ])
