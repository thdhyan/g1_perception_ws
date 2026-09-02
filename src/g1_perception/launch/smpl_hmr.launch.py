"""
smpl_hmr.launch.py — Live SMPL body-mesh estimation from G1 LiDAR.

Starts smpl_hmr_node alongside the detection pipeline.

Usage:
    ros2 launch g1_perception smpl_hmr.launch.py

Options (override on command line):
    checkpoint:=humanm3        LiDAR-HMR checkpoint (humanm3 / sloper4d / waymov2 / lidarh26m)
    device:=cuda               cuda or cpu
    min_score:=0.15            Detection confidence threshold
    max_range:=6.0             Ignore detections > N metres from sensor
    show_mesh:=true            Publish SMPL mesh markers
    show_skeleton:=true        Publish joint + skeleton markers
    show_boxes:=false          Publish detection bounding boxes
    detection_topic:=/g1/detections/livox
    cloud_topic:=/livox/mid360/points

RViz topics:
    /g1/smpl/mesh       — TRIANGLE_LIST, one per person (semi-transparent body mesh)
    /g1/smpl/joints     — SPHERE_LIST, 24 joints per person
    /g1/smpl/skeleton   — LINE_LIST, skeleton edges per person
    /g1/smpl/boxes      — CUBE, detection bounding boxes (optional)
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_WS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')
)
_HMR_DIR = os.path.join(_WS_ROOT, 'LiDAR-HMR')


def generate_launch_description():
    args = [
        DeclareLaunchArgument('checkpoint',       default_value='humanm3'),
        DeclareLaunchArgument('device',           default_value='cuda'),
        DeclareLaunchArgument('min_score',        default_value='0.15'),
        DeclareLaunchArgument('max_range',        default_value='6.0'),
        DeclareLaunchArgument('n_pts',            default_value='256'),
        DeclareLaunchArgument('sync_slop',        default_value='0.15'),
        DeclareLaunchArgument('marker_lifetime',  default_value='0.5'),
        DeclareLaunchArgument('show_mesh',        default_value='true'),
        DeclareLaunchArgument('show_skeleton',    default_value='true'),
        DeclareLaunchArgument('show_boxes',       default_value='false'),
        DeclareLaunchArgument('detection_topic',  default_value='/g1/detections/livox'),
        DeclareLaunchArgument('cloud_topic',      default_value='/livox/mid360/points'),
    ]

    smpl_node = Node(
        package='g1_perception',
        executable='smpl_hmr_node',
        name='smpl_hmr_node',
        output='screen',
        parameters=[{
            'checkpoint':       LaunchConfiguration('checkpoint'),
            'config_path':      [
                # resolved relative to _HMR_DIR at node startup via _load_hmr()
                'configs/mesh/', LaunchConfiguration('checkpoint'), '.yaml'
            ],
            'device':           LaunchConfiguration('device'),
            'min_score':        LaunchConfiguration('min_score'),
            'max_range':        LaunchConfiguration('max_range'),
            'n_pts':            LaunchConfiguration('n_pts'),
            'sync_slop':        LaunchConfiguration('sync_slop'),
            'marker_lifetime':  LaunchConfiguration('marker_lifetime'),
            'show_mesh':        LaunchConfiguration('show_mesh'),
            'show_skeleton':    LaunchConfiguration('show_skeleton'),
            'show_boxes':       LaunchConfiguration('show_boxes'),
            'detection_topic':  LaunchConfiguration('detection_topic'),
            'cloud_topic':      LaunchConfiguration('cloud_topic'),
        }],
    )

    return LaunchDescription(args + [smpl_node])
