"""LiDAR-inertial odometry + SLAM on the laptop from the REAL robot's Mid-360 over ethernet.

plain_slam_ros2's lio_3d_node (odometry: /pslam/imu_odom, /pslam/imu_pose) + slam_3d_node
(loop-closed map: /pslam/filtered_map_cloud, pose graph) on /livox/lidar + /livox/imu, with
config/pslam_real/*_params.yaml (Mid-360 internal IMU extrinsic, IMU in g) instead of the
sim's pelvis-IMU calibration.

Frames are lio_map -> lio_odom -> lio_imu, NOT map/odom/imu: the robot's TF already has
mid360_link under torso_link, and slam_toolbox / Nav2 use map/odom. lio_map is gravity
aligned and starts at the IMU's first pose; it is not tied to phasespace (RViz fixed
frame lio_map to look at the map).

Robot side first: ./scripts/start_robot_sensors.sh (livox_ros_driver2 publishes /livox/*).

ros2 launch g1_intent lidar_slam.launch.py                 # LIO + SLAM
ros2 launch g1_intent lidar_slam.launch.py loop_closure:=false   # LIO odometry only
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def node_params(path, node, **overrides):
    """A plain_slam_ros2 config's params with overrides applied: rcl lets the file's
    node-specific keys beat a later wildcard (dict) override, so merge here instead."""
    with open(path) as f:
        params = yaml.safe_load(f)[node]["ros__parameters"]
    return {**params, **overrides}


def generate_launch_description():
    params_dir = os.path.join(get_package_share_directory("g1_intent"), "config", "pslam_real")
    pslam_cfg = os.path.join(get_package_share_directory("plain_slam_ros2"), "config")
    cfg = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument("loop_closure", default_value="true", description="false: lio_3d_node only"),
        DeclareLaunchArgument("map_dir", default_value="/Storage/pslam_maps/",
                              description="slam_3d_node saves its map here (created if missing; home is full)"),
        Node(package="plain_slam_ros2", executable="lio_3d_node", name="lio_3d_node", output="log",
             parameters=[node_params(os.path.join(pslam_cfg, "lio_3d_config.yaml"), "lio_3d_node",
                                     param_files_dir=params_dir, odom_frame="lio_odom", imu_frame="lio_imu")]),
        Node(package="plain_slam_ros2", executable="slam_3d_node", name="slam_3d_node", output="log",
             condition=IfCondition(cfg("loop_closure")),
             parameters=[node_params(os.path.join(pslam_cfg, "slam_3d_config.yaml"), "slam_3d_node",
                                     param_files_dir=params_dir, map_cloud_dir=cfg("map_dir"),
                                     map_frame="lio_map", odom_frame="lio_odom", imu_frame="lio_imu")]),
    ])
