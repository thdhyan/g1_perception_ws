#!/usr/bin/env python3
"""
Robot-side sensor & description launch file for Unitree G1 (runs on the Orin NX, ROS Foxy).

Launches:
  - robot_state_publisher      : URDF -> /robot_description + /tf, /tf_static
  - lowstate_to_jointstate     : Unitree /lowstate encoders -> /joint_states  (REAL joint angles)
  - livox_ros_driver2          : Mid-360 -> /livox/lidar (frame_id=mid360_link) + /livox/imu
  - realsense2_camera          : D435i -> /camera/color/*, /camera/depth/*,
                                 /camera/depth/color/points, /camera/imu

ALL TF IS COMPUTED HERE, ON THE ROBOT. The laptop must not run its own
robot_state_publisher/joint_state_publisher: the robot is Foxy and the laptop
Jazzy, and Foxy's rmw_cyclonedds cannot deserialise Jazzy's XCDR2 -- anything
the laptop publishes on /joint_states or /tf floods this machine with
'invalid data size ... serdata.cpp:308'. The reverse direction works fine, so
the laptop simply consumes what this file publishes.

Usage on the robot:
    ros2 launch src/g1_sensors.launch.py
    ros2 launch src/g1_sensors.launch.py camera:=false pointcloud:=false

Bandwidth note: the D435i depth point cloud is by far the heaviest topic here
(640x480x15 ~= 55 MB/s uncompressed on the wire). It is on by default because
the laptop-side stack wants it, but turn it off with pointcloud:=false when
only the LiDAR pipeline is running.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_g1_desc = get_package_share_directory("g1_description")
    urdf_path = os.path.join(pkg_g1_desc, "urdf", "g1_29dof.urdf")
    with open(urdf_path, "r") as f:
        robot_desc = f.read()

    lidar_config_path = (
        "/home/unitree/Projects/ros2_ws/src/livox_ros_driver2/config/MID360_config.json"
    )

    lidar_enabled = LaunchConfiguration("lidar", default="true")
    joint_states_enabled = LaunchConfiguration("joint_states", default="true")
    camera_enabled = LaunchConfiguration("camera", default="true")

    livox_ros2_params = [{
        "xfer_format": 0,
        "multi_topic": 0,
        "data_src": 0,
        "publish_freq": 10.0,
        "output_data_type": 0,
        "frame_id": "mid360_link",
        "user_config_path": lidar_config_path,
    }]
    # The Mid-360's onboard IMU needs no parameter here: the driver publishes
    # /livox/imu unconditionally, fed by the imu_data_port in MID360_config.json.
    # It declares no IMU-related parameters at all (checked in
    # livox_ros_driver2.cpp) -- adding one only produces an unknown-parameter error.

    nodes = [
        # 1. Robot State Publisher (publishes /robot_description and the kinematic TF tree)
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_desc, "use_sim_time": False}],
        ),
        # 2. Livox Mid-360 LiDAR driver (frame_id=mid360_link, no livox_frame shim needed)
        Node(
            package="livox_ros_driver2",
            executable="livox_ros_driver2_node",
            name="livox_lidar_publisher",
            output="screen",
            parameters=livox_ros2_params,
            condition=IfCondition(lidar_enabled),
        ),
        # 3. Unitree Low-State to JointState converter -- the ONLY source of real
        #    joint angles. Without it robot_state_publisher has no input and the
        #    whole TF tree freezes in the URDF zero pose.
        Node(
            package="lowstate_to_jointstate",
            executable="lowstate_to_jointstate_node",
            name="lowstate_to_jointstate_node",
            output="screen",
            condition=IfCondition(joint_states_enabled),
        ),
        # 4. RealSense D435i: colour, depth, aligned depth, depth point cloud, IMU.
        #    Topics land under the 'camera' namespace: /camera/color/image_raw,
        #    /camera/depth/image_rect_raw, /camera/depth/color/points, /camera/imu.
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            namespace="camera",
            name="camera",
            output="screen",
            condition=IfCondition(camera_enabled),
            parameters=[{
                "camera_name": "camera",
                "enable_color": True,
                "enable_depth": True,
                "rgb_camera.profile": LaunchConfiguration("rgb_profile"),
                "depth_module.profile": LaunchConfiguration("depth_profile"),
                # align_depth gives /camera/aligned_depth_to_color/image_raw, which
                # is what an RGBD consumer (cuVSLAM, nvblox) needs -- the raw depth
                # is in the depth optical frame, not the colour one.
                "align_depth.enable": True,
                "enable_sync": True,
                # No IR: the stereo IR pair is not used by anything here, and leaving
                # it on also creates the /camera/infra*, /camera/aligned_depth_to_infra1
                # topics and their TF frames.
                "enable_infra1": False,
                "enable_infra2": False,
                "pointcloud.enable": LaunchConfiguration("pointcloud"),
                # Colour the depth cloud from the RGB stream. Without an explicit
                # stream_filter the node textures from nothing and publishes
                # /camera/depth/color/points with x,y,z only -- which is why it
                # renders grey. 2 = RS2_STREAM_COLOR (0=ANY, 1=DEPTH, 3=INFRARED).
                "pointcloud.stream_filter": 2,
                "pointcloud.stream_index_filter": 0,
                # Keep points the colour camera cannot see (its FOV is narrower than
                # the depth module's) rather than dropping them.
                "allow_no_texture_points": True,
                # D435i IMU. unite_imu_method=2 (linear interpolation) merges the
                # separate gyro/accel streams into a single /camera/imu; without it
                # only /camera/gyro/sample and /camera/accel/sample are published
                # and most consumers reject them.
                "enable_gyro": LaunchConfiguration("camera_imu"),
                "enable_accel": LaunchConfiguration("camera_imu"),
                "unite_imu_method": 2,
                # NOTE: camera_imu defaults to FALSE on this robot. The D435i's IMU
                # is reached through the kernel HID/iio path here, which fails:
                #   HID set_power 1 failed for .../iio:device0/buffer/enable
                #   iio_hid_sensor: Frames didn't arrived within the predefined interval
                # /camera/imu then exists but never publishes, and the warning repeats
                # every 5s forever. The fix needs root -- blacklist the kernel HID
                # sensor drivers so librealsense falls back to its own USB backend:
                #   sudo tee /etc/modprobe.d/blacklist-realsense-hid.conf <<'EOF'
                #   blacklist hid_sensor_accel_3d
                #   blacklist hid_sensor_gyro_3d
                #   blacklist hid_sensor_trigger
                #   blacklist hid_sensor_iio_common
                #   EOF
                #   sudo rmmod hid_sensor_accel_3d hid_sensor_gyro_3d hid_sensor_trigger
                # Once that is done, launch with camera_imu:=true.
                # USB3 devices frequently come up in a bad state after an unclean
                # shutdown; this costs ~2s at startup and avoids 'failed to set
                # power state' loops.
                "initial_reset": True,
            }],
        ),
        # 5. Bind the RealSense TF tree to the robot's.
        #    realsense2_camera publishes its own internal tree (camera_link ->
        #    camera_depth_frame -> *_optical_frame ...) but nothing joins its root
        #    to the URDF, so every camera frame shows up in RViz as
        #    'No transform from [camera_*_frame] to [pelvis]'. The URDF's mount
        #    point is d435_link (fixed to torso_link, pitched 0.83 rad down).
        #    Identity here: d435_link and camera_link share the ROS body
        #    convention (x forward, y left, z up) and both sit at the depth
        #    module, so the residual is the few mm between the URDF's nominal
        #    mount point and the physical sensor origin -- uncalibrated, and the
        #    thing to measure if camera/LiDAR fusion ever looks systematically off.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_d435_to_camera_link",
            arguments=["0", "0", "0", "0", "0", "0", "d435_link", "camera_link"],
            output="log",
            condition=IfCondition(camera_enabled),
        ),
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            "lidar",
            default_value="true",
            description="Whether to launch the Livox Mid-360 LiDAR driver",
        ),
        DeclareLaunchArgument(
            "joint_states",
            default_value="true",
            description="Whether to launch the lowstate_to_jointstate converter (real joint angles)",
        ),
        DeclareLaunchArgument(
            "camera",
            default_value="true",
            description="Whether to launch the RealSense D435i driver",
        ),
        DeclareLaunchArgument(
            "pointcloud",
            default_value="true",
            description="Whether the D435i publishes /camera/depth/color/points (heaviest topic here)",
        ),
        DeclareLaunchArgument(
            "camera_imu",
            default_value="false",
            description=("Whether the D435i publishes its IMU on /camera/imu. Off by "
                         "default: the kernel HID/iio path is broken on this Jetson and "
                         "only produces a 5s warning loop -- see the note on the node"),
        ),
        DeclareLaunchArgument(
            "rgb_profile",
            default_value="640,480,15",
            description="D435i colour stream profile as width,height,fps",
        ),
        DeclareLaunchArgument(
            "depth_profile",
            default_value="640,480,15",
            description="D435i depth stream profile as width,height,fps",
        ),
        *nodes,
    ])
