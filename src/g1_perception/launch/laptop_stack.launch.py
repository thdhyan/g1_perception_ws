#!/usr/bin/env python3
"""Full G1 perception + mapping + navigation stack (LAPTOP-SIDE).

This launch file orchestrates the complete perception and navigation pipeline
that runs on the laptop. The real Unitree G1 robot publishes sensor data over
DDS (ethernet), which this stack ingests and processes for SLAM, object detection,
and autonomous navigation.

================================================================================
SETUP OVERVIEW
================================================================================

Robot Side (Unitree G1 running Ubuntu):
  1. On robot: tmux / ssh ubuntu@<robot-ip>
  2. Run: ros2 launch unitree_ros2 g1_sensors.launch.py
  3. Robot publishes:
     - /livox/lidar (sensor_msgs/PointCloud2) @ 10.8 Hz
     - /joint_states (sensor_msgs/JointState)
     - /tf, /tf_static (TF skeleton + sensor frames)
     - /camera/* (color image, depth, camera_info)

Laptop Side (This Launch):
  1. Ensure ROS_DOMAIN_ID matches robot (e.g., export ROS_DOMAIN_ID=1)
  2. Ensure G1_INTERFACE env var set (e.g., export G1_INTERFACE=eno2)
  3. Run: ros2 launch g1_perception laptop_stack.launch.py
  4. Monitor via: rviz2, or ros2 topic list

Saving the Map:
  After exploring:
    ros2 run nav2_map_server map_saver_cli -f ~/maps/my_room
  This creates:
    ~/maps/my_room.pgm   (occupancy grid image)
    ~/maps/my_room.yaml  (metadata)

Launching Nav2 Separately (if desired):
  ros2 launch nav2_bringup navigation_launch.py \
    map:=~/maps/my_room.yaml \
    use_sim_time:=false

================================================================================
NODES LAUNCHED (in order)
================================================================================

1. robot_state_publisher
   - Reads URDF (g1_29dof.urdf)
   - Publishes /tf for robot skeleton based on /joint_states
   - Merges with /tf_static from robot (sensor frames)

2. lidar_bridge (g1_perception)
   - source="real": subscribes /livox/lidar (robot's native LiDAR)
   - republishes as /livox/mid360/points (frame: mid360_link)

3. pointcloud_to_laserscan (pointcloud_to_laserscan pkg)
   - Converts /livox/mid360/points → /scan (LaserScan)
   - Config: pc2scan_params.yaml

4. async_slam_toolbox_node (slam_toolbox pkg)
   - Subscribes: /scan
   - Publishes: /map, /map_metadata, TF map→odom
   - Config: slam_toolbox_params.yaml

5. livox_detection_node (livox_detection package, VoxelNeXt backend)
   - 3D object detection from /livox/lidar
   - Publishes: /g1/detections/livox, /g1/detection_markers/livox
     (+ /g1/detections/voxelnext and /g1/detection_markers/voxelnext aliases)
   - Checkpoint: pt/voxelnext_nuscenes.pth (default)
   - Device: cuda (default)

6. human_selector_node (g1_perception)
   - Interactive: filters detections, prompts user for target human
   - Subscribes: /g1/detections/livox

7. nav_goal_node (g1_perception)
   - Sends Nav2 NavigateToPose goals
   - Subscribes: selected target pose from human_selector_node

8. nav2point (g1_perception)
   - Inverse kinematics planning for G1
   - Subscribes: target pose
   - Uses odom_topic (fallback: /unitree/slam_mapping/odom)
   - standoff_distance: 0.60 m (default)

9. loco_client (g1_perception)
   - Motion control interface for G1 locomotion
   - Requires: interface, use_robot parameters

================================================================================
LAUNCH ARGUMENTS
================================================================================
"""

from pathlib import Path
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    EnvironmentVariable,
)
from launch_ros.actions import Node, LifecycleNode
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import TimerAction, ExecuteProcess

# Workspace venv has torch+CUDA + spconv — inject into PYTHONPATH so the
# VoxelNeXt detection node (livox_detection package) can import it
_WS_VENV_SITE_PKGS = os.path.expanduser(
    "~/Projects/thesis/g1_perception_ws/.venv/lib/python3.12/site-packages"
)
_EXTRA_PYTHONPATH = _WS_VENV_SITE_PKGS


def launch_setup(context, *args, **kwargs):
    """Setup function to read URDF file at launch time."""
    urdf_file = LaunchConfiguration("urdf_file").perform(context)

    # Read URDF content
    urdf_content = Path(urdf_file).read_text()

    # Create robot_state_publisher with URDF content
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": ParameterValue(urdf_content, value_type=str),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
    )

    return [robot_state_publisher]


def generate_launch_description():
    # ── declare launch arguments ──────────────────────────────────────────────
    declare_urdf_file = DeclareLaunchArgument(
        "urdf_file",
        default_value=PathJoinSubstitution([
            FindPackageShare("g1_perception"),
            "config",
            "g1_29dof.urdf"
        ]),
        description="Path to G1 URDF file"
    )

    declare_checkpoint_path = DeclareLaunchArgument(
        "checkpoint_path",
        default_value=os.path.expanduser("~/Projects/thesis/g1_perception_ws/pt/voxelnext_nuscenes.pth"),
        description="Path to VoxelNeXt checkpoint"
    )

    declare_device = DeclareLaunchArgument(
        "device",
        default_value="cuda",
        description="Device for VoxelNeXt inference: 'cuda' or 'cpu'"
    )

    declare_score_threshold = DeclareLaunchArgument(
        "score_threshold",
        default_value="0.4",
        description="Score threshold for VoxelNeXt detections"
    )

    declare_interface = DeclareLaunchArgument(
        "interface",
        default_value=EnvironmentVariable("G1_INTERFACE", default_value="eno2"),
        description="Network interface for loco_client (set G1_INTERFACE env var)"
    )

    declare_use_robot = DeclareLaunchArgument(
        "use_robot",
        default_value="true",
        description="Enable actual robot motion control (true/false)"
    )

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation time (false for real robot, true for rosbag)"
    )

    # ── substitutions ─────────────────────────────────────────────────────────
    urdf_file = LaunchConfiguration("urdf_file")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    device = LaunchConfiguration("device")
    score_threshold = LaunchConfiguration("score_threshold")
    interface = LaunchConfiguration("interface")
    use_robot = LaunchConfiguration("use_robot")
    use_sim_time = LaunchConfiguration("use_sim_time")

    # ── config file paths ─────────────────────────────────────────────────────
    slam_params_file = PathJoinSubstitution([
        FindPackageShare("g1_perception"),
        "config",
        "slam_toolbox_params.yaml"
    ])

    pc2scan_params_file = PathJoinSubstitution([
        FindPackageShare("g1_perception"),
        "config",
        "pc2scan_params.yaml"
    ])

    # ── nodes ─────────────────────────────────────────────────────────────────

    # 1. Robot State Publisher (reads URDF at launch time via OpaqueFunction)
    opaque_rsp = OpaqueFunction(function=launch_setup)

    # 2. PointCloud2 to LaserScan conversion (directly from robot)
    pointcloud_to_laserscan = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan_node",
        remappings=[("cloud_in", "/livox/lidar")],
        parameters=[pc2scan_params_file],
        output="screen",
    )

    # 3b. Restamp /scan with laptop clock (fixes robot 8h clock skew for SLAM TF lookup)
    scan_restamper = Node(
        package="g1_perception",
        executable="scan_restamper",
        name="scan_restamper",
        output="screen",
    )

    # 4. SLAM Toolbox: /scan_synced → /map + TF map→odom
    # async_slam_toolbox_node is a lifecycle node in Jazzy — needs configure+activate.
    slam_toolbox = LifecycleNode(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox_node",
        namespace="",
        parameters=[
            slam_params_file,
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )
    # Auto-configure (10s) then activate (15s) — give node time to fully start
    slam_configure = TimerAction(
        period=10.0,
        actions=[ExecuteProcess(
            cmd=["ros2", "lifecycle", "set", "/slam_toolbox_node", "configure"],
            output="screen",
        )],
    )
    slam_activate = TimerAction(
        period=15.0,
        actions=[ExecuteProcess(
            cmd=["ros2", "lifecycle", "set", "/slam_toolbox_node", "activate"],
            output="screen",
        )],
    )

    # 5. VoxelNeXt 3D Object Detection (livox_detection package)
    detection_node = Node(
        package="livox_detection",
        executable="livox_detection_node",
        name="livox_detection_node",
        parameters=[{
            "algorithm": "voxelnext",
            "checkpoint_path": checkpoint_path,
            "voxelnext_cfg": os.path.expanduser(
                "~/Projects/thesis/g1_perception_ws/VoxelNeXt/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
            ),
            "voxelnext_dir": os.path.expanduser("~/Projects/thesis/g1_perception_ws/VoxelNeXt"),
            "device": device,
            "score_threshold": score_threshold,
            "input_topic": "/livox/lidar",
            "class_filter": "pedestrian",
            "target_frame": "pelvis",
        }],
        output="screen",
    )

    # 6. Human Selector: interactive selection of target person
    human_selector_node = Node(
        package="g1_perception",
        executable="human_selector_node",
        name="human_selector_node",
        parameters=[{
            "detection_topic": "/g1/detections/livox",
            "min_score": score_threshold,
        }],
        output="screen",
    )

    # 7. Nav Goal Node: send Nav2 NavigateToPose goals
    nav_goal_node = Node(
        package="g1_perception",
        executable="nav_goal_node",
        name="nav_goal_node",
        parameters=[{
            "map_frame": "map",
        }],
        output="screen",
    )

    # 8. LiDAR odometry: /livox/lidar → /odom + TF odom→base_link
    #    ICP scan matching via scipy+numpy — no open3d dependency.
    lidar_odometry = Node(
        package="g1_perception",
        executable="lidar_odometry_node",
        name="g1_lidar_odometry",
        parameters=[{
            "input_topic": "/livox/lidar",
            "odom_frame": "odom",
            "base_frame": "livox_frame",
            "publish_tf": True,
            "voxel_size": 0.1,
            "max_correspondence_dist": 0.5,
            "fitness_threshold": 0.3,
        }],
        output="screen",
    )

    # 9. Nav2Point: path following → joy → loco_client → SDK
    nav2point_node = Node(
        package="g1_perception",
        executable="nav2point",
        name="nav2point",
        parameters=[{
            "odom_topic": "/odom",   # our ICP odometry
            "standoff_distance": 0.60,
        }],
        output="screen",
    )

    # 10. LocoClient: motion control for G1
    loco_client_node = Node(
        package="g1_perception",
        executable="loco_client",
        name="loco_client",
        parameters=[{
            "interface": interface,
            "use_robot": use_robot,
        }],
        output="screen",
    )

    default_rviz = PathJoinSubstitution(
        [FindPackageShare("g1_perception"), "config", "g1_perception.rviz"]
    )

    declare_rviz_config = DeclareLaunchArgument(
        "rviz_config", default_value=default_rviz,
        description="RViz config file"
    )
    declare_launch_rviz = DeclareLaunchArgument(
        "launch_rviz", default_value="true",
        description="Set false to skip RViz"
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        output="screen",
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
    )

    # Inject torch-capable PYTHONPATH for the VoxelNeXt detection node (workspace venv)
    _existing_pp = os.environ.get("PYTHONPATH", "")
    _merged_pp = f"{_EXTRA_PYTHONPATH}:{_existing_pp}" if _existing_pp else _EXTRA_PYTHONPATH
    set_pythonpath = SetEnvironmentVariable("PYTHONPATH", _merged_pp)

    # ── return full launch description ─────────────────────────────────────────
    return LaunchDescription([
        # Inject PYTHONPATH first so all child processes inherit it
        set_pythonpath,
        # Declare arguments
        declare_urdf_file,
        declare_checkpoint_path,
        declare_device,
        declare_score_threshold,
        declare_interface,
        declare_use_robot,
        declare_use_sim_time,
        declare_rviz_config,
        declare_launch_rviz,

        # Launch nodes in order
        # NOTE: nav_goal_node, nav2point_node, loco_client_node omitted:
        #   - nav2 not installed (nav_goal_node / nav2point require nav2_msgs)
        #   - loco_client imports unitree_sdk2py — segfaults with rclpy in same process
        #   Run robot_bridge.py + cmd_pose_bridge separately for motion control.
        opaque_rsp,
        pointcloud_to_laserscan,
        scan_restamper,
        slam_toolbox,
        slam_configure,
        slam_activate,
        detection_node,
        human_selector_node,
        lidar_odometry,
        rviz_node,
    ])
