#!/usr/bin/env python3
"""
Gazebo Harmonic simulation launch with WASDQE keyboard teleoperation for Unitree G1 robot.

Launches:
  - Gazebo server + GUI with G1 warehouse world
  - robot_state_publisher (TF from URDF)
  - ros_gz_bridge (Gazebo <-> ROS2 topic bridge)
  - G1 model spawned into Gazebo
  - Static TFs: mid360_link->livox_frame, base_link->pelvis,
                d435_link->camera_color_optical_frame,
                d435_link->camera_depth_optical_frame
  - 3D PointPillars human detection (PointCloud2 input from /livox/mid360/points)
  - 3D LiDAR-Inertial SLAM (plain_slam_ros2)
  - WBC (GR00T balance & walk ONNX controller)
  - WASDQE Keyboard Teleop Node (interactive keyboard control in terminal)
  - RViz2 (optional)
"""

import os
import re
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Absolute checkpoint path (relative fallback handled by centerpoint_node itself)
CHECKPOINT_PATH = str(
    Path.home() / "Projects/thesis/G1_sim/detection/pt/livox_model_1.pt"
)


def preprocess_urdf(context, *args, **kwargs):
    """Strip <mujoco> tags and rewrite mesh paths to absolute file:// URIs for Gazebo."""
    urdf_path = context.launch_configurations["urdf_path"]
    g1_desc_share = os.path.abspath(get_package_share_directory("g1_description"))
    temp_urdf = os.path.join(g1_desc_share, "g1_gazebo_sim.urdf")
    try:
        with open(urdf_path) as f:
            content = f.read()

        # Strip <mujoco>...</mujoco> block
        content = re.sub(r'<mujoco>.*?</mujoco>', '', content, flags=re.DOTALL)

        # Resolve package:// URIs -> absolute file:// paths
        def resolve_pkg_uri(match):
            pkg_name = match.group(1)
            rel_path = match.group(2)
            try:
                share_dir = os.path.abspath(get_package_share_directory(pkg_name))
                abs_path = os.path.join(share_dir, rel_path)
                return f'file://{abs_path}'
            except Exception:
                return match.group(0)

        content = re.sub(
            r'package://([^/]+)/([^"\']+)',
            resolve_pkg_uri,
            content,
        )

        # Resolve $(find pkg) macros -> absolute share dir paths
        def resolve_find_expr(match):
            pkg_name = match.group(1)
            try:
                return os.path.abspath(get_package_share_directory(pkg_name))
            except Exception:
                return match.group(0)

        content = re.sub(
            r'\$\(find\s+([^)]+)\)',
            resolve_find_expr,
            content,
        )

        # Rewrite bare relative "meshes/foo.STL" paths -> absolute file:// URIs.
        meshes_abs = os.path.join(g1_desc_share, "meshes")

        def resolve_relative_mesh(match):
            rel = match.group(1)
            return f'filename="file://{meshes_abs}/{rel}"'

        content = re.sub(
            r'filename="meshes/([^"]+)"',
            resolve_relative_mesh,
            content,
        )

        with open(temp_urdf, 'w') as f:
            f.write(content)
        print(f"[sim_teleop.launch.py] Preprocessed URDF -> {temp_urdf} (mesh URIs resolved to {meshes_abs})")
        return []
    except Exception as e:
        print(f"[sim_teleop.launch.py] ERROR: URDF preprocess failed: {e}")
        raise


def launch_setup(context, *args, **kwargs):
    """Build launch actions after context is available."""
    urdf_path = context.launch_configurations["urdf_path"]
    world_path = context.launch_configurations["world"]
    use_sim_time_str = context.launch_configurations["use_sim_time"]
    use_sim_time = use_sim_time_str == "true"
    headless = context.launch_configurations["headless"]
    rviz_cfg = context.launch_configurations["rviz"]
    x_pos = context.launch_configurations["x"]
    y_pos = context.launch_configurations["y"]
    z_pos = context.launch_configurations["z"]
    detection = context.launch_configurations["detection"]
    checkpoint = context.launch_configurations["checkpoint_path"]
    device = context.launch_configurations["device"]

    g1_assets_dir = str(Path.home() / "Projects/thesis/G1_sim/assets")
    g1_desc_share = get_package_share_directory("g1_description")

    vendor_lib_dirs = ":".join(
        str(p) for p in Path("/opt/ros/jazzy/opt").glob("*/lib") if p.is_dir()
    )
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ws_lib = os.path.join(get_package_prefix("ros2_livox_simulation"), "lib")

    gz_env = {
        "GZ_SIM_RESOURCE_PATH": f"{g1_assets_dir}:{g1_desc_share}",
        "GZ_SIM_SYSTEM_PLUGIN_PATH": ws_lib,
        "LD_LIBRARY_PATH": f"{ws_lib}:{vendor_lib_dirs}:{existing_ld}",
    }
    if headless == "true":
        gz_env["GZ_HEADLESS"] = "1"

    bridge_config = os.path.join(
        get_package_share_directory("g1_bringup"), "config", "gz_bridge.yaml"
    )

    nodes = []

    # ── Gazebo server ──────────────────────────────────────────────────────────
    gz_cmd = ["gz", "sim"]
    if headless == "true":
        gz_cmd.append("-s")
    if context.launch_configurations.get("paused") != "true":
        gz_cmd.append("-r")
    gz_cmd.append(world_path)

    nodes.append(
        ExecuteProcess(
            cmd=gz_cmd,
            additional_env=gz_env,
            output="screen",
        )
    )

    # ── robot_state_publisher ─────────────────────────────────────────────────
    processed_urdf_path = os.path.join(g1_desc_share, "g1_gazebo_sim.urdf")
    try:
        robot_description = open(processed_urdf_path).read()
    except FileNotFoundError:
        raw = open(urdf_path).read()
        robot_description = re.sub(r"<mujoco>.*?</mujoco>", "", raw, flags=re.DOTALL)

    nodes.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[
                {"robot_description": robot_description},
                {"use_sim_time": use_sim_time},
            ],
            output="screen",
        )
    )

    # ── Spawn G1 into Gazebo ───────────────────────────────────────────────────
    nodes.append(
        Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-name", "g1",
                "-file", processed_urdf_path,
                "-x", x_pos,
                "-y", y_pos,
                "-z", z_pos,
            ],
            output="screen",
        )
    )

    # ── ros_gz_bridge ─────────────────────────────────────────────────────────
    nodes.append(
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            parameters=[{"config_file": bridge_config}],
            output="screen",
        )
    )

    # ── Static TFs: Sensor frames and Tree roots ──────────────────────────────
    # Sensor optical / frame bindings
    nodes.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_mid360_to_livox",
            arguments=[
                "--frame-id", "mid360_link",
                "--child-frame-id", "livox_frame",
            ],
            output="screen",
        )
    )

    # Sim Ground Truth TF Branch:
    # Gazebo PosePublisher publishes `warehouse -> g1` (ground truth world pose).
    # We bridge `g1` / `g1_29dof` -> `gt_base_link` -> `gt_pelvis` for ground truth tracking without colliding with the real SLAM/proprioception TF tree.
    nodes.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_g1_to_gt_base",
            arguments=[
                "--frame-id", "g1",
                "--child-frame-id", "gt_base_link",
            ],
            output="screen",
        )
    )

    nodes.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_g1_29dof_to_gt_base",
            arguments=[
                "--frame-id", "g1_29dof",
                "--child-frame-id", "gt_base_link",
            ],
            output="screen",
        )
    )

    nodes.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_gt_base_to_gt_pelvis",
            arguments=[
                "--frame-id", "gt_base_link",
                "--child-frame-id", "gt_pelvis",
            ],
            output="screen",
        )
    )

    # Connect Ground Truth World (warehouse) to SLAM Map (map)
    nodes.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_warehouse_to_map",
            arguments=[
                "--frame-id", "warehouse",
                "--child-frame-id", "map",
            ],
            output="screen",
        )
    )

    # Base link alias for pelvis in robot_state_publisher tree
    nodes.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_base_to_pelvis",
            arguments=[
                "--frame-id", "base_link",
                "--child-frame-id", "pelvis",
            ],
            output="screen",
        )
    )

    nodes.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_d435_to_color_optical",
            arguments=[
                "--roll", "-1.5707963",
                "--pitch", "0",
                "--yaw", "-1.5707963",
                "--frame-id", "d435_link",
                "--child-frame-id", "camera_color_optical_frame",
            ],
            output="screen",
        )
    )

    nodes.append(
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_d435_to_depth_optical",
            arguments=[
                "--roll", "-1.5707963",
                "--pitch", "0",
                "--yaw", "-1.5707963",
                "--frame-id", "d435_link",
                "--child-frame-id", "camera_depth_optical_frame",
            ],
            output="screen",
        )
    )

    # ── PointPillars 3D Human Detection ───────────────────────────────────────
    detection_algo = context.launch_configurations.get("detection_algorithm", "pointpillar")
    if detection == "true":
        nodes.append(
            Node(
                package="livox_detection",
                executable="livox_detection_node",
                name="livox_detection_node",
                output="screen",
                parameters=[{
                    "algorithm": detection_algo,
                    "checkpoint_path": checkpoint,
                    "device": device,
                    "input_topic": "/livox/mid360/points",
                    "target_frame": "pelvis",
                    "score_threshold": 0.10,
                    "accumulate_frames": 4,
                    "max_hz": 5.0,
                    "use_sim_time": use_sim_time,
                }],
            )
        )

    # ── WBC controller ────────────────────────────────────────────────────────
    policy_dir = Path.home() / "Projects/thesis/G1_sim/assets/policy"
    balance_onnx = str(policy_dir / "GR00T-WholeBodyControl-Balance.onnx")
    walk_onnx = str(policy_dir / "GR00T-WholeBodyControl-Walk.onnx")
    nodes.append(
        Node(
            package="g1_wbc",
            executable="wbc_node",
            name="g1_wbc_node",
            output="screen",
            parameters=[{
                "balance_policy_path": balance_onnx,
                "walk_policy_path": walk_onnx,
                "control_hz": 50.0,
                "use_sim_time": use_sim_time,
                "joint_topic_prefix": "/g1/joint",
            }],
        )
    )

    # ── SLAM Pipeline ─────────────────────────────────────────────────────────
    slam = context.launch_configurations.get("slam", "true")
    slam_type = context.launch_configurations.get("slam_type", "3d")

    if slam == "true":
        if slam_type == "3d":
            plain_slam_dir = get_package_share_directory("plain_slam_ros2")
            lio_config = os.path.join(plain_slam_dir, "config", "lio_3d_config.yaml")
            slam_config = os.path.join(plain_slam_dir, "config", "slam_3d_config.yaml")
            plain_config_dir = os.path.join(plain_slam_dir, "config")

            nodes.append(
                Node(
                    package="plain_slam_ros2",
                    executable="lio_3d_node",
                    name="lio_3d_node",
                    output="screen",
                    parameters=[
                        lio_config,
                        {
                            "param_files_dir": plain_config_dir,
                            "pointcloud_topic": "/livox/mid360/points",
                            "imu_topic": "/imu/data",
                            "odom_frame": "odom",
                            "imu_frame": "pelvis",
                            "use_sim_time": use_sim_time,
                        },
                    ],
                )
            )

            nodes.append(
                Node(
                    package="plain_slam_ros2",
                    executable="slam_3d_node",
                    name="slam_3d_node",
                    output="screen",
                    parameters=[
                        slam_config,
                        {
                            "param_files_dir": plain_config_dir,
                            "map_frame": "map",
                            "odom_frame": "odom",
                            "imu_frame": "pelvis",
                            "use_sim_time": use_sim_time,
                        },
                    ],
                )
            )
        else:
            nodes.append(
                Node(
                    package="pointcloud_to_laserscan",
                    executable="pointcloud_to_laserscan_node",
                    name="pointcloud_to_laserscan",
                    remappings=[("cloud_in", "/livox/mid360/points"), ("scan", "/scan")],
                    parameters=[{
                        "target_frame": "mid360_link",
                        "transform_tolerance": 0.05,
                        "min_height": -0.4,
                        "max_height": 1.2,
                        "angle_min": -3.14159265,
                        "angle_max": 3.14159265,
                        "angle_increment": 0.0087,
                        "scan_time": 0.1,
                        "range_min": 0.2,
                        "range_max": 30.0,
                        "use_inf": True,
                        "use_sim_time": use_sim_time,
                    }],
                    output="screen",
                )
            )

            slam_pkg_share = get_package_share_directory("slam_toolbox")
            slam_launch_file = os.path.join(slam_pkg_share, "launch", "online_async_launch.py")
            g1_bringup_share = get_package_share_directory("g1_bringup")
            slam_config = os.path.join(g1_bringup_share, "config", "g1_sim_slam.yaml")

            from launch.actions import IncludeLaunchDescription
            from launch.launch_description_sources import PythonLaunchDescriptionSource

            nodes.append(
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(slam_launch_file),
                    launch_arguments={
                        "slam_params_file": slam_config,
                        "use_sim_time": use_sim_time_str,
                    }.items(),
                )
            )



    # ── Dual RViz2 Displays (SLAM/Mapping and Ground Truth) ───────────────────
    if rviz_cfg == "true":
        g1_desc_share = get_package_share_directory("g1_description")
        mapping_rviz = os.path.join(g1_desc_share, "config", "g1_sim_mapping.rviz")
        gt_rviz = os.path.join(g1_desc_share, "config", "g1_sim_ground_truth.rviz")

        # 1. Primary Mapping & SLAM RViz (Fixed Frame: map)
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_mapping",
                arguments=["-d", mapping_rviz, "--title", "G1 - SLAM & Proprioception (map)"],
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
            )
        )

        # 2. Ground Truth Sim RViz (Fixed Frame: warehouse)
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2_ground_truth",
                arguments=["-d", gt_rviz, "--title", "G1 - Simulation Ground Truth (warehouse)"],
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
            )
        )

    return nodes


def generate_launch_description():
    pkg_share = get_package_share_directory("g1_bringup")
    default_world = os.path.join(pkg_share, "worlds", "g1_warehouse.sdf")

    default_urdf = os.path.join(
        get_package_share_directory("g1_description"),
        "urdf",
        "g1_29dof.urdf"
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "world",
            default_value=str(default_world),
            description="Path to Gazebo world SDF",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulated clock from Gazebo",
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Headless Gazebo (no GUI)",
        ),
        DeclareLaunchArgument(
            "urdf_path",
            default_value=default_urdf,
            description="Path to G1 URDF (mujoco tags stripped automatically)",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="true",
            description="Launch RViz2",
        ),
        DeclareLaunchArgument(
            "detection",
            default_value="true",
            description="Launch 3D human detection",
        ),
        DeclareLaunchArgument(
            "detection_algorithm",
            default_value="pointpillar",
            description="3D human detection algorithm ('pointpillar' or 'centerpoint')",
        ),
        DeclareLaunchArgument(
            "slam",
            default_value="true",
            description="Launch real-time SLAM",
        ),
        DeclareLaunchArgument(
            "slam_type",
            default_value="3d",
            description="SLAM algorithm: '3d' (plain_slam_ros2) or '2d' (slam_toolbox)",
        ),
        DeclareLaunchArgument(
            "checkpoint_path",
            default_value=CHECKPOINT_PATH,
            description="CenterPoint checkpoint path",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cuda",
            description="Torch device for detection (cuda or cpu)",
        ),
        DeclareLaunchArgument(
            "x", default_value="0.0", description="Spawn X (m)",
        ),
        DeclareLaunchArgument(
            "y", default_value="0.0", description="Spawn Y (m)",
        ),
        DeclareLaunchArgument(
            "z", default_value="0.75", description="Spawn Z (m)",
        ),
        DeclareLaunchArgument(
            "paused", default_value="false",
            description="Start Gazebo paused (no -r flag)",
        ),
        OpaqueFunction(function=preprocess_urdf),
        OpaqueFunction(function=launch_setup),
    ])
