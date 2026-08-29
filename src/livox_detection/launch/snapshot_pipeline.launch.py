import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("livox_detection")
    default_rviz_config = os.path.join(pkg_share, "config", "livox_snapshot_viz.rviz")

    algorithm_arg = DeclareLaunchArgument(
        "algorithm",
        default_value="voxelnext",
        description="Detection algorithm (only 'voxelnext' is supported)",
    )
    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint_path",
        default_value=os.path.expanduser("~/Projects/thesis/g1_perception_ws/pt/voxelnext_nuscenes.pth"),
        description="Path to VoxelNeXt checkpoint (.pth)",
    )
    score_threshold_arg = DeclareLaunchArgument(
        "score_threshold",
        default_value="0.01",
        description="Detection confidence threshold",
    )
    collect_frames_arg = DeclareLaunchArgument(
        "collect_frames",
        default_value="5",
        description="Number of LiDAR frames to accumulate in Pass 1",
    )
    collect_duration_arg = DeclareLaunchArgument(
        "collect_duration_sec",
        default_value="1.0",
        description="Duration in seconds to accumulate point clouds in Pass 1",
    )
    input_topic_arg = DeclareLaunchArgument(
        "input_topic",
        default_value="/livox/lidar",
        description="Input LiDAR topic",
    )
    auto_start_arg = DeclareLaunchArgument(
        "auto_start",
        default_value="false",
        description="Whether to immediately capture snapshot on startup (false = stream until Enter/trigger)",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Whether to launch RViz visualization",
    )
    jsp_arg = DeclareLaunchArgument(
        "joint_state_publisher",
        default_value="true",
        description="Whether to launch dummy joint_state_publisher",
    )

    # Load G1 URDF
    pkg_g1_desc = get_package_share_directory("g1_description")
    urdf_path = os.path.join(pkg_g1_desc, "urdf", "g1_29dof.urdf")
    with open(urdf_path, "r") as f:
        robot_desc = f.read()

    return LaunchDescription([
        algorithm_arg,
        checkpoint_arg,
        score_threshold_arg,
        collect_frames_arg,
        collect_duration_arg,
        input_topic_arg,
        auto_start_arg,
        rviz_arg,
        jsp_arg,

        # 1. Robot Model Visualization
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_desc, "use_sim_time": False}],
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

        # 2. Front 15m Filter & Snapshot Service Node
        Node(
            package="livox_detection",
            executable="livox_front_filter_node",
            name="livox_front_filter_node",
            output="screen",
            parameters=[{
                "input_topic": LaunchConfiguration("input_topic"),
                "min_x": 0.0,
                "max_range": 15.0,
            }],
        ),

        # 3. 2-Pass Snapshot Pipeline (Pass 1 Accumulate + Pass 2 VoxelNeXt 3D Detection + CLI Menu)
        Node(
            package="livox_detection",
            executable="livox_snapshot_pipeline_node",
            name="livox_snapshot_pipeline_node",
            output="screen",
            parameters=[{
                "algorithm": LaunchConfiguration("algorithm"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "input_topic": LaunchConfiguration("input_topic"),
                "target_frame": "pelvis",
                "score_threshold": LaunchConfiguration("score_threshold"),
                "collect_frames": LaunchConfiguration("collect_frames"),
                "collect_duration_sec": LaunchConfiguration("collect_duration_sec"),
                "auto_start": LaunchConfiguration("auto_start"),
                "offset_ground": 1.33,
            }],
        ),

        # 4. RViz 3D Visualization
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_livox_snapshot",
            arguments=["-d", default_rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
