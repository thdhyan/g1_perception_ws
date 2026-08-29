import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_livox = get_package_share_directory("livox_detection")
    pkg_g1_desc = get_package_share_directory("g1_description")
    default_rviz_config = os.path.join(pkg_livox, "config", "livox_snapshot_viz.rviz")
    urdf_path = os.path.join(pkg_g1_desc, "urdf", "g1_29dof.urdf")

    with open(urdf_path, "r") as f:
        robot_desc = f.read()

    # Launch Arguments
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
        default_value="0.10",
        description="Detection confidence threshold",
    )
    collect_frames_arg = DeclareLaunchArgument(
        "collect_frames",
        default_value="10",
        description="Number of frames to accumulate in Pass 1",
    )
    collect_duration_arg = DeclareLaunchArgument(
        "collect_duration_sec",
        default_value="2.0",
        description="Duration in seconds to accumulate point clouds in Pass 1",
    )
    input_topic_arg = DeclareLaunchArgument(
        "input_topic",
        default_value="/livox/lidar",
        description="Input LiDAR topic",
    )
    standoff_arg = DeclareLaunchArgument(
        "standoff_distance",
        default_value="0.60",
        description="Standoff distance in front of target human (meters, default 0.60m = 60cm)",
    )
    greeting_action_arg = DeclareLaunchArgument(
        "greeting_action",
        default_value="shake_hand",
        description="Post-walkup arm greeting action: 'shake_hand', 'low_wave', 'high_wave', or 'wave_and_shake'",
    )
    auto_execute_arg = DeclareLaunchArgument(
        "auto_execute",
        default_value="true",
        description="Whether to auto-execute locomotion walk-up upon human selection",
    )
    auto_greet_arg = DeclareLaunchArgument(
        "auto_greet",
        default_value="true",
        description="Whether to auto-execute arm greeting upon arriving at 60cm standoff",
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
    publish_tf_arg = DeclareLaunchArgument(
        "publish_tf",
        default_value="false",
        description="Whether to publish static TFs from laptop",
    )

    return LaunchDescription([
        algorithm_arg,
        checkpoint_arg,
        score_threshold_arg,
        collect_frames_arg,
        collect_duration_arg,
        input_topic_arg,
        standoff_arg,
        greeting_action_arg,
        auto_execute_arg,
        auto_greet_arg,
        rviz_arg,
        jsp_arg,
        publish_tf_arg,

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

        # Optional static TFs for standalone laptop / offline tests
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_odom_to_base",
            arguments=["--x", "0", "--y", "0", "--z", "0", "--frame-id", "odom", "--child-frame-id", "base_link"],
            condition=IfCondition(LaunchConfiguration("publish_tf")),
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_base_to_pelvis",
            arguments=["--x", "0", "--y", "0", "--z", "0.76", "--frame-id", "base_link", "--child-frame-id", "pelvis"],
            condition=IfCondition(LaunchConfiguration("publish_tf")),
        ),

        # 2. 2-Pass Snapshot Pipeline (LiDAR Data Accumulation + 3D Detection + Interactive Selector)
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
                "auto_start": True,
            }],
        ),

        # 3. G1 Arm Controller Node (Exposes Arm Action Services: /g1/arm/shake_hand, /g1/arm/low_wave)
        Node(
            package="g1_arm_control",
            executable="g1_arm_controller_node",
            name="g1_arm_controller_node",
            output="screen",
            parameters=[{
                "socket_path": "/tmp/g1_robot_bridge.sock",
                "mock_mode": False,
                "default_hold_seconds": 3.0,
            }],
        ),

        # 4. Human Follow and Greet Coordinator (Locomotion Walk-Up to 60cm + Post-Walkup Arm Gesture)
        Node(
            package="g1_arm_control",
            executable="human_follow_and_greet_node",
            name="human_follow_and_greet_node",
            output="screen",
            parameters=[{
                "standoff_distance": LaunchConfiguration("standoff_distance"),
                "greeting_action": LaunchConfiguration("greeting_action"),
                "auto_execute": LaunchConfiguration("auto_execute"),
                "auto_greet": LaunchConfiguration("auto_greet"),
                "linear_speed": 0.20,
                "yaw_rate": 0.50,
                "socket_path": "/tmp/g1_robot_bridge.sock",
            }],
        ),

        # 5. RViz 3D Visualization
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_snapshot_follow_greet",
            arguments=["-d", default_rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
