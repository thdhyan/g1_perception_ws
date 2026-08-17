import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_detection = get_package_share_directory("livox_detection")
    default_rviz_config = os.path.join(pkg_detection, "config", "livox_human_viz.rviz")
    default_csv = "/home/thakk100/Projects/thesis/g1_perception_ws/l;ong_test.csv.Csv"

    # Get G1 robot URDF description
    robot_desc = ""
    try:
        pkg_desc = get_package_share_directory("g1_description")
        urdf_path = os.path.join(pkg_desc, "urdf", "g1_29dof.urdf")
        if os.path.exists(urdf_path):
            with open(urdf_path, "r") as f:
                robot_desc = f.read()
    except Exception:
        pass

    csv_path_arg = DeclareLaunchArgument(
        "csv_path",
        default_value=default_csv,
        description="Path to Livox CSV recording file",
    )
    rate_hz_arg = DeclareLaunchArgument(
        "rate_hz",
        default_value="10.0",
        description="LiDAR streaming frequency in Hz",
    )
    loop_arg = DeclareLaunchArgument(
        "loop",
        default_value="true",
        description="Loop playback continuously",
    )
    playback_speed_arg = DeclareLaunchArgument(
        "playback_speed",
        default_value="1.0",
        description="Playback speed multiplier (e.g. 1.0, 2.0)",
    )
    algorithm_arg = DeclareLaunchArgument(
        "algorithm",
        default_value="centerpoint",
        description="Detection algorithm: 'centerpoint' or 'pointpillar'",
    )
    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint_path",
        default_value="/home/thakk100/Projects/Thesis/livox_detection/pt/livox_model_1.pt",
        description="Path to model checkpoint (.pt file)",
    )
    score_threshold_arg = DeclareLaunchArgument(
        "score_threshold",
        default_value="0.25",
        description="Detection confidence threshold",
    )
    accumulate_frames_arg = DeclareLaunchArgument(
        "accumulate_frames",
        default_value="2",
        description="Point cloud accumulation window",
    )
    target_frame_arg = DeclareLaunchArgument(
        "target_frame",
        default_value="pelvis",
        description="Target frame for detections",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Whether to launch RViz2 visualization",
    )

    return LaunchDescription([
        csv_path_arg,
        rate_hz_arg,
        loop_arg,
        playback_speed_arg,
        algorithm_arg,
        checkpoint_arg,
        score_threshold_arg,
        accumulate_frames_arg,
        target_frame_arg,
        rviz_arg,

        # 1. Robot State Publisher (publishes G1 3D robot model)
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_desc,
                "use_sim_time": False,
            }],
        ),

        # 2. Joint State Publisher (publishes default zero joint state for static display)
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            name="joint_state_publisher",
            output="screen",
            parameters=[{
                "use_sim_time": False,
            }],
        ),

        # 3. Livox CSV Player/Streamer Node
        Node(
            package="livox_detection",
            executable="livox_csv_player_node",
            name="livox_csv_player_node",
            output="screen",
            parameters=[{
                "csv_path": LaunchConfiguration("csv_path"),
                "topic": "/livox/lidar",
                "frame_id": "mid360_link",
                "rate_hz": LaunchConfiguration("rate_hz"),
                "time_window_ms": 100.0,
                "filter_zeros": True,
                "loop": LaunchConfiguration("loop"),
                "playback_speed": LaunchConfiguration("playback_speed"),
                "publish_tf": True,
            }],
        ),

        # 4. 3D Object & Human Detector (CenterPoint / PointPillars)
        Node(
            package="livox_detection",
            executable="livox_detection_node",
            name="livox_detection_node",
            output="screen",
            parameters=[{
                "algorithm": LaunchConfiguration("algorithm"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "target_frame": LaunchConfiguration("target_frame"),
                "score_threshold": LaunchConfiguration("score_threshold"),
                "accumulate_frames": LaunchConfiguration("accumulate_frames"),
                "max_hz": 10.0,
                "input_topic": "/livox/lidar",
            }],
        ),

        # 5. Distance Sorter Node (ranks detected humans by distance)
        Node(
            package="livox_detection",
            executable="human_distance_sorter_node",
            name="human_distance_sorter_node",
            output="screen",
            parameters=[{
                "input_topic": "/g1/detections/livox",
                "output_topic": "/g1/sorted_humans",
                "min_score": 0.20,
            }],
        ),

        # 6. RViz 3D Visualizer
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_livox_playback",
            arguments=["-d", default_rviz_config] if os.path.exists(default_rviz_config) else [],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="log",
        ),
    ])
