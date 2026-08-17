import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("livox_detection")
    default_rviz_config = os.path.join(pkg_share, "config", "livox_human_viz.rviz")

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
        default_value="0.10",
        description="Detection confidence threshold",
    )
    accumulate_frames_arg = DeclareLaunchArgument(
        "accumulate_frames",
        default_value="4",
        description="Point cloud accumulation window",
    )
    target_frame_arg = DeclareLaunchArgument(
        "target_frame",
        default_value="pelvis",
        description="Target coordinate frame for human detections",
    )
    standoff_arg = DeclareLaunchArgument(
        "standoff_distance",
        default_value="0.60",
        description="Standoff distance in front of target human in meters (default 0.6m = 60cm)",
    )
    auto_execute_arg = DeclareLaunchArgument(
        "auto_execute",
        default_value="true",
        description="Whether loco approach node auto-executes motion upon human selection",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Whether to launch RViz visualization",
    )
    jsp_arg = DeclareLaunchArgument(
        "joint_state_publisher",
        default_value="true",
        description="Whether to launch dummy joint_state_publisher (set false when real robot publishes /joint_states)",
    )

    publish_tf_arg = DeclareLaunchArgument(
        "publish_tf",
        default_value="false",
        description="Whether to publish static TFs from laptop (set false when connected to real robot)",
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
        accumulate_frames_arg,
        target_frame_arg,
        standoff_arg,
        auto_execute_arg,
        rviz_arg,
        jsp_arg,
        publish_tf_arg,

        # Robot State Publisher (Provides robot_description URDF model for RViz)
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

        # Optional static TFs (only used when robot TF tree is offline)
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


        # 1. 3D Human & Object Detector (CenterPoint / PointPillars)
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

        # 2. Node 1: Distance Sorter
        Node(
            package="livox_detection",
            executable="human_distance_sorter_node",
            name="human_distance_sorter_node",
            output="screen",
            parameters=[{
                "input_topic": "/g1/detections/livox",
                "output_topic": "/g1/sorted_humans",
                "min_score": LaunchConfiguration("score_threshold"),
            }],
        ),

        # 3. Node 2: Keyboard Selector (Terminal CLI)
        Node(
            package="livox_detection",
            executable="human_keyboard_selector_node",
            name="human_keyboard_selector_node",
            output="screen",
            parameters=[{
                "input_topic": "/g1/sorted_humans",
                "menu_cooldown_sec": 3.0,
            }],
        ),

        # 4. Node 3: Locomotion Approach Controller (60cm standoff)
        Node(
            package="livox_detection",
            executable="human_loco_approach_node",
            name="human_loco_approach_node",
            output="screen",
            parameters=[{
                "standoff_distance": LaunchConfiguration("standoff_distance"),
                "auto_execute": LaunchConfiguration("auto_execute"),
                "linear_speed": 0.3,
            }],
        ),

        # 5. RViz 3D Visualization
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_livox_human",
            arguments=["-d", default_rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
