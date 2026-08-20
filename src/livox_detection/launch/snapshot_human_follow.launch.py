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
        default_value="centerpoint",
        description="Detection algorithm: 'centerpoint', 'pointpillar', or 'voxelnext'",
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
    max_distance_arg = DeclareLaunchArgument(
        "max_distance",
        default_value="25.0",
        description=("Discard detections farther than this 2D distance (m) from the sensor. "
                     "Inference runs on the full accumulated cloud, so this is the only "
                     "range gate -- turn it down (e.g. 5.0) for a close-range human demo."),
    )
    offset_ground_arg = DeclareLaunchArgument(
        "offset_ground",
        default_value="1.33",
        description=("Z-shift (m) applied to points before inference and subtracted from "
                     "output boxes -- effectively the sensor height above ground. Should "
                     "match the Mid-360's actual mount height on the G1; a wrong value "
                     "moves the ground plane and makes the model miss nearby people."),
    )
    collect_frames_arg = DeclareLaunchArgument(
        "collect_frames",
        default_value="10",
        description="Number of LiDAR frames to accumulate in Pass 1",
    )
    collect_duration_arg = DeclareLaunchArgument(
        "collect_duration_sec",
        default_value="2.0",
        description="Duration in seconds to accumulate point clouds in Pass 1",
    )
    input_topic_arg = DeclareLaunchArgument(
        "input_topic",
        default_value="/livox/lidar",
        description="Input LiDAR topic (PointCloud2 or CustomMsg)",
    )
    target_frame_arg = DeclareLaunchArgument(
        "target_frame",
        default_value="pelvis",
        description="Target coordinate frame for human detections and robot control",
    )
    standoff_arg = DeclareLaunchArgument(
        "standoff_distance",
        default_value="0.80",
        description="Standoff distance in front of target human in meters (default 0.8m = 80cm)",
    )
    greeting_arg = DeclareLaunchArgument(
        "greeting_action",
        default_value="shake_hand",
        description="Greeting gesture to execute: 'shake_hand', 'low_wave', or 'high_wave'",
    )
    linear_speed_arg = DeclareLaunchArgument(
        "linear_speed",
        default_value="0.20",
        description="Approach walking speed in m/s (default 0.20 m/s)",
    )
    auto_execute_arg = DeclareLaunchArgument(
        "auto_execute",
        default_value="true",
        description="Whether to auto-execute approach motion upon human selection",
    )
    auto_greet_arg = DeclareLaunchArgument(
        "auto_greet",
        default_value="true",
        description="Whether to auto-execute greeting gesture upon arrival at standoff",
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

    # Load G1 URDF
    pkg_g1_desc = get_package_share_directory("g1_description")
    urdf_path = os.path.join(pkg_g1_desc, "urdf", "g1_29dof.urdf")
    with open(urdf_path, "r") as f:
        robot_desc = f.read()

    return LaunchDescription([
        algorithm_arg,
        checkpoint_arg,
        score_threshold_arg,
        max_distance_arg,
        offset_ground_arg,
        collect_frames_arg,
        collect_duration_arg,
        input_topic_arg,
        target_frame_arg,
        standoff_arg,
        greeting_arg,
        linear_speed_arg,
        auto_execute_arg,
        auto_greet_arg,
        rviz_arg,
        jsp_arg,
        publish_tf_arg,

        # 1. Robot State Publisher (URDF Model for RViz)
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

        # 1b. Sensor frame binding: the Livox driver publishes clouds in
        # 'livox_frame' while the URDF chain ends at 'mid360_link'. Without this
        # identity link the tree is split between the two and detections cannot
        # be transformed into 'pelvis' -- inference still runs, so the symptom
        # is "detections appear but have no TF". sim.launch.py has always
        # published this. Publish it laptop-side: the robot is on Foxy and the
        # laptop on Jazzy, and TF/String messages do not survive that CDR gap
        # (the rmw_cyclonedds serdata.cpp:308 errors).
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="tf_mid360_to_livox",
            arguments=["--frame-id", "mid360_link", "--child-frame-id", "livox_frame"],
            parameters=[{"use_sim_time": False}],
            output="log",
        ),

        # 2. 2-Pass Snapshot Pipeline Node (Collects 10 frames -> Freezes Dense Cloud -> CenterPoint 3D Detection)
        Node(
            package="livox_detection",
            executable="livox_snapshot_pipeline_node",
            name="livox_snapshot_pipeline_node",
            output="screen",
            parameters=[{
                "algorithm": LaunchConfiguration("algorithm"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "input_topic": LaunchConfiguration("input_topic"),
                "target_frame": LaunchConfiguration("target_frame"),
                "score_threshold": LaunchConfiguration("score_threshold"),
                "max_distance": LaunchConfiguration("max_distance"),
                "collect_frames": LaunchConfiguration("collect_frames"),
                "collect_duration_sec": LaunchConfiguration("collect_duration_sec"),
                "auto_start": True,
                "offset_ground": LaunchConfiguration("offset_ground"),
                "enable_cli_input": False,
            }],
        ),

        # 4. Human Follow & Greet Controller (Rotates, Walks at 0.20 m/s, Stops, Greets)
        Node(
            package="g1_arm_control",
            executable="human_follow_and_greet_node",
            name="human_follow_and_greet_node",
            output="screen",
            parameters=[{
                "standoff_distance": LaunchConfiguration("standoff_distance"),
                "greeting_action": LaunchConfiguration("greeting_action"),
                "linear_speed": LaunchConfiguration("linear_speed"),
                "auto_execute": LaunchConfiguration("auto_execute"),
                "auto_greet": LaunchConfiguration("auto_greet"),
            }],
        ),

        # 5. RViz 3D Visualization
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_livox_snapshot",
            arguments=["-d", default_rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
