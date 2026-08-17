from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments with defaults
    checkpoint_arg = DeclareLaunchArgument(
        "checkpoint_path",
        default_value="pt/livox_model_1.pt",
        description="Path to CenterPoint checkpoint (relative to G1_sim/detection or absolute)",
    )
    max_hz_arg = DeclareLaunchArgument(
        "max_hz",
        default_value="5.0",
        description="Max inference frequency (Hz); frames arriving faster are dropped",
    )
    score_threshold_arg = DeclareLaunchArgument(
        "score_threshold",
        default_value="0.4",
        description="Detection confidence threshold [0, 1]",
    )
    device_arg = DeclareLaunchArgument(
        "device",
        default_value="cuda",
        description="Torch device (cuda or cpu)",
    )
    input_topic_arg = DeclareLaunchArgument(
        "input_topic",
        default_value="/livox/lidar",
        description="Input Livox CustomMsg topic",
    )
    frame_override_arg = DeclareLaunchArgument(
        "frame_override",
        default_value="",
        description="Optional frame_id override for published detections",
    )

    return LaunchDescription([
        checkpoint_arg,
        max_hz_arg,
        score_threshold_arg,
        device_arg,
        input_topic_arg,
        frame_override_arg,
        Node(
            package="g1_perception",
            executable="livox_detection_node",
            name="livox_detection_node",
            output="screen",
            parameters=[{
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "max_hz": LaunchConfiguration("max_hz"),
                "score_threshold": LaunchConfiguration("score_threshold"),
                "device": LaunchConfiguration("device"),
                "input_topic": LaunchConfiguration("input_topic"),
                "frame_override": LaunchConfiguration("frame_override"),
            }],
        ),
    ])
