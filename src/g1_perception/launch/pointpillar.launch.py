"""Launch PointPillar detection node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "checkpoint",
                default_value="G1_sim/detection/pt/pointpillar_model.pt",
                description="Path to PointPillar model checkpoint.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cuda",
                description="Torch device (cuda or cpu).",
            ),
            DeclareLaunchArgument(
                "score_threshold",
                default_value="0.4",
                description="Detection score threshold.",
            ),
            DeclareLaunchArgument(
                "input_topic",
                default_value="/livox/mid360/points",
                description="Input point cloud topic.",
            ),
            DeclareLaunchArgument(
                "max_hz",
                default_value="10.0",
                description="Maximum inference frequency (Hz).",
            ),
            DeclareLaunchArgument(
                "frame",
                default_value="",
                description="Override frame_id for published detections (empty = use cloud's frame).",
            ),
            Node(
                package="g1_perception",
                executable="pointpillar_node",
                name="pointpillar_detection",
                arguments=[
                    "--checkpoint", LaunchConfiguration("checkpoint"),
                    "--device", LaunchConfiguration("device"),
                    "--score-threshold", LaunchConfiguration("score_threshold"),
                    "--input-topic", LaunchConfiguration("input_topic"),
                    "--max-hz", LaunchConfiguration("max_hz"),
                    "--frame", LaunchConfiguration("frame"),
                ],
                output="screen",
            ),
        ]
    )
