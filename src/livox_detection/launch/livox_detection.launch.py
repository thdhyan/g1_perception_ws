import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
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
    target_frame_arg = DeclareLaunchArgument(
        "target_frame",
        default_value="pelvis",
        description="Target coordinate frame for 3D detections (e.g., 'pelvis')",
    )
    score_threshold_arg = DeclareLaunchArgument(
        "score_threshold",
        default_value="0.10",
        description="Detection confidence score threshold [0.0 - 1.0]",
    )
    accumulate_frames_arg = DeclareLaunchArgument(
        "accumulate_frames",
        default_value="4",
        description="Number of consecutive point cloud sweeps to accumulate",
    )
    device_arg = DeclareLaunchArgument(
        "device",
        default_value="cuda",
        description="PyTorch inference device ('cuda' or 'cpu')",
    )
    input_topic_arg = DeclareLaunchArgument(
        "input_topic",
        default_value="/livox/lidar",
        description="Input LiDAR PointCloud2 / CustomMsg topic",
    )
    max_hz_arg = DeclareLaunchArgument(
        "max_hz",
        default_value="10.0",
        description="Maximum inference frequency (Hz)",
    )

    return LaunchDescription([
        algorithm_arg,
        checkpoint_arg,
        target_frame_arg,
        score_threshold_arg,
        accumulate_frames_arg,
        device_arg,
        input_topic_arg,
        max_hz_arg,
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
                "device": LaunchConfiguration("device"),
                "input_topic": LaunchConfiguration("input_topic"),
                "max_hz": LaunchConfiguration("max_hz"),
            }],
        ),
    ])
