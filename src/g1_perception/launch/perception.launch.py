"""Launch the G1 perception pipeline: LiDAR bridge + detection + depth completion."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    source_arg = DeclareLaunchArgument(
        "source",
        default_value="sim",
        description="LiDAR source: 'sim' (Isaac Sim RTX LiDAR) or 'real' (Unitree G1 onboard)",
        choices=["sim", "real"],
    )

    detection_node_path_arg = DeclareLaunchArgument(
        "detection_node_path",
        default_value="",
        description="Path to G1_sim/detection/detection_node.py (empty = auto-locate)",
    )

    ccvnorm_mode_arg = DeclareLaunchArgument(
        "ccvnorm_mode",
        default_value="ccvnorm_pseudo_stereo",
        description="Depth completion mode: 'ccvnorm_pseudo_stereo' (default) or 'ccvnorm_network'",
        choices=["ccvnorm_pseudo_stereo", "ccvnorm_network"],
    )

    source_cfg = LaunchConfiguration("source")
    detection_node_cfg = LaunchConfiguration("detection_node_path")
    ccvnorm_mode_cfg = LaunchConfiguration("ccvnorm_mode")

    # LiDAR bridge node: normalizes sim/real LiDAR onto /livox/mid360/points
    lidar_bridge_node = Node(
        package="g1_perception",
        executable="lidar_bridge",
        name="lidar_bridge",
        output="screen",
        parameters=[{"source": source_cfg}],
    )

    # Detection bridge node: runs G1_sim's detection pipeline
    detection_bridge_node = Node(
        package="g1_perception",
        executable="detection_bridge",
        name="detection_bridge",
        output="screen",
        parameters=[
            {"detection_node_path": detection_node_cfg},
            {"input_topic": "/livox/mid360/points"},
            {"backend": "livox"},
            {"device": "cuda"},
            {"score_threshold": 0.4},
        ],
    )

    # CCVNorm depth completion node: fuses LiDAR + RGB-D
    ccvnorm_node = Node(
        package="g1_perception",
        executable="ccvnorm_node",
        name="ccvnorm_node",
        output="screen",
        parameters=[
            {"lidar_topic": "/livox/mid360/points"},
            {"depth_topic": "/g1/camera/depth"},
            {"rgb_topic": "/g1/camera/rgb"},
            {"camera_info_topic": "/g1/camera/rgb/camera_info"},
            {"output_topic": "/g1/mapping/depth_completed"},
            {"camera_frame": "d435_color_optical_frame"},
            {"fusion_mode": ccvnorm_mode_cfg},
        ],
    )

    return LaunchDescription(
        [
            source_arg,
            detection_node_path_arg,
            ccvnorm_mode_arg,
            lidar_bridge_node,
            detection_bridge_node,
            ccvnorm_node,
        ]
    )
