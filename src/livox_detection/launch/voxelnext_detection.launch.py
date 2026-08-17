"""Launch VoxelNeXt 3D detection node for Livox Mid-360 LiDAR."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("livox_detection")

    # Resolve workspace root for VoxelNeXt defaults
    ws_root = Path(__file__).resolve().parents[4]  # launch/ -> share/ -> livox_detection/ -> install/ -> ws_root
    # Fallback: try to find VoxelNeXt relative to source
    voxelnext_dir_default = str(ws_root / "VoxelNeXt")
    voxelnext_cfg_default = str(
        ws_root / "VoxelNeXt" / "tools" / "cfgs" / "nuscenes_models" / "cbgs_voxel0075_voxelnext.yaml"
    )
    voxelnext_ckpt_default = str(ws_root / "pt" / "voxelnext_nuscenes.pth")

    return LaunchDescription([
        DeclareLaunchArgument(
            "checkpoint_path",
            default_value=voxelnext_ckpt_default,
            description="Path to VoxelNeXt .pth checkpoint",
        ),
        DeclareLaunchArgument(
            "voxelnext_cfg",
            default_value=voxelnext_cfg_default,
            description="Path to VoxelNeXt YAML config file",
        ),
        DeclareLaunchArgument(
            "voxelnext_dir",
            default_value=voxelnext_dir_default,
            description="Path to cloned VoxelNeXt repo (for pcdet import)",
        ),
        DeclareLaunchArgument(
            "score_threshold",
            default_value="0.25",
            description="Detection confidence threshold",
        ),
        DeclareLaunchArgument(
            "input_topic",
            default_value="/livox/lidar",
            description="Input LiDAR topic (PointCloud2 or Livox CustomMsg)",
        ),
        DeclareLaunchArgument(
            "target_frame",
            default_value="pelvis",
            description="Target TF frame for detection output",
        ),
        DeclareLaunchArgument(
            "max_hz",
            default_value="5.0",
            description="Maximum inference rate (Hz)",
        ),
        DeclareLaunchArgument(
            "offset_ground",
            default_value="1.33",
            description="Z-offset for Livox Mid-360 ground alignment (m)",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="false",
            description="Launch RViz visualization",
        ),

        # VoxelNeXt Detection Node
        Node(
            package="livox_detection",
            executable="livox_detection_node",
            name="voxelnext_detection",
            output="screen",
            parameters=[{
                "algorithm": "voxelnext",
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "voxelnext_cfg": LaunchConfiguration("voxelnext_cfg"),
                "voxelnext_dir": LaunchConfiguration("voxelnext_dir"),
                "score_threshold": LaunchConfiguration("score_threshold"),
                "input_topic": LaunchConfiguration("input_topic"),
                "target_frame": LaunchConfiguration("target_frame"),
                "max_hz": LaunchConfiguration("max_hz"),
                "offset_ground": LaunchConfiguration("offset_ground"),
                "accumulate_frames": 4,
                "device": "cuda",
            }],
        ),

        # Optional RViz
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_voxelnext",
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
