"""Launch LiDAR-based odometry as an alternative to /unitree/slam_mapping/odom.

Run this instead of relying on the robot's onboard SLAM for odometry.
Pure LiDAR odometry via ICP scan matching on /livox/mid360/points.

Usage:
    ros2 launch g1_perception lidar_odometry.launch.py
    ros2 launch g1_perception lidar_odometry.launch.py voxel_size:=0.05 max_iterations:=50
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments with defaults
    input_topic_arg = DeclareLaunchArgument(
        "input_topic",
        default_value="/livox/mid360/points",
        description="Input PointCloud2 topic (from LiDAR)",
    )
    odom_frame_arg = DeclareLaunchArgument(
        "odom_frame",
        default_value="odom",
        description="Odometry frame ID",
    )
    base_frame_arg = DeclareLaunchArgument(
        "base_frame",
        default_value="base_link",
        description="Base link frame ID",
    )
    publish_tf_arg = DeclareLaunchArgument(
        "publish_tf",
        default_value="true",
        description="Publish odom → base_link transform",
    )
    voxel_size_arg = DeclareLaunchArgument(
        "voxel_size",
        default_value="0.1",
        description="Voxel size for downsampling before ICP (meters)",
    )
    max_correspondence_dist_arg = DeclareLaunchArgument(
        "max_correspondence_dist",
        default_value="0.5",
        description="Max correspondence distance for ICP (meters)",
    )
    max_iterations_arg = DeclareLaunchArgument(
        "max_iterations",
        default_value="30",
        description="Max ICP iterations per frame",
    )
    min_points_arg = DeclareLaunchArgument(
        "min_points",
        default_value="100",
        description="Min points to process; skip frame if fewer",
    )
    fitness_threshold_arg = DeclareLaunchArgument(
        "fitness_threshold",
        default_value="0.3",
        description="ICP fitness threshold; warn and skip update if below",
    )

    return LaunchDescription([
        input_topic_arg,
        odom_frame_arg,
        base_frame_arg,
        publish_tf_arg,
        voxel_size_arg,
        max_correspondence_dist_arg,
        max_iterations_arg,
        min_points_arg,
        fitness_threshold_arg,
        Node(
            package="g1_perception",
            executable="lidar_odometry_node",
            name="g1_lidar_odometry",
            output="screen",
            parameters=[{
                "input_topic": LaunchConfiguration("input_topic"),
                "odom_frame": LaunchConfiguration("odom_frame"),
                "base_frame": LaunchConfiguration("base_frame"),
                "publish_tf": LaunchConfiguration("publish_tf"),
                "voxel_size": LaunchConfiguration("voxel_size"),
                "max_correspondence_dist": LaunchConfiguration("max_correspondence_dist"),
                "max_iterations": LaunchConfiguration("max_iterations"),
                "min_points": LaunchConfiguration("min_points"),
                "fitness_threshold": LaunchConfiguration("fitness_threshold"),
            }],
        ),
    ])
