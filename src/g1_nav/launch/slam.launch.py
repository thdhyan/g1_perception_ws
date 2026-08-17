import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("g1_nav")
    slam_toolbox_dir = get_package_share_directory("slam_toolbox")

    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    slam_params_file = LaunchConfiguration(
        "slam_params_file",
        default=os.path.join(pkg_dir, "config", "slam_params.yaml"),
    )
    input_cloud_topic = LaunchConfiguration("input_cloud_topic", default="/livox/lidar")

    # 1. PointCloud2 -> LaserScan Converter Node
    pcl2scan_node = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        parameters=[{
            "target_frame": "mid360_link",
            "transform_tolerance": 0.01,
            "min_height": -0.5,
            "max_height": 1.5,
            "angle_min": -3.14159,
            "angle_max": 3.14159,
            "angle_increment": 0.0087,
            "scan_time": 0.1,
            "range_min": 0.2,
            "range_max": 40.0,
            "use_inf": True,
            "inf_epsilon": 1.0,
            "use_sim_time": use_sim_time,
        }],
        remappings=[
            ("cloud_in", input_cloud_topic),
            ("scan", "/scan"),
        ],
    )

    # 2. SLAM Toolbox Online Async Node
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_dir, "launch", "online_async_launch.py")
        ),
        launch_arguments={
            "slam_params_file": slam_params_file,
            "use_sim_time": use_sim_time,
        }.items(),
    )

    return LaunchDescription([
        pcl2scan_node,
        slam_launch,
    ])
