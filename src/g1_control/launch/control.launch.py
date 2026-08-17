"""
G1 control launch file.

NOTE: Start robot_bridge.py manually before this launch:
    cd src/g1_control/g1_control && python3 robot_bridge.py <network_interface>

This launch file starts cmd_vel_bridge and human_follower_node.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    socket_path_arg = DeclareLaunchArgument(
        "socket_path",
        default_value="/tmp/g1_robot_bridge.sock",
        description="Path to robot_bridge Unix socket",
    )

    standoff_distance_arg = DeclareLaunchArgument(
        "standoff_distance",
        default_value="0.6",
        description="Standoff distance from human (m)",
    )

    map_frame_arg = DeclareLaunchArgument(
        "map_frame",
        default_value="map",
        description="Map frame ID for nav goals",
    )

    socket_path = LaunchConfiguration("socket_path")
    standoff_distance = LaunchConfiguration("standoff_distance")
    map_frame = LaunchConfiguration("map_frame")

    cmd_vel_bridge_node = Node(
        package="g1_control",
        executable="cmd_vel_bridge",
        name="g1_cmd_vel_bridge",
        parameters=[
            {"socket_path": socket_path},
            {"publish_hz": 10.0},
            {"deadman_timeout": 0.5},
            {"vx_scale": 1.0},
            {"vy_scale": 1.0},
            {"wz_scale": 1.0},
        ],
        output="screen",
    )

    human_follower_node = Node(
        package="g1_control",
        executable="human_follower_node",
        name="g1_human_follower",
        parameters=[
            {"standoff_distance": standoff_distance},
            {"map_frame": map_frame},
            {"pos_threshold": 0.15},
            {"yaw_threshold": 0.26},
            {"publish_rate": 2.0},
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            socket_path_arg,
            standoff_distance_arg,
            map_frame_arg,
            cmd_vel_bridge_node,
            human_follower_node,
        ]
    )
