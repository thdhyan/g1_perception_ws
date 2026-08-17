import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    socket_path_arg = DeclareLaunchArgument(
        "socket_path",
        default_value="/tmp/g1_robot_bridge.sock",
        description="Path to robot_bridge Unix Domain Socket",
    )
    mock_mode_arg = DeclareLaunchArgument(
        "mock_mode",
        default_value="false",
        description="Whether to run arm controller in simulation/mock mode",
    )
    hold_sec_arg = DeclareLaunchArgument(
        "default_hold_seconds",
        default_value="3.0",
        description="Default gesture hold time in seconds",
    )

    return LaunchDescription([
        socket_path_arg,
        mock_mode_arg,
        hold_sec_arg,
        Node(
            package="g1_arm_control",
            executable="g1_arm_controller_node",
            name="g1_arm_controller_node",
            output="screen",
            parameters=[{
                "socket_path": LaunchConfiguration("socket_path"),
                "mock_mode": LaunchConfiguration("mock_mode"),
                "default_hold_seconds": LaunchConfiguration("default_hold_seconds"),
            }],
        ),
    ])
