"""
G1 cmd_pose bridge launch file.

Subscribes to /g1/cmd_pose (geometry_msgs/msg/Twist) and forwards
relative (dx, dy, dyaw_degrees) motion deltas to robot_bridge.py.
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
    topic_arg = DeclareLaunchArgument(
        "topic",
        default_value="/g1/cmd_pose",
        description="Twist topic for relative pose delta commands",
    )

    return LaunchDescription([
        socket_path_arg,
        topic_arg,
        Node(
            package="g1_control",
            executable="cmd_pose_bridge",
            name="cmd_pose_bridge",
            output="screen",
            parameters=[{
                "socket_path": LaunchConfiguration("socket_path"),
                "topic": LaunchConfiguration("topic"),
            }],
        ),
    ])
