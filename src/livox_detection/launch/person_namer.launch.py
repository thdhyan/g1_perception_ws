from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="livox_detection",
                executable="person_namer_node",
                name="person_namer",
                output="screen",
                emulate_tty=True,  # keep stdin/stdout connected for keyboard input
            ),
        ]
    )
