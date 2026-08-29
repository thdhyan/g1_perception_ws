"""Bring up the ROS 2 side of the G1 voice stack.

audio_backend.py is NOT launched here: it must run outside any rclpy process
(see its docstring) and is started by hand, once, per session:

    python3 src/g1_voice/g1_voice/audio_backend.py enp2s0
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("g1_voice"), "config", "voice.yaml"
    )

    use_mic = LaunchConfiguration("use_mic")
    use_dialog = LaunchConfiguration("use_dialog")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_mic", default_value="false",
            description="Publish the raw microphone multicast stream. Only "
                        "needed for custom ASR; the onboard ASR is always on.",
        ),
        DeclareLaunchArgument(
            "use_dialog", default_value="true",
            description="Run the LLM dialog node.",
        ),
        Node(
            package="g1_voice",
            executable="audio_bridge_node",
            name="g1_audio_bridge",
            parameters=[config],
            output="screen",
        ),
        Node(
            package="g1_voice",
            executable="mic_node",
            name="g1_mic",
            parameters=[config],
            output="screen",
            condition=IfCondition(use_mic),
        ),
        Node(
            package="g1_voice",
            executable="dialog_node",
            name="g1_dialog",
            parameters=[config],
            output="screen",
            condition=IfCondition(use_dialog),
        ),
    ])
