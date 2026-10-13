"""PhaseSpace tracker + SMPL-X humans + G1 head + RViz.

ros2 launch g1_phasespace phasespace.launch.py [robot_group:=dhyan-hat-3] [rviz:=false]
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_group = LaunchConfiguration("robot_group")
    return LaunchDescription([
        DeclareLaunchArgument("device", default_value="cs-phasespace.cs.umn.edu"),
        DeclareLaunchArgument("profile", default_value="", description="session profile (empty: server default)"),
        DeclareLaunchArgument("robot_group", default_value="dhyan-hat-3", description="LED group on the G1 head"),
        DeclareLaunchArgument("smplx_model", default_value="~/Projects/HOVER/body_models/smplx/SMPLX_NEUTRAL.npz"),
        DeclareLaunchArgument("rviz", default_value="true"),
        GroupAction(scoped=True, actions=[IncludeLaunchDescription(  # scoped: its rviz:=false must not leak to ours
            PathJoinSubstitution([FindPackageShare("phasespace_ros2"), "launch", "phasespace.launch.py"]),
            launch_arguments={"device": LaunchConfiguration("device"), "profile": LaunchConfiguration("profile"),
                              "rviz": "false"}.items(),
        )]),
        Node(package="g1_phasespace", executable="humans", output="screen",
             parameters=[{"robot_group": robot_group, "smplx_model": LaunchConfiguration("smplx_model")}]),
        Node(package="g1_phasespace", executable="robot_head", output="screen",
             parameters=[{"robot_group": robot_group}]),
        Node(
            package="rviz2", executable="rviz2", output="log", condition=IfCondition(LaunchConfiguration("rviz")),
            arguments=["-d", PathJoinSubstitution([FindPackageShare("g1_phasespace"), "rviz", "phasespace.rviz"])],
        ),
    ])
