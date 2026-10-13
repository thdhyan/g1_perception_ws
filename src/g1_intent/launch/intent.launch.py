"""Labeler + intent-sim planner + review GUI + goto executor (+ PhaseSpace, real or fake).

ros2 launch g1_intent intent.launch.py                                  # real PhaseSpace, real robot
ros2 launch g1_intent intent.launch.py fake_mocap:=true dry_run:=true   # laptop only
ros2 launch g1_intent intent.launch.py provider:=openrouter model:=anthropic/claude-sonnet-4.5
ros2 launch g1_intent intent.launch.py phasespace:=false                # PhaseSpace already running
ros2 launch g1_intent intent.launch.py phasespace:=false use_sim_time:=true dry_run:=true   # ros2 bag play --clock
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import AndSubstitution, LaunchConfiguration, NotSubstitution, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cfg = LaunchConfiguration
    canvas = {"canvas_origin_x": cfg("canvas_origin_x"), "canvas_origin_y": cfg("canvas_origin_y"),
              "intent_sim_dir": cfg("intent_sim_dir")}
    return LaunchDescription([
        DeclareLaunchArgument("provider", default_value="openai",
                              description="openai | openrouter | nim | gemini | ollama | dl | spark01..03"),
        DeclareLaunchArgument("model", default_value="gpt-6.1-sol"),
        DeclareLaunchArgument("base_url", default_value="", description="override endpoint, e.g. http://localhost:11434/v1"),
        DeclareLaunchArgument("prompt_style", default_value="proxemics"),
        DeclareLaunchArgument("temperature", default_value="0.7"),
        DeclareLaunchArgument("dry_run", default_value="false", description="true: no robot_bridge calls"),
        DeclareLaunchArgument("fake_mocap", default_value="false", description="true: fake_phasespace instead"),
        DeclareLaunchArgument("phasespace", default_value="true", description="start g1_phasespace (real)"),
        DeclareLaunchArgument("robot_group", default_value="dhyan-hat-3"),
        DeclareLaunchArgument("intent_sim_dir", default_value="~/Projects/thesis/intent-sim"),
        DeclareLaunchArgument("runs_dir", default_value="~/Projects/thesis/g1_perception_ws/intent_runs"),
        DeclareLaunchArgument("canvas_origin_x", default_value="-5.0",
                              description="PhaseSpace x at the canvas top-left"),
        DeclareLaunchArgument("canvas_origin_y", default_value="3.5",
                              description="PhaseSpace y at the canvas top-left"),
        DeclareLaunchArgument("labels_file", default_value="~/.ros/g1_mocap_labels.yaml"),
        DeclareLaunchArgument("use_sim_time", default_value="false", description="true for ros2 bag play --clock"),
        SetParameter(name="use_sim_time", value=cfg("use_sim_time")),

        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare("g1_phasespace"), "launch", "phasespace.launch.py"]),
            launch_arguments={"robot_group": cfg("robot_group"), "rviz": "false"}.items(),
            condition=IfCondition(AndSubstitution(cfg("phasespace"), NotSubstitution(cfg("fake_mocap")))),
        ),
        Node(package="g1_mocap_labeler", executable="fake_phasespace", output="screen",
             parameters=[{"robot_group": cfg("robot_group")}], condition=IfCondition(cfg("fake_mocap"))),
        Node(package="g1_mocap_labeler", executable="labeler_gui", output="screen",
             parameters=[{"labels_file": cfg("labels_file")}]),
        Node(package="g1_intent", executable="intent_planner", output="screen",
             parameters=[canvas, {"provider": cfg("provider"), "model": cfg("model"), "base_url": cfg("base_url"),
                                  "prompt_style": cfg("prompt_style"), "temperature": cfg("temperature"),
                                  "runs_dir": cfg("runs_dir")}]),
        Node(package="g1_intent", executable="intent_review_gui", output="screen", parameters=[canvas]),
        Node(package="g1_intent", executable="goto_executor", output="screen",
             parameters=[{"dry_run": cfg("dry_run")}]),
    ])
