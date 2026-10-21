"""Labeler + intent-sim planner + review UI + goto executor (+ PhaseSpace, real or fake).

ui:=web (default): one browser page (web_console, http://localhost:8090) holds the labeler,
the planner settings, the scene / prompt / SNGNN / Nav / VO views, plan review + approve and
the robot_bridge buttons; plus one RViz (g1_phasespace, with the G1 model on the head hat).
ui:=tk: the old Tk labeler + pygame review windows.

ros2 launch g1_intent intent.launch.py                                  # real PhaseSpace, real robot
ros2 launch g1_intent intent.launch.py fake_mocap:=true dry_run:=true   # laptop only
ros2 launch g1_intent intent.launch.py provider:=openrouter model:=anthropic/claude-sonnet-4.5
ros2 launch g1_intent intent.launch.py phasespace:=false                # PhaseSpace already running
ros2 launch g1_intent intent.launch.py phasespace:=false use_sim_time:=true dry_run:=true   # ros2 bag play --clock
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import (EqualsSubstitution, LaunchConfiguration, NotSubstitution,
                                  PathJoinSubstitution)
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
        DeclareLaunchArgument("prompt_variant", default_value="a",
                              description="a: global move_to_x_y | r: move_to_relative, every person's frame | "
                                          "ra: global scene + one assigned anchor (pick it in the web UI) | "
                                          "rs: global scene + select_anchor + move_to_relative"),
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
        DeclareLaunchArgument("ui", default_value="web", description="web: browser console | tk: Tk + pygame windows"),
        DeclareLaunchArgument("port", default_value="8090", description="web console port"),
        DeclareLaunchArgument("rviz", default_value="true"),
        SetParameter(name="use_sim_time", value=cfg("use_sim_time")),

        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare("g1_phasespace"), "launch", "phasespace.launch.py"]),
            # fake_mocap: no tracker, but robot_head, the G1 model and RViz still run on the fake TFs
            launch_arguments={"robot_group": cfg("robot_group"), "rviz": cfg("rviz"),
                              "tracker": NotSubstitution(cfg("fake_mocap"))}.items(),
            condition=IfCondition(cfg("phasespace")),
        ),
        Node(package="g1_mocap_labeler", executable="fake_phasespace", output="screen",
             parameters=[{"robot_group": cfg("robot_group")}], condition=IfCondition(cfg("fake_mocap"))),
        Node(package="g1_mocap_labeler", executable="labeler_gui", output="screen",
             parameters=[{"labels_file": cfg("labels_file")}], condition=IfCondition(EqualsSubstitution(cfg("ui"), "tk"))),
        Node(package="g1_intent", executable="web_console", output="screen",  # runs the labeler in-process
             parameters=[canvas, {"labels_file": cfg("labels_file"), "runs_dir": cfg("runs_dir"),
                                  "dry_run": cfg("dry_run"), "port": cfg("port"), "provider": cfg("provider"),
                                  "model": cfg("model"), "prompt_style": cfg("prompt_style"),
                                  "prompt_variant": cfg("prompt_variant"), "temperature": cfg("temperature")}],
             condition=IfCondition(EqualsSubstitution(cfg("ui"), "web"))),
        Node(package="g1_intent", executable="intent_planner", output="screen",
             parameters=[canvas, {"provider": cfg("provider"), "model": cfg("model"), "base_url": cfg("base_url"),
                                  "prompt_style": cfg("prompt_style"), "prompt_variant": cfg("prompt_variant"),
                                  "temperature": cfg("temperature"),
                                  "runs_dir": cfg("runs_dir")}]),
        Node(package="g1_intent", executable="intent_review_gui", output="screen", parameters=[canvas],
             condition=IfCondition(EqualsSubstitution(cfg("ui"), "tk"))),
        Node(package="g1_intent", executable="goto_executor", output="screen",
             parameters=[{"dry_run": cfg("dry_run")}]),
    ])
