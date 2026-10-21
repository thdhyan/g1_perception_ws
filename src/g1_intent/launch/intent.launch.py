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
ros2 launch g1_intent intent.launch.py record:=true slam:=true obstacles:=true   # bag + LIO/SLAM + LiDAR walls/tables

record:=true   ros2 bag (sqlite3, 4 GB splits) of PhaseSpace, /mocap/agents, /intent/*, joints, Mid-360, D435i, TF and the
               LIO/SLAM outputs into bag_dir/intent_<date>_<time>
slam:=true     lidar_slam.launch.py: plain_slam_ros2 LIO odometry + SLAM on the robot's /livox/* (frames lio_*)
obstacles:=true  lidar_obstacles: /livox/lidar -> walls / tables as scene obstacles (prompt + A* waypoints)
"""

import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import (EqualsSubstitution, LaunchConfiguration, NotSubstitution,
                                  PathJoinSubstitution)
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

# scripts/record_bag.sh --source real, plus the experiment. /lowstate and /sportmodestate need
# unitree_hg / unitree_go on the laptop, else ros2 bag skips them (unknown type).
BAG_TOPICS = [
    "/livox/lidar", "/livox/imu",
    "/camera/color/image_raw", "/camera/color/camera_info", "/camera/depth/image_rect_raw",
    "/camera/depth/camera_info", "/camera/aligned_depth_to_color/image_raw", "/camera/depth/color/points",
    "/camera/imu", "/joint_states", "/lowstate", "/secondary_imu", "/state_estimator/odom_pelvis",
    "/sportmodestate", "/unitree/slam_mapping/points", "/tf", "/tf_static", "/robot_description",
    "/mocap/agents",
]
# regex only: with an explicit topic list ros2 bag stops discovery early and misses topics that
# appear after it starts (the robot's sensors, PhaseSpace groups)
BAG_REGEX = "^(" + "|".join(BAG_TOPICS) + \
    r"|/phasespace/.*|/intent/.*|/pslam/(imu_odom|imu_pose|pose_graph_poses|filtered_map_cloud))$"


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
        DeclareLaunchArgument("record", default_value="false", description="true: ros2 bag record (see BAG_TOPICS)"),
        DeclareLaunchArgument("bag_dir", default_value="/Storage/g1_bags", description="home is full"),
        DeclareLaunchArgument("slam", default_value="false", description="true: LIO odometry + SLAM on /livox/*"),
        DeclareLaunchArgument("obstacles", default_value="false",
                              description="true: LiDAR walls / tables as scene obstacles (needs the G1 model TF)"),
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
        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare("g1_intent"), "launch", "lidar_slam.launch.py"]),
            condition=IfCondition(cfg("slam"))),
        Node(package="g1_intent", executable="lidar_obstacles", output="screen",
             parameters=[canvas],
             condition=IfCondition(cfg("obstacles"))),
        ExecuteProcess(
            cmd=["ros2", "bag", "record", "-s", "sqlite3", "--max-bag-size", str(4 * 1024 ** 3),  # as record_intent_session.sh
                 "-o", PathJoinSubstitution([cfg("bag_dir"), time.strftime("intent_%Y%m%d_%H%M%S")]),
                 "-e", BAG_REGEX],
            output="screen", condition=IfCondition(cfg("record"))),
    ])
