#!/usr/bin/env python3
"""Isaac ROS cuVSLAM + nvblox (RGBD, GPU) alongside the G1 sim + WBC.

Single-launch smoke/comparison harness for g1_isaac_slam:
  - Gazebo sim + robot_state_publisher + WBC balance/walk policy + TF fallback
    statics, via g1_bringup's sim_teleop.launch.py with slam:=false (so
    plain_slam_ros2 stays out of the way — cuVSLAM publishes to isolated
    vslam_odom/vslam_map frames instead of odom/map, so the two SLAM stacks
    never claim the same TF parent/child pair).
  - cuVSLAM (isaac_ros_visual_slam, RGBD mode) + nvblox (+ optional human
    segmentation) in ONE Docker container (own CUDA 13 runtime), crossing
    into the host's ROS2 Jazzy DDS graph via --network host.
  - A third RViz window (Fixed Frame: vslam_map) showing cuVSLAM's tracked
    path and nvblox's reconstructed mesh. It intentionally does NOT show the
    robot URDF — cuVSLAM's tree has no static link back to the main
    map->odom->pelvis tree (adding one would recreate the multi-parent TF
    conflict already fixed once this session). Use the sim_teleop RViz
    windows (Fixed Frame: map / warehouse) for the robot + WBC balance view.

Usage:
    ros2 launch g1_isaac_slam isaac_slam.launch.py
    ros2 launch g1_isaac_slam isaac_slam.launch.py detection:=true people_segmentation:=true
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Built by: docker build -t g1_isaac_slam:latest -f isaac_ros_ws/Dockerfile.g1_isaac_slam isaac_ros_ws/
# (extends the isaac-ros-cli's base image with ros-jazzy-isaac-ros-visual-slam,
# ros-jazzy-isaac-ros-nvblox, ros-jazzy-isaac-ros-unet, and the peoplesemseg
# model installer baked in — see isaac_ros_ws/Dockerfile.g1_isaac_slam)
DEFAULT_IMAGE = "g1_isaac_slam:latest"


def generate_launch_description():
    g1_bringup_share = get_package_share_directory("g1_bringup")
    isaac_share = get_package_share_directory("g1_isaac_slam")

    docker_image = LaunchConfiguration("docker_image")
    detection = LaunchConfiguration("detection")
    people_segmentation = LaunchConfiguration("people_segmentation")
    rviz = LaunchConfiguration("isaac_rviz")
    sim_rviz = LaunchConfiguration("sim_rviz")
    headless = LaunchConfiguration("headless")
    paused = LaunchConfiguration("paused")

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(g1_bringup_share, "launch", "sim_teleop.launch.py")
        ),
        launch_arguments={
            "slam": "false",          # plain_slam_ros2 stays off — cuVSLAM owns vslam_odom/vslam_map only
            "detection": detection,
            "rviz": sim_rviz,         # sim's own RViz windows (robot TF + LiDAR clouds); off by
                                      # default since this package brings its own window (below)
            "headless": headless,
            "paused": paused,
            "use_sim_time": "true",
        }.items(),
    )

    # cuVSLAM + nvblox (+ optional human segmentation), one container, one
    # combined ros2 launch. --network host + matching ROS_DOMAIN_ID/RMW crosses
    # into the same DDS graph as the host-side sim (same pattern already used
    # for robot<->laptop in DATA_COLLECTION.md).
    # --rm only cleans up on a graceful exit; a killed launch (or a crashed
    # sim taking the whole group down) leaves the container behind and the
    # next run dies with "container name is already in use".
    remove_stale_container = ExecuteProcess(
        cmd=["docker", "rm", "-f", "g1_isaac_slam_container"],
        output="log",
    )

    isaac_container = ExecuteProcess(
        cmd=[
            "docker", "run", "--rm",
            "--network", "host",
            "--gpus", "all",
            "--name", "g1_isaac_slam_container",
            "-e", "ROS_DOMAIN_ID=" + os.environ.get("ROS_DOMAIN_ID", "0"),
            "-e", "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
            docker_image,
            "bash", "-c",
            PythonExpression([
                "'. /opt/g1_ws/install/setup.bash && ros2 launch g1_isaac_slam "
                "_container_isaac_slam.launch.py people_segmentation:=' + '",
                people_segmentation, "'",
            ]),
        ],
        output="screen",
    )

    # Identity map -> vslam_map, so cuVSLAM's path/landmarks and nvblox's
    # reconstruction are viewable in the sim's own RViz (Fixed Frame: map).
    # Safe against the multi-parent rule: nothing else publishes a parent for
    # vslam_map, and cuVSLAM no longer claims pelvis (publish_odom_to_base_tf
    # is False in _container_isaac_slam.launch.py). It is a *visualization*
    # link only -- it asserts the two maps start aligned, which is true at
    # t=0 and drifts exactly as much as cuVSLAM drifts.
    map_to_vslam_map = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_vslam_map",
        arguments=["--frame-id", "map", "--child-frame-id", "vslam_map"],
        parameters=[{"use_sim_time": True}],
        output="log",
    )

    isaac_rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_isaac_slam",
        arguments=[
            # g1_sim_isaac.rviz = g1_description's g1_sim_mapping.rviz (robot
            # URDF, TF tree, Mid-360 + depth clouds, detections) with the
            # cuVSLAM/nvblox displays appended, all in Fixed Frame 'map' via
            # the map -> vslam_map static above. One window, both stacks.
            "-d", os.path.join(isaac_share, "config", "g1_sim_isaac.rviz"),
            "--title", "G1 - sim + cuVSLAM + nvblox (map)",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("docker_image", default_value=DEFAULT_IMAGE,
                               description="Isaac ROS image with visual_slam+nvblox+unet installed"),
        DeclareLaunchArgument("detection", default_value="false",
                               description="Launch VoxelNeXt human detection too (VRAM budget!)"),
        DeclareLaunchArgument("people_segmentation", default_value="false",
                               description="Feed nvblox a PeopleSemSegNet mask so moving humans "
                                           "are excluded from the static reconstruction (best-effort, "
                                           "see isaac_ros_ws/README — least-verified part of this package)"),
        DeclareLaunchArgument("isaac_rviz", default_value="true", description="Launch the cuVSLAM/nvblox RViz window"),
        DeclareLaunchArgument("sim_rviz", default_value="false",
                               description="Also launch the sim's own RViz windows (robot URDF + TF tree "
                                           "+ LiDAR point clouds, Fixed Frame map/warehouse)"),
        DeclareLaunchArgument("headless", default_value="false", description="Headless Gazebo (no GUI)"),
        DeclareLaunchArgument("paused", default_value="false", description="Start Gazebo paused"),
        sim,
        remove_stale_container,
        map_to_vslam_map,
        # Small delay: let the sim/cameras come up before cuVSLAM starts tracking.
        TimerAction(period=3.0, actions=[isaac_container]),
        isaac_rviz,
    ])
