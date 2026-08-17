import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_detection = get_package_share_directory("livox_detection")
    pkg_g1_desc = get_package_share_directory("g1_description")

    rviz_config_path = os.path.join(pkg_detection, "config", "livox_robot_lidar.rviz")
    if not os.path.exists(rviz_config_path):
        rviz_config_path = os.path.join(pkg_g1_desc, "config", "g1_robot_lidar.rviz")

    # Load G1 URDF model
    urdf_path = os.path.join(pkg_g1_desc, "urdf", "g1_29dof.urdf")
    with open(urdf_path, "r") as f:
        robot_desc = f.read()

    return LaunchDescription([
        # 1. Robot State Publisher:
        #    Subscribes to live /joint_states from robot and publishes full kinematic TF tree to /tf and /tf_static
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": robot_desc,
                "use_sim_time": False,
                "publish_frequency": 30.0,
            }],
        ),

        # 2. RViz2: Visualizes all TFs and LiDAR PointCloud (no robot mesh, no detection, no slam)
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_robot_lidar",
            arguments=["-d", rviz_config_path],
            output="screen",
        ),
    ])
