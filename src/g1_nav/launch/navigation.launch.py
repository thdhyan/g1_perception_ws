import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_dir = get_package_share_directory("g1_nav")
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")

    map_yaml_file = LaunchConfiguration("map", default="")
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    params_file = LaunchConfiguration(
        "params_file",
        default=os.path.join(pkg_dir, "config", "nav2_params.yaml"),
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, "launch", "navigation_launch.py")
        ),
        launch_arguments={
            "map": map_yaml_file,
            "use_sim_time": use_sim_time,
            "params_file": params_file,
        }.items(),
    )

    return LaunchDescription([nav2_launch])
