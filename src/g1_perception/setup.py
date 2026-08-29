from glob import glob

from setuptools import find_packages, setup

package_name = "g1_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/config", glob("config/*.rviz")),
        ("share/" + package_name + "/config", glob("config/*.urdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thakk100",
    maintainer_email="th.dhyan.us@gmail.com",
    description=(
        "Bridges Unitree G1 LiDAR topics (sim or real robot) onto "
        "/livox/mid360/points, runs the VoxelNeXt detection pipeline (livox_detection), "
        "and fuses LiDAR + D435 RGB-D via CCVNorm-based depth completion."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "lidar_bridge = g1_perception.lidar_bridge:main",
            "lidar_odometry_node = g1_perception.lidar_odometry_node:main",
            "detection_bridge = g1_perception.detection_bridge:main",
            "ccvnorm_node = g1_perception.ccvnorm_node:main",
            "nav_goal_node = g1_perception.nav_goal_node:main",
            "human_selector_node = g1_perception.human_selector_node:main",
            "loco_client = g1_perception.loco_client:main",
            "nav2point = g1_perception.nav2point:main",
            "scan_restamper = g1_perception.scan_restamper:main",
            "move_to_xy = g1_perception.move_to_xy:main",
            "cmd_pose_bridge = g1_perception.cmd_pose_bridge:main",
        ],
    },
)
