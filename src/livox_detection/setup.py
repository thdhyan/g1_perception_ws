from glob import glob

from setuptools import find_packages, setup

package_name = "livox_detection"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thakk100",
    maintainer_email="th.dhyan.us@gmail.com",
    description="Livox LiDAR 3D object detection using CenterPoint inference.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "livox_detection_node = livox_detection.livox_detection_node:main",
            "livox_csv_player_node = livox_detection.livox_csv_player_node:main",
            "livox_streamer_node = livox_detection.livox_csv_player_node:main",
            "human_distance_sorter_node = livox_detection.human_distance_sorter_node:main",
            "human_keyboard_selector_node = livox_detection.human_keyboard_selector_node:main",
            "human_loco_approach_node = livox_detection.human_loco_approach_node:main",
            "livox_snapshot_pipeline_node = livox_detection.livox_snapshot_pipeline_node:main",
            "livox_front_filter_node = livox_detection.livox_front_filter_node:main",
        ],
    },
)
