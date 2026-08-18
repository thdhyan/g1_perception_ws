from glob import glob

from setuptools import find_packages, setup

package_name = "g1_isaac_slam"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml") + glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thakk100",
    maintainer_email="th.dhyan.us@gmail.com",
    description=(
        "Isaac ROS cuVSLAM + nvblox (RGBD, GPU) + human segmentation for the "
        "Unitree G1, run in Docker alongside plain_slam_ros2 for comparison."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
