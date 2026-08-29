from glob import glob

from setuptools import find_packages, setup

package_name = "g1_livox_pose"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thakk100",
    maintainer_email="th.dhyan.us@gmail.com",
    description="3D human pose estimation from Livox LiDAR with time-continuous skeleton sequences.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "human_pose_node = g1_livox_pose.human_pose_node:main",
            "pose_sequence_assembler_node = g1_livox_pose.pose_sequence_assembler_node:main",
        ],
    },
)
