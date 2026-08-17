from glob import glob

from setuptools import find_packages, setup

package_name = "g1_control"

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
    description=(
        "G1 locomotion control bridges: cmd_vel_bridge (Nav2 velocity controller), "
        "cmd_pose_bridge (relative pose deltas), human_follower_node (follow human "
        "with standoff), and robot_bridge (standalone Unix socket RPC to G1 SDK)."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cmd_vel_bridge = g1_control.cmd_vel_bridge:main",
            "cmd_pose_bridge = g1_control.cmd_pose_bridge:main",
            "human_follower_node = g1_control.human_follower_node:main",
            "robot_bridge = g1_control.robot_bridge:main",
        ],
    },
)
