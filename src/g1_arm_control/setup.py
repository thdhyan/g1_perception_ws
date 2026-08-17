from glob import glob
from setuptools import find_packages, setup

package_name = "g1_arm_control"

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
    description="G1 Arm Action Control and Human Interaction (Handshake, Low Wave, High Wave, Follow & Greet)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "g1_arm_controller_node = g1_arm_control.g1_arm_controller_node:main",
            "human_follow_and_greet_node = g1_arm_control.human_follow_and_greet_node:main",
        ],
    },
)
