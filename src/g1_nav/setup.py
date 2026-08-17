from glob import glob

from setuptools import find_packages, setup

package_name = "g1_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thakk100",
    maintainer_email="th.dhyan.us@gmail.com",
    description=(
        "Navigation stack for Unitree G1 humanoid using LiDAR-based SLAM "
        "and Nav2 with Livox Mid-360 PointCloud2 input."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "autonomous_mapper = g1_nav.autonomous_mapper:main",
            "keyboard_teleop = g1_nav.keyboard_teleop:main",
        ],
    },
)
