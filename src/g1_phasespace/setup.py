from glob import glob

from setuptools import setup

package_name = "g1_phasespace"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thakk100",
    maintainer_email="th.dhyan.us@gmail.com",
    description="PhaseSpace hats as SMPL-X humans and the G1 head",
    license="MIT",
    entry_points={"console_scripts": [
        "humans = g1_phasespace.humans:main",
        "robot_head = g1_phasespace.robot_head:main",
    ]},
)
