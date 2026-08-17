from glob import glob
from setuptools import find_packages, setup

package_name = "g1_wbc"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        # ROS2 expects executables in lib/<package>/ (libexec dir)
        ("lib/" + package_name, glob("scripts/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thakk100",
    maintainer_email="th.dhyan.us@gmail.com",
    description="Sim-agnostic GR00T WBC node for Unitree G1",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "wbc_node = g1_wbc.wbc_node:main",
        ],
    },
)
