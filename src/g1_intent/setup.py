from glob import glob

from setuptools import setup

package_name = "g1_intent"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config/pslam_real", glob("config/pslam_real/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thakk100",
    maintainer_email="th.dhyan.us@gmail.com",
    description="intent-sim LLM planner, review GUI and PhaseSpace goto executor for the G1",
    license="MIT",
    entry_points={"console_scripts": [
        "intent_planner = g1_intent.intent_planner:main",
        "intent_review_gui = g1_intent.intent_review_gui:main",
        "goto_executor = g1_intent.goto_executor:main",
        "lidar_obstacles = g1_intent.lidar_obstacles:main",
    ]},
)
