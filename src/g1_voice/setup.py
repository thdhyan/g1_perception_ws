from glob import glob

from setuptools import find_packages, setup

package_name = "g1_voice"

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
        "G1 voice interaction: audio_backend (standalone Unix socket RPC to the "
        "G1 VUI/AudioClient service), audio_bridge_node (ROS 2 topics for TTS, "
        "PCM playback, volume, LED, ASR), mic_node (raw microphone multicast), "
        "and dialog_node (Ollama/OpenAI replies into the onboard TTS)."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "audio_backend = g1_voice.audio_backend:main",
            "audio_bridge_node = g1_voice.audio_bridge_node:main",
            "mic_node = g1_voice.mic_node:main",
            "dialog_node = g1_voice.dialog_node:main",
        ],
    },
)
