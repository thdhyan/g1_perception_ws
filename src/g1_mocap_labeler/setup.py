from setuptools import setup

package_name = "g1_mocap_labeler"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="thakk100",
    maintainer_email="th.dhyan.us@gmail.com",
    description="Name PhaseSpace groups and publish them as TF + /mocap/agents",
    license="MIT",
    entry_points={"console_scripts": [
        "labeler_gui = g1_mocap_labeler.labeler_gui:main",
        "fake_phasespace = g1_mocap_labeler.fake_phasespace:main",
    ]},
)
