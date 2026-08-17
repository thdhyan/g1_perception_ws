#!/usr/bin/env bash
# Source this before building/running anything in this workspace that talks
# to the G1 over unitree_sdk2py.
#
#   source setup_g1_env.sh
#
# Why this exists: conda's base python3 (3.13) shadows the system python3
# (3.12) that ROS2 Jazzy and unitree_sdk2py's cyclonedds==0.10.2 build
# are compiled against. With conda's bin ahead in PATH, `import rclpy` and
# `import cyclonedds._clayer` both fail with C-extension symbol errors
# (see ../unitree_sdk2_python/docs/SDK_TROUBLESHOOTING.md). Putting
# /usr/bin first in PATH fixes both at once.

export PATH=/usr/bin:/bin:$PATH
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_TYPE_DESCRIPTION_PUBLISH=0
export CYCLONEDDS_HOME=/home/thakk100/cyclonedds/install

# Default wired interface to the G1 -- override if needed:
export G1_INTERFACE="${G1_INTERFACE:-enp2s0}"

# Activate virtual environment if present
if [ -f "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate" ]; then
    source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"
    VENV_SITE="$(dirname "${BASH_SOURCE[0]}")/.venv/lib/python3.12/site-packages"
    if [ -d "$VENV_SITE" ]; then
        export PYTHONPATH="$VENV_SITE${PYTHONPATH:+:$PYTHONPATH}"
    fi
fi

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$WS_DIR/src/g1_control:$WS_DIR/src/g1_arm_control:$WS_DIR/src/g1_wbc:$WS_DIR/src/g1_perception:$WS_DIR/src/livox_detection${PYTHONPATH:+:$PYTHONPATH}"

for pkg_dir in "$WS_DIR"/install/*; do
    if [ -d "$pkg_dir" ]; then
        export AMENT_PREFIX_PATH="$pkg_dir${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
        if [ -d "$pkg_dir/lib/python3.12/site-packages" ]; then
            export PYTHONPATH="$pkg_dir/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
        fi
        if [ -d "$pkg_dir/bin" ]; then
            export PATH="$pkg_dir/bin:$PATH"
        fi
    fi
done

echo "python3:       $(which python3) ($(python3 --version))"
echo "G1_INTERFACE:  $G1_INTERFACE"
echo "CYCLONEDDS_HOME: $CYCLONEDDS_HOME"

