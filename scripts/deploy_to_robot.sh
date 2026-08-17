#!/usr/bin/env bash
# Copy g1_perception package to robot and rebuild there (if needed).
# Only needed if you want to run nodes ON the robot.
# For laptop-side perception: NOT needed — just run laptop_stack.launch.py.
#
# Current architecture:
#   Robot  → publishes /livox/lidar + /joint_states + /tf_static
#   Laptop → runs all perception, SLAM, detection, navigation
#
# Usage: ./scripts/deploy_to_robot.sh

ROBOT="unitree@ubuntu.local"
ROBOT_WS="~/Projects/ros2_ws"

echo "=== Syncing g1_perception to robot ==="
rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='build/' --exclude='install/' \
    /home/thakk100/Projects/thesis/g1_perception_ws/src/g1_perception/ \
    "$ROBOT:$ROBOT_WS/src/g1_perception/"

echo "=== Building on robot ==="
ssh "$ROBOT" "source /opt/ros/foxy/setup.bash && \
    cd $ROBOT_WS && \
    colcon build --symlink-install --packages-select g1_perception 2>&1 | tail -20"

echo "Done. On robot: source $ROBOT_WS/install/setup.bash"
