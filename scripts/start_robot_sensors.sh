#!/usr/bin/env bash
# Start sensor launch on real G1 robot (ROS Foxy) in a tmux session.
#
# The robot publishes LiDAR (+ its IMU), the D435i colour/depth/pointcloud/IMU,
# real /joint_states from the Unitree encoders, and the whole TF tree. The
# laptop publishes no TF of its own -- see g1_sensors.launch.py's docstring.
#
# Usage:
#   ./scripts/start_robot_sensors.sh               # starts src/g1_sensors.launch.py by default
#   ./scripts/start_robot_sensors.sh <launch_file> # starts custom launch file
#
# Launch arguments can be appended to the target, quoted as one word:
#   ./scripts/start_robot_sensors.sh "src/g1_sensors.launch.py camera:=false pointcloud:=false"

# mDNS (ubuntu.local) does not resolve on the laptop -- use the wired IP.
# Override with:  ROBOT_HOST=unitree@other-host ./scripts/start_robot_sensors.sh
ROBOT_HOST="${ROBOT_HOST:-unitree@192.168.123.164}"
ROS2_WS="/home/unitree/Projects/ros2_ws"
TMUX_SESSION="sensors"
LAUNCH_TARGET="${1:-src/g1_sensors.launch.py}"

echo "[*] Connecting to $ROBOT_HOST to start G1 sensor stack ($LAUNCH_TARGET)..."

# Kill any existing sensors tmux session or leftover processes
ssh -o StrictHostKeyChecking=no "$ROBOT_HOST" "
    tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
    pkill -f g1_sensors 2>/dev/null || true
    pkill -f livox_ros_driver2 2>/dev/null || true
    pkill -f lowstate_to_jointstate 2>/dev/null || true
    pkill -f realsense2_camera_node 2>/dev/null || true
"

# Launch in tmux with complete CycloneDDS + ROS Foxy environment
ssh -o StrictHostKeyChecking=no "$ROBOT_HOST" "tmux new-session -d -s $TMUX_SESSION -x 220 -y 50 'bash --noprofile --norc -c \"
source /opt/ros/foxy/setup.bash
source /home/unitree/cyclonedds_ws/install/setup.bash 2>/dev/null || true
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/home/unitree/cyclonedds_ws/cyclonedds.xml
export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH
export ROS_DOMAIN_ID=0
source /home/unitree/unitree_ros2/install/setup.bash 2>/dev/null || true
source ${ROS2_WS}/install/setup.bash 2>/dev/null || true
cd ${ROS2_WS}
echo \\\"[tmux] Launching G1 Sensors: ${LAUNCH_TARGET}...\\\"
ros2 launch ${LAUNCH_TARGET}
echo \\\"[tmux] Process ended. Press enter to exit.\\\"
read
\"'"

echo "[✓] Successfully launched '$LAUNCH_TARGET' in tmux session '$TMUX_SESSION' on $ROBOT_HOST"
echo "To attach: ssh $ROBOT_HOST -t 'tmux attach -t $TMUX_SESSION'"
