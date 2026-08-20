#!/usr/bin/env bash
# Helper script to start, stop, or check status of the G1 sensor stack on the robot (unitree@ubuntu.local).
#
# Usage:
#   ./scripts/robot_sensors_remote.sh start    # Starts G1 sensor launch (Mid-360 + D435 + TFs)
#   ./scripts/robot_sensors_remote.sh stop     # Stops remote sensor launch
#   ./scripts/robot_sensors_remote.sh status   # Checks if sensors and ROS2 topics are active
#   ./scripts/robot_sensors_remote.sh logs     # Tails logs of remote sensor node

# mDNS (ubuntu.local) does not resolve on the laptop -- use the wired IP.
# Override with:  ROBOT_HOST=unitree@other-host ./scripts/robot_sensors_remote.sh ...
ROBOT_HOST="${ROBOT_HOST:-unitree@192.168.123.164}"
ACTION="${1:-status}"

case "$ACTION" in
  start)
    echo "[*] Connecting to $ROBOT_HOST to start G1 sensor stack..."
    ssh -o ConnectTimeout=4 "$ROBOT_HOST" "nohup bash /home/unitree/Projects/ros2_ws/scripts/run_g1_sensors.sh > /tmp/g1_sensors.log 2>&1 &"
    echo "[*] Sensor stack launch command dispatched. Waiting 3 seconds..."
    sleep 3
    "$0" status
    ;;

  stop)
    echo "[*] Stopping G1 sensor stack on $ROBOT_HOST..."
    ssh -o ConnectTimeout=4 "$ROBOT_HOST" "pkill -f g1_sensors.launch.py || true; pkill -f livox_ros_driver2_node || true; pkill -f realsense2_camera || true"
    echo "[✓] Stopped."
    ;;

  status)
    echo "=== G1 Robot Sensor Status ($ROBOT_HOST) ==="
    if ping -c 1 -W 2 ubuntu.local >/dev/null 2>&1; then
      echo "  [✓] Network: Robot is REACHABLE"
      ssh -o ConnectTimeout=3 "$ROBOT_HOST" "ps aux | grep -E 'g1_sensors|livox_ros_driver2|robot_state_publisher' | grep -v grep || echo '  [-] No active sensor processes found on robot.'"
    else
      echo "  [✗] Network: Robot host 'ubuntu.local' is NOT reachable."
    fi
    ;;

  logs)
    ssh "$ROBOT_HOST" "tail -f /tmp/g1_sensors.log"
    ;;

  *)
    echo "Usage: $0 {start|stop|status|logs}"
    exit 1
    ;;
esac
