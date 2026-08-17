#!/usr/bin/env bash
# Check DDS config matches between laptop and robot.
# Both must have same ROS_DOMAIN_ID to see each other's topics.

ROBOT="unitree@ubuntu.local"

echo "=== LAPTOP ==="
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0 (default)}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp (default)}"
echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-not set}"

echo ""
echo "=== ROBOT ==="
ssh "$ROBOT" "echo ROS_DOMAIN_ID=\${ROS_DOMAIN_ID:-0} && \
    echo RMW=\${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp} && \
    echo ROS_LOCALHOST_ONLY=\${ROS_LOCALHOST_ONLY:-not set} && \
    echo CYCLONEDDS_URI=\${CYCLONEDDS_URI:-not set}"

echo ""
echo "=== Verify laptop sees robot topics ==="
echo "Run: ros2 topic hz /livox/lidar"
echo "If empty: ROS_DOMAIN_ID mismatch or firewall blocking UDP multicast"
echo "Fix:  export ROS_DOMAIN_ID=<robot_value>"
echo "      sudo ufw allow in on <iface> from 192.168.123.0/24"
