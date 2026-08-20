#!/usr/bin/env bash
# Record a rosbag2 of G1 sensor data: LiDAR, camera, IMU, joint states.
#
# Works against either the real robot or the Isaac Sim scene — same script,
# different topic sets, selected with --source. Real-robot topic names come
# from Unitree's G1 LiDAR/ROS2 docs and the /lowstate convention used across
# their SDK (unitree_go/unitree_hg msg, not sensor_msgs/Imu or JointState);
# sim topics are whatever g1_rtx_sim.py / rtx_camera.py publish.
#
# Usage:
#   ./record_bag.sh --source real  [--out DIR] [--name NAME] [--duration SEC]
#   ./record_bag.sh --source sim   [--out DIR] [--name NAME] [--duration SEC]
#   ./record_bag.sh --source real --extra /some/other/topic
#
# Requires: `source /opt/ros/jazzy/setup.bash` first (and the workspace's own
# install/setup.bash if you rely on g1_perception's remapped topics).

set -euo pipefail

SOURCE=""
OUT_DIR="$(pwd)/bags"
NAME=""
DURATION=""
EXTRA_TOPICS=()

usage() {
  echo "Usage: $0 --source {real|sim} [--out DIR] [--name NAME] [--duration SEC] [--extra TOPIC]..."
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)   SOURCE="$2"; shift 2 ;;
    --out)      OUT_DIR="$2"; shift 2 ;;
    --name)     NAME="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --extra)    EXTRA_TOPICS+=("$2"); shift 2 ;;
    -h|--help)  usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

if [[ "$SOURCE" != "real" && "$SOURCE" != "sim" ]]; then
  echo "Error: --source must be 'real' or 'sim'"
  usage
fi

if ! command -v ros2 &>/dev/null; then
  echo "Error: ros2 not found. Run 'source /opt/ros/jazzy/setup.bash' first."
  exit 1
fi

# --- Topic sets -------------------------------------------------------

if [[ "$SOURCE" == "real" ]]; then
  # Topics confirmed from `ros2 topic list` on robot at 192.168.123.164 (2026-07-29).
  # Robot runs ROS2 Foxy + rmw_cyclonedds_cpp, ROS_DOMAIN_ID=0.
  # /lowstate and /lf/lowstate carry the full joint state + IMU in one
  # unitree_hg/msg/LowState message (not standard sensor_msgs types).
  # /lf/secondary_imu and /secondary_imu carry raw IMU separately.
  # LiDAR: /unitree/slam_mapping/points is the SLAM-processed cloud.
  # No /utlidar/cloud observed — use /unitree/slam_mapping/points instead.
  TOPICS=(
    # Sensors published by g1_sensors.launch.py on the robot.
    /livox/lidar                      # PointCloud2 - raw Mid-360 sweep, frame_id=mid360_link
    /livox/imu                        # Mid-360 onboard IMU, 200Hz
    /camera/color/image_raw           # D435i colour
    /camera/color/camera_info
    /camera/depth/image_rect_raw      # D435i depth, 16UC1 mm, depth optical frame
    /camera/depth/camera_info
    /camera/aligned_depth_to_color/image_raw   # depth reprojected into the colour frame
    /camera/depth/color/points        # RGB-textured depth cloud (heaviest topic here)
    /camera/imu                       # only present when launched with camera_imu:=true
    /joint_states                     # real encoder angles via lowstate_to_jointstate
    # Unitree's own state, in unitree_hg messages rather than sensor_msgs.
    /lowstate                         # joint angles/vel/torque + IMU in one message
    /secondary_imu                    # raw IMU (accel + gyro)
    /state_estimator/odom_pelvis      # pelvis odometry
    /sportmodestate                   # high-level motion mode state
    /unitree/slam_mapping/points      # SLAM-processed cloud, separate from /livox/lidar
    /tf
    /tf_static
  )
else
  # Isaac Sim topics (g1_rtx_sim.py / rtx_camera.py).
  TOPICS=(
    /livox/mid360/points
    /g1/camera/rgb
    /g1/camera/depth
    /g1/camera/semantic
    /g1/camera/camera_info
    /g1/joint_states
    /tf
    /tf_static
    /clock
  )
fi

TOPICS+=("${EXTRA_TOPICS[@]}")

# --- Output path --------------------------------------------------------

mkdir -p "$OUT_DIR"
if [[ -z "$NAME" ]]; then
  NAME="g1_${SOURCE}_$(date +%Y%m%d_%H%M%S)"
fi
BAG_PATH="$OUT_DIR/$NAME"

echo "[record_bag] source     : $SOURCE"
echo "[record_bag] output     : $BAG_PATH"
echo "[record_bag] topics     :"
for t in "${TOPICS[@]}"; do echo "  $t"; done

CMD=(ros2 bag record -o "$BAG_PATH")
if [[ "$SOURCE" == "sim" ]]; then
  # Sim publishes /clock; rosbag2 should use_sim_time so timestamps line up
  # with the recorded /clock rather than wall time.
  CMD+=(--use-sim-time)
fi
if [[ -n "$DURATION" ]]; then
  CMD+=(--max-bag-duration "$DURATION")
fi
CMD+=("${TOPICS[@]}")

echo "[record_bag] running: ${CMD[*]}"
exec "${CMD[@]}"
