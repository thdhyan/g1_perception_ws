#!/usr/bin/env bash
# Record EVERY ROS2 topic currently discoverable on the network into one
# rosbag2 — tf, joint_states, lowstate/highstate, cmd_vel, LiDAR, cameras,
# whatever else is being published. No topic list to maintain: `ros2 bag
# record -a` subscribes to everything discovered at start AND anything that
# appears later (e.g. a driver that comes up after you hit record).
#
# Usage:
#   source /opt/ros/foxy/setup.bash        # match the robot's ROS2 distro
#   ./scripts/record_all.sh                # discover + record, Ctrl-C to stop
#   ./scripts/record_all.sh --name test1
#   ./scripts/record_all.sh --out ~/bags --wait 5
#   ./scripts/record_all.sh --duration 300 # auto-stop after 300s
#   ./scripts/record_all.sh --hidden       # also record hidden (_-prefixed) topics
#
# Requires: on the same network/ROS_DOMAIN_ID as the robot (plug into
# ethernet, robot is 192.168.123.164 per robot_sensors_remote.sh) and
# `ros2` sourced. Playback with:
#   ros2 bag play <bag_dir>

set -euo pipefail

OUT_DIR="$(pwd)/bags"
NAME=""
WAIT_SEC=3
STORAGE="sqlite3"   # built into every ROS2 install, no extra deps needed
DURATION=""
HIDDEN=0

usage() {
  echo "Usage: $0 [--out DIR] [--name NAME] [--wait SEC] [--storage mcap|sqlite3] [--duration SEC] [--hidden]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)      OUT_DIR="$2"; shift 2 ;;
    --name)     NAME="$2"; shift 2 ;;
    --wait)     WAIT_SEC="$2"; shift 2 ;;
    --storage)  STORAGE="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --hidden)   HIDDEN=1; shift ;;
    -h|--help)  usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

if ! command -v ros2 &>/dev/null; then
  echo "Error: ros2 not found. Run 'source /opt/ros/<distro>/setup.bash' first."
  exit 1
fi

# --- Give the network a moment (robot may still be bringing sensors up) -----
echo "[record_all] waiting ${WAIT_SEC}s for topics to appear on the network..."
sleep "$WAIT_SEC"

N_TOPICS=$(ros2 topic list 2>/dev/null | wc -l)
if [[ "$N_TOPICS" -eq 0 ]]; then
  echo "Error: no topics discovered. Check ethernet/ROS_DOMAIN_ID/RMW and that the robot is publishing."
  echo "  (see scripts/robot_sensors_remote.sh status)"
  exit 1
fi

echo "[record_all] discovered $N_TOPICS topics:"
ros2 topic list | sed 's/^/  /'

# --- Output path --------------------------------------------------------
mkdir -p "$OUT_DIR"
if [[ -z "$NAME" ]]; then
  NAME="g1_$(date +%Y%m%d_%H%M%S)"
fi
BAG_PATH="$OUT_DIR/$NAME"

CMD=(ros2 bag record -a -o "$BAG_PATH" --storage "$STORAGE")
[[ "$HIDDEN" -eq 1 ]]     && CMD+=(--include-hidden-topics)
[[ -n "$DURATION" ]]      && CMD+=(--max-bag-duration "$DURATION")

echo "[record_all] storage    : $STORAGE"
echo "[record_all] output     : $BAG_PATH"
echo "[record_all] running: ${CMD[*]}"
echo "[record_all] Ctrl-C to stop recording."
echo
exec "${CMD[@]}"
