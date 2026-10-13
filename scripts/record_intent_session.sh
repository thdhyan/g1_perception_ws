#!/usr/bin/env bash
# Record an intent / PhaseSpace session for later off-robot, off-PhaseSpace testing,
# and (optionally) run VoxelNeXt live detection so its output is in the bag too.
#
#   source setup_g1_env.sh && source install/setup.bash
#   ./scripts/record_intent_session.sh                       # detection + record, Ctrl-C stops both
#   ./scripts/record_intent_session.sh --name lab_take1 --out /Storage/bags
#   ./scripts/record_intent_session.sh --no-detect           # record only
#   ./scripts/record_intent_session.sh --no-images           # skip raw camera images (~45 MB/s)
#
# Records (sqlite3, readable by offline_pipeline/bag_processor.py):
#   robot:      /livox/lidar /livox/imu /utlidar/cloud /joint_states /lowstate /tf /tf_static
#               /camera/color/image_raw /camera/depth/image_rect_raw /camera/*/camera_info
#   phasespace: /phasespace/* /phasespace_* (groups, humans, robot head, LEDs, rigids)
#   labeler:    /mocap/agents      intent: /intent/*      detection: /g1/* /livox/mid360/points
#
# Replay later with the robot and PhaseSpace off:
#   ros2 bag play <bag> --clock         (the labeler / review GUI run on the replayed TF)
set -euo pipefail

OUT_DIR="$(pwd)/bags"
NAME="intent_$(date +%Y%m%d_%H%M%S)"
DETECT=1
IMAGES=1
SPLIT_GB=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)       OUT_DIR="$2"; shift 2 ;;
    --name)      NAME="$2"; shift 2 ;;
    --no-detect) DETECT=0; shift ;;
    --no-images) IMAGES=0; shift ;;
    *) echo "usage: $0 [--out DIR] [--name NAME] [--no-detect] [--no-images]"; exit 1 ;;
  esac
done

command -v ros2 >/dev/null || { echo "source setup_g1_env.sh and install/setup.bash first"; exit 1; }
mkdir -p "$OUT_DIR"
FREE_GB=$(df -BG --output=avail "$OUT_DIR" | tail -1 | tr -dc 0-9)
echo "[record] $OUT_DIR has ${FREE_GB} GB free"
if (( IMAGES )) && (( FREE_GB < 50 )); then
  echo "[record] WARNING: raw images ~45 MB/s + LiDAR ~3 MB/s => ${FREE_GB} GB lasts ~$(( FREE_GB * 1000 / 48 / 60 )) min."
  echo "         Use --out on a bigger disk or --no-images."
fi

TOPICS=(/livox/lidar /livox/imu /utlidar/cloud /joint_states /lowstate /tf /tf_static
        /mocap/agents /livox/mid360/points)
(( IMAGES )) && TOPICS+=(/camera/color/image_raw /camera/depth/image_rect_raw)
REGEX='^/(phasespace.*|intent/.*|g1/.*|camera/.*/camera_info)$'

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done; wait; }
trap cleanup EXIT INT TERM

if (( DETECT )); then
  echo "[record] starting VoxelNeXt live detection (follow:=false, perception only)"
  ros2 launch g1_bringup real_live_detection.launch.py algorithm:=voxelnext follow:=false rviz:=false \
    > "$OUT_DIR/${NAME}_detection.log" 2>&1 &
  PIDS+=($!)
  sleep 5
fi

echo "[record] -> $OUT_DIR/$NAME   (Ctrl-C to stop)"
ros2 bag record -o "$OUT_DIR/$NAME" --storage sqlite3 --max-bag-size $(( SPLIT_GB * 1024 * 1024 * 1024 )) \
  -e "$REGEX" --topics "${TOPICS[@]}" &
PIDS+=($!)
wait
