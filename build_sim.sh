#!/bin/bash
# Build the Gazebo simulation packages.
# Needed because gz libs are installed as ROS2 vendor packages under
# /opt/ros/jazzy/opt/*, not as system libs, so pkg-config and the linker
# need explicit paths.

set -e
cd "$(dirname "$0")"

source /opt/ros/jazzy/setup.bash

# Expose gz vendor pkg-config files
export PKG_CONFIG_PATH=$(find /opt/ros/jazzy/opt -name "pkgconfig" -type d 2>/dev/null | tr '\n' ':')${PKG_CONFIG_PATH}

# Linker flags for vendor libs (both -L search path and -rpath for runtime)
VENDOR_LDIRS=$(find /opt/ros/jazzy/opt -maxdepth 2 -name "lib" -type d 2>/dev/null)
VENDOR_LDFLAGS=$(echo "$VENDOR_LDIRS" | sed 's/^/-L/' | tr '\n' ' ')
VENDOR_RPATH=$(echo "$VENDOR_LDIRS" | sed 's/^/-Wl,-rpath,/' | tr '\n' ' ')

echo "[build_sim.sh] Building ros2_livox_simulation + g1_bringup ..."

colcon build \
  --packages-select ros2_livox_simulation g1_bringup g1_description \
  --symlink-install \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_EXE_LINKER_FLAGS=${VENDOR_LDFLAGS} ${VENDOR_RPATH}" \
    "-DCMAKE_SHARED_LINKER_FLAGS=${VENDOR_LDFLAGS} ${VENDOR_RPATH}"

echo "[build_sim.sh] Done. Source install/setup.bash then:"
echo "  ros2 launch g1_bringup sim.launch.py headless:=false rviz:=true"
