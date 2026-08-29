# G1 Human-Following Navigation — Full Stack Plan

**Goal:** LiDAR detection → human selection → Nav2 → locomotion bridge → G1 walks 60 cm in front of selected human.

**Date:** 2026-08-12  
**Stack:** ROS2 Jazzy · Ubuntu 24.04 · Python 3.12 · CUDA 12.6 · PyTorch 2.7

---

## 1. New Packages to Create

| Package | Type | Role |
|---|---|---|
| `g1_description` | ament_python | URDF + robot_state_publisher + static TFs + RViz config |
| `g1_bringup` | ament_python | Launch hub — real/sim/full-stack with one command |
| `livox_detection` | ament_python | Standalone VoxelNeXt detection node using `pt/voxelnext_nuscenes.pth` |
| `g1_nav` | ament_python | SLAM Toolbox + Nav2 launches + G1-tuned params |
| `g1_control` | ament_python | Locomotion bridge + 2 new nodes (see below) |

**Existing packages on robot (do not duplicate):** `lowstate_to_jointstate`, `livox_ros_driver2`, `realsense2_camera` — live in robot's `ros2_ws`. Our bringup includes their launches by path.

---

## 2. End-to-End Data Flow

```
SENSORS
  Mid-360 LiDAR       → /livox/lidar (CustomMsg)         → livox_detection_node
  RealSense D435      → /camera/* topics                  → ccvnorm_node (depth fusion)
  LowState DDS        → lowstate_to_jointstate            → robot_state_publisher → /tf

DETECTION
  livox_detection_node → /g1/detections/livox              → human_selector_node

SELECTION
  human_selector_node  → /g1/selected_human (PoseStamped) → human_follower_node

NAVIGATION
  human_follower_node  → /g1/nav_goal (PoseStamped)       → Nav2 NavigateToPose
  Nav2 cmd_vel         → cmd_vel_bridge (NEW)              → robot_bridge.py (LocoClient) → G1

ODOMETRY
  /unitree/slam_mapping/odom (onboard, 10.8 Hz)           → Nav2 / SLAM Toolbox
  lidar_odometry_node (ICP fallback)                      → /odom + odom→base_link TF
```

> **⚠ Two-process constraint:** `robot_bridge.py` owns LocoClient DDS and must never share a process with rclpy.
> All locomotion commands cross `/tmp/g1_robot_bridge.sock` (Unix socket, JSON).
> Confirmed: Node + LocoClient in same process = segfault 15/15 runs.

---

## 3. Implementation Steps (Ordered)

### Step 1 — `g1_description` package
- `src/g1_description/urdf/g1_29dof.urdf` — copy/symlink from g1pilot
- `launch/description.launch.py` — robot_state_publisher + static TFs:
  - `mid360_link` ↔ `livox_frame` (180° yaw)
  - `base_link` ↔ `pelvis` (identity)
- `config/g1_viz.rviz` — RobotModel + PointCloud2 (`/livox/mid360/points`) + MarkerArray (`/g1/detection_markers/livox`)

**Verify:** `ros2 launch g1_description description.launch.py` → robot skeleton visible in RViz

---

### Step 2 — `g1_bringup` package
- `launch/real.launch.py` — includes:
  - livox_ros_driver2 (config: `MID360_config.json`)
  - realsense2_camera (640×480, align_depth=true)
  - lowstate_to_jointstate
  - g1_description (robot_state_publisher + TFs)
- `launch/sim.launch.py` — Isaac Sim passthrough + g1_description
- `launch/sensors_only.launch.py` — LiDAR + camera only (laptop testing, no joints)
- `launch/full_real.launch.py` — everything: sensors + detection + nav + control (Step 6)

**Mirrors** `/home/unitree/Projects/ros2_ws/src/g1_sensors.launch.py` — consolidates into the ws.

---

### Step 3 — `livox_detection` package
- Extract `livox_detection_node.py` from `g1_perception` (or symlink)
- `setup.py` entry point: `livox_detection_node`
- `launch/livox_detection.launch.py` with args:
  - `checkpoint` (default: `pt/voxelnext_nuscenes.pth` at workspace root)
  - `score_threshold` (default: 0.4)
  - `device` (default: cuda)
  - `input_topic` (default: `/livox/lidar`)

**Publishes:** `/g1/detections/livox` (Detection3DArray) + `/g1/detection_markers/livox` (MarkerArray)

---

### Step 4 — `g1_nav` package
- `launch/slam.launch.py` — SLAM Toolbox online_sync mode
- `launch/navigation.launch.py` — Nav2 with pre-built map + AMCL
- `config/slam_params.yaml` — tuned for Livox Mid-360 sparse scan:
  - `max_laser_range: 40.0`
  - `minimum_travel_distance: 0.1`
  - `minimum_travel_heading: 0.05`
  - `map_update_interval: 2.0`
- `config/nav2_params.yaml` — G1 footprint:
  - Robot footprint: 0.35 × 0.25 m (ellipse)
  - Inflation radius: 0.35 m
  - Controller: DWB
  - Observation source: `/livox/mid360/points` (PointCloud2, no LaserScan conversion needed)

**Odom source:** `/unitree/slam_mapping/odom` — primary (onboard, confirmed 10.8 Hz). `lidar_odometry_node` runs parallel for comparison only.

---

### Step 5 — `g1_control` package

**Move from `g1_perception`:**
- `robot_bridge.py` — plain Python, no rclpy, owns LocoClient DDS connection
- `cmd_pose_bridge.py` — thin rclpy node, socket client

**NEW: `cmd_vel_bridge.py`**
- Subscribes `/cmd_vel` (geometry_msgs/Twist) — Nav2 DWB controller output
- At 10 Hz: send `move` command to robot_bridge over Unix socket
  - `vx = msg.linear.x`, `vy = msg.linear.y`, `wz = msg.angular.z`
  - Short-horizon (0.1 s) — equivalent to MoveWithVelocity semantics
- Same retry pattern as cmd_pose_bridge (3 retries, fail fast on ConnectionRefused)
- Stops robot (sends `stop`) if no `/cmd_vel` received for >0.5 s (safety deadman)

**NEW: `human_follower_node.py`**
- Subscribes `/g1/selected_human` (PoseStamped)
- Computes standoff pose: 60 cm behind human (in human's facing direction)
  - Standoff = human_pos - 0.6 * [cos(yaw), sin(yaw)]
  - Goal orientation = human's yaw (robot faces same direction as human)
- Publishes `/g1/nav_goal` (PoseStamped) → Nav2
- Hysteresis: only republish if human moved >0.15 m or rotated >15° since last goal
- Republish rate: 2 Hz (Nav2 replans each time)

**`launch/control.launch.py`:**
- `cmd_vel_bridge` node
- `human_follower_node` node
- Note in docstring: start `robot_bridge.py` manually in separate terminal first

---

### Step 6 — `g1_bringup/launch/full_real.launch.py`

Single launch for entire stack:
```
ros2 launch g1_bringup full_real.launch.py \
  slam:=true \
  map:=/path/to/map.yaml \
  device:=cuda \
  checkpoint:=$PWD/pt/voxelnext_nuscenes.pth
```

Includes (in order):
1. `g1_bringup/real.launch.py` (sensors + description)
2. `livox_detection/livox_detection.launch.py` (detection)
3. `g1_perception/human_selector.launch.py` (human selection CLI)
4. `g1_nav/slam.launch.py` or `navigation.launch.py` (conditional on `slam` arg)
5. `g1_control/control.launch.py` (follower + cmd_vel bridge)

**Does NOT start `robot_bridge.py`** — that's a manual step (separate terminal, plain python3, no ROS sourced).

---

## 4. Topic Reference (Real Robot)

| Topic | Type | Source | Notes |
|---|---|---|---|
| `/livox/lidar` | `livox_ros_driver2/CustomMsg` | livox_ros_driver2 | Primary detection input |
| `/livox/imu` | `sensor_msgs/Imu` | livox_ros_driver2 | Mid-360 IMU, frame=livox_frame |
| `/camera/color/image_raw` | `sensor_msgs/Image` | realsense2_camera | 640×480 30 Hz RGB |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | realsense2_camera | Intrinsics for ccvnorm |
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | realsense2_camera | 640×480 depth |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | realsense2_camera | Aligned depth |
| `/joint_states` | `sensor_msgs/JointState` | lowstate_to_jointstate | All 29 DoF from DDS |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | robot_state_publisher | Robot skeleton |
| `/unitree/slam_mapping/odom` | `nav_msgs/Odometry` | robot onboard | 10.8 Hz confirmed |
| `/g1/detections/livox` | `vision_msgs/Detection3DArray` | livox_detection_node | Multi-class, filter class_id=1 (pedestrian) |
| `/g1/selected_human` | `geometry_msgs/PoseStamped` | human_selector_node | CLI-selected target |
| `/g1/nav_goal` | `geometry_msgs/PoseStamped` | human_follower_node | 60 cm standoff → Nav2 |
| `/cmd_vel` | `geometry_msgs/Twist` | Nav2 DWB | → cmd_vel_bridge → robot_bridge |

---

## 5. Key Design Decisions

### Nav2 vs direct nav2point.py
Use Nav2 (NavigateToPose) for human-following. Reason: obstacle avoidance free via costmap + DWB, global planner handles non-line-of-sight paths. `nav2point.py` (already written) is a drop-in fallback if Nav2 proves too slow for reactive following.

### cmd_vel → LocoClient translation
Nav2 emits continuous `/cmd_vel`. `robot_bridge.py` protocol uses blocking point-to-point `move` calls. `cmd_vel_bridge.py` converts to short-horizon (0.1 s) moves at 10 Hz — MoveWithVelocity semantics without changing robot_bridge's protocol.

### Odometry source
`/unitree/slam_mapping/odom` (onboard) is primary. `lidar_odometry_node` (ICP) runs parallel for comparison. Do not switch primary source until ICP is validated on actual hardware.

### Human detection class filter
VoxelNeXt (livox_detection) outputs multi-class detections (nuScenes classes). Filter by the `pedestrian` class in `human_selector_node`. Score threshold 0.3 (already set) — Mid-360 sparse scan yields lower confidence than dense LiDAR.

### LiDAR → Nav2 costmap
Nav2 supports PointCloud2 observation sources natively. No LaserScan conversion needed. Point `/livox/mid360/points` directly at the costmap observation layer.

### loco_client.py (old)
Still imports both rclpy + unitree_sdk2py in one process — segfault risk. Do NOT use until rewritten as thin robot_bridge socket client. Leave untouched.

---

## 6. Workspace Layout (After)

```
g1_perception_ws/src/
├── g1_perception/          # existing — keep as-is
├── g1_description/         # NEW
│   ├── g1_description/
│   │   └── __init__.py
│   ├── urdf/
│   │   └── g1_29dof.urdf
│   ├── launch/
│   │   └── description.launch.py
│   ├── config/
│   │   └── g1_viz.rviz
│   ├── package.xml
│   └── setup.py
├── g1_bringup/             # NEW
│   ├── launch/
│   │   ├── real.launch.py
│   │   ├── sim.launch.py
│   │   ├── sensors_only.launch.py
│   │   └── full_real.launch.py
│   ├── package.xml
│   └── setup.py
├── livox_detection/        # NEW
│   ├── livox_detection/
│   │   ├── __init__.py
│   │   └── livox_detection_node.py
│   ├── launch/
│   │   └── livox_detection.launch.py
│   ├── package.xml
│   └── setup.py
├── g1_nav/                 # NEW
│   ├── launch/
│   │   ├── slam.launch.py
│   │   └── navigation.launch.py
│   ├── config/
│   │   ├── slam_params.yaml
│   │   └── nav2_params.yaml
│   ├── package.xml
│   └── setup.py
└── g1_control/             # NEW
    ├── g1_control/
    │   ├── __init__.py
    │   ├── robot_bridge.py      # moved from g1_perception
    │   ├── cmd_pose_bridge.py   # moved
    │   ├── cmd_vel_bridge.py    # NEW
    │   └── human_follower_node.py  # NEW
    ├── launch/
    │   └── control.launch.py
    ├── package.xml
    └── setup.py
```

---

## 7. Build & Run Order

```bash
# Build (laptop)
cd /home/thakk100/Projects/thesis/g1_perception_ws
colcon build --symlink-install
source install/setup.bash

# Terminal 1 — robot_bridge (no rclpy, plain python3, start FIRST)
cd src/g1_control/g1_control
export PATH=/usr/bin:/bin:$PATH
export CYCLONEDDS_HOME=/home/thakk100/cyclonedds/install
python3 robot_bridge.py          # auto-detect interface

# Terminal 2 — full stack
export PATH=/usr/bin:/bin:$PATH
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch g1_bringup full_real.launch.py \
  slam:=true \
  device:=cuda \
  checkpoint:=$PWD/pt/voxelnext_nuscenes.pth

# Terminal 3 — (optional) select human via CLI
# human_selector_node already launched by full_real.launch.py
# It prints a numbered menu when detections arrive — type number + enter to select
```

---

## 8. Status Checklist

- [ ] Step 1: `g1_description` package — URDF + TFs + RViz
- [ ] Step 2: `g1_bringup` package — real/sim/sensors_only launches
- [ ] Step 3: `livox_detection` package — standalone detection node
- [ ] Step 4: `g1_nav` package — SLAM + Nav2 params
- [ ] Step 5: `g1_control` package — `cmd_vel_bridge.py` + `human_follower_node.py`
- [ ] Step 6: `g1_bringup/full_real.launch.py` — wires everything together
- [ ] Verify: robot visible in RViz with live TF from lowstate
- [ ] Verify: detections appear on `/g1/detections/livox` at >3 Hz
- [ ] Verify: human_selector CLI works, `/g1/selected_human` publishes
- [ ] Verify: `human_follower_node` produces valid `/g1/nav_goal`
- [ ] Verify: Nav2 accepts goal and robot walks to standoff position
- [ ] Verify: robot tracks moving human (goal updates with hysteresis)
