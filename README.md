# G1 Perception Workspace

ROS2 Jazzy (laptop) + ROS Foxy (robot) full-stack perception, SLAM, navigation, and control pipeline for Unitree G1 humanoid — Livox Mid-360 LiDAR + RealSense D435I + CenterPoint 3D detection + Nav2 + human follower.

**Laptop environment:**

- Ubuntu 24.04, Python 3.12, CUDA 12.6, PyTorch 2.7
- ROS2 Jazzy

**Robot (192.168.123.164):**

- Ubuntu 20.04, Python 3.8
- ROS2 Foxy + CycloneDDS (`rmw_cyclonedds_cpp`)
- Jetson Orin-class ARM, CUDA 11.4

---

## Workspace Layout

```
g1_perception_ws/
├── src/
│   ├── g1_perception/          # Original perception nodes (lidar_bridge, ccvnorm, etc.)
│   ├── g1_bringup/             # Launch files: real.launch, sim.launch, full_real.launch
│   ├── g1_description/         # URDF + RSP + RViz config
│   ├── g1_nav/                 # SLAM Toolbox + Nav2 configs
│   ├── g1_control/             # cmd_vel_bridge, human_follower, robot_bridge
│   ├── g1_wbc/                 # GR00T ONNX whole-body balance/walk policy (sim)
│   ├── livox_detection/        # VoxelNeXt/CenterPoint 3D detection node
│   └── plain_slam_ros2/        # CPU LIO + pose-graph SLAM (submodule, sim "3d" slam_type)
├── isaac_ros_ws/                # Isaac ROS (cuVSLAM + nvblox) — own CUDA 13 Docker env, see below
│   └── src/                    # isaac_ros_common/visual_slam/nvblox/image_segmentation/nitros (submodules)
├── PLAN.md                     # Architecture plan (8 steps)
├── TASKS.md                    # Task dependency graph + status
├── HANDOFF.md                  # Session-to-session handoff notes
└── README.md                   # This file
```

---

## Quick-Start: Real Robot

### 1. On robot — start sensors

```bash
# SSH to robot
ssh unitree@192.168.123.164

# Start all sensors (staggered — avoids DDS OOM kill)
bash /home/unitree/Projects/g1_start_sensors.sh
# Log: /tmp/ros_sensors.log
```

Publishes (ROS_DOMAIN_ID=1):
| Topic | Type | Hz |
|---|---|---|
| `/livox/lidar` | PointCloud2 | 10 |
| `/livox/imu` | Imu | 200 |
| `/camera/color/image_raw` | Image (640×480) | 30 |
| `/camera/depth/image_rect_raw` | Image (640×480) | 30 |
| `/camera/depth/color/points` | PointCloud2 XYZRGB | ~6 |
| `/camera/imu` | Imu (accel+gyro fused) | 200 |
| `/tf_static` | sensor frames → URDF | – |

### 2. On laptop — visualize

```bash
cd /home/thakk100/Projects/thesis/g1_perception_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1   # must match robot

ros2 launch g1_description description.launch.py rviz:=true
```

RViz shows: robot model, TF tree, LiDAR PCL, camera RGB, camera depth, detection markers, nav goal, odom path.

### 3. On laptop — run full perception + navigation

```bash
export ROS_DOMAIN_ID=1
ros2 launch g1_bringup full_real.launch.py slam:=true
```

---

## Build (laptop)

```bash
cd /home/thakk100/Projects/thesis/g1_perception_ws
colcon build --symlink-install
source install/setup.bash
```

Packages built: `g1_description`, `g1_bringup`, `g1_control`, `g1_nav`, `livox_detection`, `g1_perception`

---

## Build (robot side)

The robot runs `g1_sensors.launch.py` directly from source — no colcon build needed for the launch file. The packages that run as ROS2 nodes on the robot (`livox_ros_driver2`, `realsense2_camera`, `robot_state_publisher`, `lowstate_to_jointstate`) are already built in `/home/unitree/Projects/ros2_ws/`.

If you make changes to those packages on the robot:

```bash
ssh unitree@192.168.123.164
cd /home/unitree/Projects/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

The `g1_sensors.launch.py` and `g1_start_sensors.sh` are Python scripts, not compiled — changes take effect immediately after `scp`.

---

## G1 Startup Nodes

What each launch entrypoint actually brings up, process by process.

### Sim — `g1_bringup sim.launch.py` / `sim_teleop.launch.py`

| Process                                           | package / executable                         | Role                                                                                                                             |
| ------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `gz sim`                                          | `ExecuteProcess` (Gazebo Harmonic)           | Physics + rendering, loads `g1_warehouse.sdf`                                                                                    |
| `robot_state_publisher`                           | `robot_state_publisher`                      | URDF → TF for all links                                                                                                          |
| spawn                                             | `ros_gz_sim create`                          | Spawns URDF into Gazebo                                                                                                          |
| `parameter_bridge`                                | `ros_gz_bridge`                              | Gazebo ↔ ROS2 topic bridge (joint_states, clock, lidar, RGB, depth, camera_info)                                                 |
| `tf_mid360_to_livox` … `tf_d435_to_depth_optical` | `tf2_ros static_transform_publisher` ×8      | Fixed frames + `map→odom`/`odom→pelvis` identity fallback (overridden once SLAM publishes)                                       |
| `livox_detection_node`                            | `livox_detection`                            | VoxelNeXt human detection, `detection:=true`                                                                                     |
| `g1_wbc_node`                                     | `g1_wbc`                                     | GR00T ONNX balance/walk policy, 50Hz                                                                                             |
| `lio_3d_node` + `slam_3d_node`                    | `plain_slam_ros2`                            | LIO + pose-graph SLAM, `slam_type:=3d` (default)                                                                                 |
| _(planned)_ Isaac ROS container                   | `isaac_ros_visual_slam` + `isaac_ros_nvblox` | cuVSLAM + nvblox, `slam_type:=isaac` — see [Isaac ROS setup](#isaac-ros-cuvslam--nvblox--human-segmentation--docker-setup) above |
| `pointcloud_to_laserscan`                         | `pointcloud_to_laserscan`                    | 3D → 2D scan for `slam_type:=2d` (slam_toolbox)                                                                                  |
| `g1_autonomous_mapper`                            | `g1_nav`                                     | `sim.launch.py` only — autonomous exploration                                                                                    |
| `rviz2_mapping` / `rviz2_ground_truth`            | `rviz2` ×2                                   | `sim_teleop.launch.py` — separate SLAM (`map` fixed frame) and ground-truth (`warehouse` fixed frame) views                      |

### Real robot — `g1_sensors.launch.py` (on robot) + `g1_bringup full_real.launch.py` (laptop)

| Process                              | package / executable                            | Role                                                                       |
| ------------------------------------ | ----------------------------------------------- | -------------------------------------------------------------------------- |
| `livox_ros_driver2_node`             | `livox_ros_driver2`                             | Mid-360 → `/livox/lidar` PointCloud2, 10Hz                                 |
| `realsense2_camera_node`             | `realsense2_camera` (in `g1_sensors.launch.py`) | D435I RGB + depth + PCL2 + IMU                                             |
| `lowstate_to_jointstate_node`        | `lowstate_to_jointstate`                        | Unitree DDS lowstate → `/joint_states`                                     |
| `robot_state_publisher`              | `robot_state_publisher`                         | URDF → TF                                                                  |
| 4× static TF                         | `tf2_ros static_transform_publisher`            | mid360→livox_frame, base_link→pelvis, d435_link→color/depth optical        |
| `robot_bridge.py`                    | `g1_control` (plain Python, no rclpy)           | Owns `LocoClient` DDS — **two-process constraint**, see `g1_control` above |
| `cmd_vel_bridge` / `cmd_pose_bridge` | `g1_control`                                    | rclpy ↔ Unix socket → robot_bridge                                         |
| `human_follower_node`                | `g1_control`                                    | `/g1/selected_human` → nav goal                                            |
| SLAM/Nav2                            | `g1_nav`, `slam_toolbox`                        | `full_real.launch.py slam:=true`                                           |

Prefer `g1_start_sensors.sh` over launching `g1_sensors.launch.py` directly —
staggers node startup 4–6s apart to avoid the DDS discovery OOM (see
[Known Issues](#dds-discovery-oom-on-robot-critical)).

---

## Packages

### `g1_description`

URDF robot model, TF publishers, RViz config.

- `launch/description.launch.py` — robot_state_publisher + 2 static TFs + optional RViz2
- `urdf/g1_29dof.urdf` — copied from robot at `/home/unitree/Projects/g1pilot/description_files/urdf/g1_29dof.urdf`
- `config/g1_viz.rviz` — RobotModel, TF, LiDAR, IMU, camera RGB/depth, detections, nav goal, odom path

Static TFs published:

- `mid360_link` → `livox_frame` (180° yaw, so X points forward)
- `base_link` → `pelvis` (identity; `pelvis` is URDF root)

### `g1_bringup`

Launch orchestration.

- `launch/real.launch.py` — livox_ros_driver2 + realsense + lowstate_to_jointstate + description
- `launch/sim.launch.py` — Gazebo Harmonic + URDF preprocessing + spawn + ros_gz_bridge + TFs
- `launch/full_real.launch.py` — wires all packages (slam/nav, detection, control)
- `worlds/g1_warehouse.sdf` — Gazebo warehouse world
- `config/gz_bridge.yaml` — 6 topic mappings (joint_states, clock, lidar, RGB, depth, camera_info)

### `g1_control`

Robot control nodes.

| Node                     | Role                                                                   |
| ------------------------ | ---------------------------------------------------------------------- |
| `robot_bridge.py`        | Plain Python (no rclpy), owns `LocoClient` DDS, listens on Unix socket |
| `cmd_pose_bridge.py`     | rclpy node, one-shot relative moves via socket                         |
| `cmd_vel_bridge.py`      | rclpy node, `/cmd_vel` → robot_bridge at 10Hz, 0.5s deadman            |
| `human_follower_node.py` | `/g1/selected_human` → 60cm standoff → `/g1/nav_goal`, 2Hz             |

**Two-process constraint**: `LocoClient` (unitree_sdk2py) + rclpy **cannot coexist in one process** — segfaults 15/15 times. All control goes via Unix socket `/tmp/g1_robot_bridge.sock`.

### `g1_nav`

Navigation stack configs.

- `config/slam_params.yaml` — SLAM Toolbox online_async, max_laser_range 40m, scan_topic `/livox/mid360/points`
- `config/nav2_params.yaml` — G1 footprint 0.35×0.25m, inflation 0.35m, DWB controller, PointCloud2 obstacle layer

### `livox_detection`

Standalone CenterPoint 3D detection from LiDAR PointCloud2.

- `livox_detection/livox_detection_node.py` — subscribes `/livox/lidar`, publishes `/g1/detections/centerpoint` + `/g1/detection_markers/centerpoint`
- `launch/livox_detection.launch.py` — args: `checkpoint_path`, `score_threshold` (0.4), `device` (cuda), `max_hz` (5.0)
- Default checkpoint: `/home/thakk100/Projects/Thesis/livox_detection/livoxdetection/livox_model_1.pt`

---

## Isaac ROS (cuVSLAM + nvblox + human segmentation) — Docker setup

GPU-accelerated SLAM stack, added as a comparison alternative to `plain_slam_ros2`
(item T4 — CPU-only, no CUDA path). Runs in its own Docker container with its
own CUDA 13 runtime, side by side with the rest of the stack which stays on the
host's CUDA 12.0 toolkit — **no host CUDA/driver changes**, containers cross
into the native ROS2 Jazzy DDS graph via `--network host`.

**Why a container:** NVIDIA's Isaac ROS apt packages hard-depend on
`cuda-toolkit-13-0` (confirmed via `apt-get install --simulate
ros-jazzy-isaac-ros-visual-slam` — unmet dependency against this host's CUDA
12.0). Installing natively would break the VoxelNeXt/torch/spconv detection
stack, which is pinned to cu121. Isolating in a container sidesteps that
entirely.

**Why RGBD mode, not stereo:** `g1_29dof.urdf`'s D435i is a single Gazebo
`rgbd_camera` sensor — no `infra1`/`infra2` stereo pair exists in sim. cuVSLAM
RGBD mode reuses the existing `/camera/color/image_raw`,
`/camera/depth/image_rect_raw`, `/camera/color/camera_info` topics unmodified.

### One-time host setup (already done on this machine, kept here for a fresh clone)

```bash
# 1. Add the Isaac ROS apt repo (Ubuntu 24.04 "noble" x86_64)
sudo apt install curl gnupg software-properties-common
sudo add-apt-repository -y universe
k="/usr/share/keyrings/nvidia-isaac-ros.gpg"
curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key | sudo gpg --dearmor | sudo tee -a $k > /dev/null
f="/etc/apt/sources.list.d/nvidia-isaac-ros.list"
sudo touch $f
echo "deb [signed-by=$k] https://isaac.download.nvidia.com/isaac-ros/release-4.5 noble main" | sudo tee -a $f
sudo apt-get update

# 2. Install the Isaac ROS CLI (replaces the old hand-rolled run_dev.sh workflow)
sudo apt-get install -y isaac-ros-cli
sudo isaac-ros init docker --yes

# 3. Point the CLI at this workspace's isaac_ros_ws (not ~/workspaces — kept in-repo)
echo 'export ISAAC_ROS_WS="${ISAAC_ROS_WS:-$HOME/Projects/thesis/g1_perception_ws/isaac_ros_ws}"' >> ~/.bashrc
source ~/.bashrc
```

`nvidia-container-toolkit` must also be installed and passing
`docker run --rm --gpus all nvidia/cuda:...-base-ubuntu24.04 nvidia-smi` — was
already present/verified on this machine.

### Workspace layout

`isaac_ros_ws/src/` holds 5 packages as git submodules, pointed at upstream
`NVIDIA-ISAAC-ROS/*` (forks exist at `github.com/thdhyan/*` for all five, but
`origin` stays on upstream — only re-pointed to the fork if we actually patch
one, same convention as `livox_laser_simulation_RO2`/`Ultra-Fusion`):

| Submodule                      | Role                                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------- |
| `isaac_ros_common`             | Docker/devcontainer scaffolding consumed by `isaac-ros-cli`                                       |
| `isaac_ros_visual_slam`        | cuVSLAM — GPU stereo/RGBD VIO                                                                     |
| `isaac_ros_nvblox`             | GPU 3D reconstruction (mesh + ESDF) from depth + pose                                             |
| `isaac_ros_image_segmentation` | U-Net decoder — backs PeopleSemSegNet human segmentation, masks people out of nvblox's static map |
| `isaac_ros_nitros`             | Zero-copy GPU tensor/image transport the above depend on                                          |

The actual ROS packages (`ros-jazzy-isaac-ros-visual-slam`,
`ros-jazzy-isaac-ros-nvblox`, `ros-jazzy-isaac-ros-unet`,
`ros-jazzy-isaac-ros-peoplesemseg-models-install`, …) install as **apt debs
inside the container** — the base `isaac_ros` container image already has the
CUDA 13 toolkit and this same apt repo baked in. The submodules above are for
reading launch files/params and for local patches, not for a from-source
colcon build.

### Build / activate the container

```bash
export ISAAC_ROS_WS=/home/thakk100/Projects/thesis/g1_perception_ws/isaac_ros_ws
isaac-ros activate --build-local --build-only   # first time: pulls/builds the image
isaac-ros activate                               # enters the dev container
```

Inside the container:

```bash
sudo apt-get update
sudo apt-get install -y ros-jazzy-isaac-ros-visual-slam ros-jazzy-isaac-ros-nvblox \
    ros-jazzy-isaac-ros-unet ros-jazzy-isaac-ros-peoplesemseg-models-install
```

### Running alongside `plain_slam_ros2`

Isaac ROS publishes to **isolated frame names** (`vslam_odom`/`vslam_map`, not
`odom`/`map`) so `slam_type:=isaac` can run _simultaneously_ with
`slam_type:=3d` (plain_slam_ros2) for direct comparison — same TF rule
established earlier in this project: never publish the same parent/child pair
from two independent sources.

```bash
ros2 launch g1_bringup sim_teleop.launch.py slam_type:=isaac rviz:=true
```

**VRAM note:** RTX 4060 Laptop, 8188 MiB — right at nvblox's stated minimum,
shared with Gazebo rendering + WBC ONNX + VoxelNeXt CUDA all running natively
alongside the containerized Isaac ROS stack. Check `nvidia-smi` with detection
on before trusting a full concurrent run.

---

## Robot Sensor Launch

The file `/home/unitree/Projects/ros2_ws/src/g1_sensors.launch.py` (deployed to robot) runs:

- Livox Mid-360 via `livox_ros_driver2_node` — PointCloud2 format, 10Hz
- RealSense D435I via `realsense2_camera_node` — RGB + depth + PCL2 + IMU (no IR stereo)
- `robot_state_publisher` with URDF from `/home/unitree/Projects/g1pilot/description_files/urdf/g1_29dof.urdf`
- 4 static TF publishers: mid360→livox_frame, base_link→pelvis, d435_link→camera_color_optical_frame, d435_link→camera_depth_optical_frame

**Using the shell script instead** (`g1_start_sensors.sh`) is recommended — it staggers node startup by 4-6s each to avoid simultaneous DDS discovery memory spikes.

Camera streams (D435I hardware constraints):

- ✅ RGB 640×480 30Hz
- ✅ Depth 640×480 30Hz
- ✅ Aligned depth to color
- ✅ PointCloud2 XYZRGB ~6Hz
- ✅ IMU (gyro + accel fused, `unite_imu_method=2`)
- ❌ IR1/IR2 disabled — 5 simultaneous USB streams exhausts ARM USB frame buffers

---

## Known Issues & Workarounds

### DDS Discovery OOM on robot (CRITICAL)

**Symptom**: Nodes start then get SIGKILL'd (exit code -9) seconds after launch. Memory spikes from ~2GB to 14GB during startup, then drops back after kill.

**Root cause**: Robot has 280+ unitree SDK DDS entities on domain 0. Every new ROS2 participant probes participant indices 0..N, causing a VMS spike proportional to N (30GB+ VSZ). On this 15GB Jetson platform, the kernel OOM killer fires.

**Fix**:

1. Use `g1_start_sensors.sh` (staggered startup — each node waits 4-6s for previous to finish DDS discovery)
2. Export `CYCLONEDDS_URI=/home/unitree/Projects/cyclone_sensors.xml` (caps `MaxAutoParticipantIndex=4`)
3. Export `ROS_DOMAIN_ID=1` (isolates sensor nodes from unitree SDK on domain 0)
4. **Laptop must also use `ROS_DOMAIN_ID=1`** to see robot sensor topics

### Camera HID IMU deadlock

**Symptom**: `realsense2_camera_node` killed after ~10s. Log: `iio_hid_sensor: Frames didn't arrived`

**Fix**: Camera IMU (HID) disabled. Use `enable_gyro:=false enable_accel:=false`. The D435I on this robot hardware (likely Jetson Orin) has a known `iio_hid_sensor` deadlock. Note: `unite_imu_method` must be integer (0/1/2), not string.

**Update**: Re-enabled with `enable_gyro:=true enable_accel:=true unite_imu_method:=2` — monitor for deadlock.

### Camera USB frame exhaustion

**Symptom**: `Out of frame resources` error, camera dies.

**Cause**: 5 simultaneous streams (IR1+IR2+depth+color+align) exceeds USB frame buffer capacity on ARM.

**Fix**: Disable IR stereo (`enable_infra1:=false enable_infra2:=false`). Max 4 streams (RGB + depth + aligned + PCL2).

### Camera `Device or resource busy` (VIDIOC_S_FMT)

**Symptom**: Camera fails to open immediately after a crash or restart.

**Fix**: USB power-cycle:

```bash
echo '123' | sudo -S bash -c "echo '2-3' > /sys/bus/usb/drivers/usb/unbind && sleep 2 && echo '2-3' > /sys/bus/usb/drivers/usb/bind"
```

### TF: `camera_color_optical_frame` not in tree

**Symptom**: RViz error "Could not transform from camera_color_optical_frame to pelvis".

**Fix**: Static TF `d435_link → camera_color_optical_frame` with optical-frame rotation `(roll=-1.5708, yaw=-1.5708)`. Already in `g1_sensors.launch.py` and `g1_start_sensors.sh`. RSP chains `d435_link → torso_link → ... → pelvis` via URDF.

### SSH drops when killing processes

**Symptom**: `pkill -f` kills SSH session. `kill` also drops if PID chain matches.

**Fix**: Use targeted PIDs: `kill $(pgrep -f realsense2_camera_node)` (not `pkill -f`). Run each kill in a separate SSH command.

### `bad_alloc` spam in logs

Not real OOM — it's DDS discovery buffer allocation failures from the 280+ entity flood. Cosmetic. Nodes survive past the spam if staggered properly.

### `unite_imu_method` parameter type error

`"unite_imu_method": "none"` fails — expects integer. Use `2` (linear_interpolation), not `"linear_interpolation"`.

---

## DDS / Domain Setup

| Component                          | Domain | CycloneDDS URI                               |
| ---------------------------------- | ------ | -------------------------------------------- |
| Unitree SDK (robot permanent)      | 0      | `/home/unitree/cyclonedds_ws/cyclonedds.xml` |
| Sensor nodes (g1_start_sensors.sh) | 1      | `/home/unitree/Projects/cyclone_sensors.xml` |
| Laptop (perception/nav)            | 1      | default (Jazzy)                              |

Set on laptop before running any ros2 commands:

```bash
export ROS_DOMAIN_ID=1
```

Or add to `~/.bashrc`:

```bash
echo 'export ROS_DOMAIN_ID=1' >> ~/.bashrc
```

---

## TF Tree

```
map
└── odom
    └── base_link ← (identity static TF) ← pelvis
                                              └── torso_link (URDF)
                                                  ├── d435_link
                                                  │   ├── camera_color_optical_frame (static TF)
                                                  │   └── camera_depth_optical_frame (static TF)
                                                  └── mid360_link
                                                      └── livox_frame (static TF, 180° yaw)
```

---

## Robot Control (two-process constraint)

```
[cmd_vel_bridge.py] ─── Unix socket /tmp/g1_robot_bridge.sock ──> [robot_bridge.py] ─── DDS ──> G1
   (rclpy node)                                                       (no rclpy, owns LocoClient)
```

**rclpy + LocoClient in one process = segfault 100% of the time.** Architecture enforces isolation.

### Running robot control

```bash
# Terminal 1: start robot bridge (plain python, no ros)
cd src/g1_control/g1_control
python3 robot_bridge.py

# Terminal 2: ROS2 cmd_vel subscriber
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 run g1_control cmd_vel_bridge

# Send a velocity command
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.0}, angular: {z: 0.0}}" --once
```

---

## Topic Table (full stack)

| Topic                               | Type             | Publisher                 | Subscriber      |
| ----------------------------------- | ---------------- | ------------------------- | --------------- |
| `/livox/lidar`                      | PointCloud2      | livox_ros_driver2 (robot) | livox_detection |
| `/livox/imu`                        | Imu              | livox_ros_driver2 (robot) | slam_toolbox    |
| `/camera/color/image_raw`           | Image            | realsense2_camera (robot) | RViz, ccvnorm   |
| `/camera/depth/image_rect_raw`      | Image            | realsense2_camera (robot) | ccvnorm         |
| `/camera/depth/color/points`        | PointCloud2      | realsense2_camera (robot) | Nav2 costmap    |
| `/camera/imu`                       | Imu              | realsense2_camera (robot) | –               |
| `/g1/detections/centerpoint`        | Detection3DArray | livox_detection_node      | human_selector  |
| `/g1/detection_markers/centerpoint` | MarkerArray      | livox_detection_node      | RViz            |
| `/g1/selected_human`                | Detection3D      | human_selector_node       | human_follower  |
| `/g1/nav_goal`                      | PoseStamped      | human_follower_node       | Nav2            |
| `/cmd_vel`                          | Twist            | Nav2 DWB controller       | cmd_vel_bridge  |

---

## Status

- [x] **T1** `g1_description` — URDF + RSP + TFs + RViz config
- [x] **T2** `g1_bringup` — real.launch, sim.launch, full_real.launch
- [x] **T3** `livox_detection` — CenterPoint node, launch file
- [x] **T4** `g1_nav` — SLAM Toolbox + Nav2 params
- [x] **T5** `g1_control` — robot_bridge, cmd_vel_bridge, human_follower
- [x] **T6** `full_real.launch.py` — all packages wired
- [x] **T7** `sim.launch.py` — Gazebo Harmonic warehouse world
- [ ] **T8** End-to-end verification (blocked: DDS OOM issue on robot being resolved)

### Active Issue: Robot DDS OOM

Sensor nodes on robot get SIGKILL'd by OOM killer during DDS discovery surge from 280+ unitree SDK entities. Current mitigation:

- Staggered startup (`g1_start_sensors.sh`) — nodes start 4-6s apart
- `ROS_DOMAIN_ID=1` — isolates from domain-0 unitree entities
- `MaxAutoParticipantIndex=4` in `cyclone_sensors.xml`

Status: `/livox/lidar` + `/tf_static` confirmed working. Camera node starts but may be killed before finishing stream init. Investigating further.

---

## Original Perception Pipeline

The original `g1_perception` package (lidar_bridge, ccvnorm, centerpoint, pointpillar) is preserved. See the bottom of this file for its full documentation.

<details>
<summary>Original g1_perception docs</summary>

### lidar_bridge

Normalizes sim/real LiDAR → `/livox/mid360/points`. Passthrough on sim; re-frames `/utlidar/cloud` (Foxy frame_id) on real.

### ccvnorm_node

Fuses LiDAR + D435 RGB-D depth. Two modes:

1. `ccvnorm_pseudo_stereo` — confidence-weighted merge, no GPU
2. `ccvnorm_network` — stereo GCNetLiDAR model, requires fine-tuning

### centerpoint_node / pointpillar_node

Both subscribe `/livox/mid360/points`, publish `Detection3DArray` + `MarkerArray`. CenterPoint uses ported `livox_model_1.pt` weights.

</details>

---

## Locomotion Control Reference

See the locomotion section for full two-process constraint explanation, environment setup, FSM IDs, and diagnostics.

```bash
# Confirmed FSM IDs for this robot/firmware:
# 0 = Zero torque, 1 = Damp, 4 = Ready/standing, 501 = Walk mode

# Diagnose DDS connectivity:
export CYCLONEDDS_HOME=/home/thakk100/cyclonedds/install
cyclonedds ps   # lists all DDS participants — healthy robot = ~15-20
```
