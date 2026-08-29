# G1 Perception & Navigation Stack — System Updates & Changelog

This document tracks all recent updates, technical implementations, bug fixes, and operational instructions.

---

## Change History

### [2026-08-16] — Parallel TF Trees, Keyboard Teleop, and WBC Stabilization

#### 1. Dual Parallel TF Tree Architecture
- **Problem**: Conflicting transforms between Gazebo simulation ground-truth (`warehouse -> g1 -> pelvis`) and 3D SLAM / Odometry (`map -> odom -> pelvis`), leading to dropped TF frames in RViz and `mid360_link -> warehouse` transform lookup errors.
- **Solution**: Decoupled the transforms into two separate trees:
  1. **Ground Truth Branch**: `warehouse -> g1 -> gt_base_link -> gt_pelvis` (published by Gazebo `PosePublisher` bridge).
  2. **Proprioception & Mapping Branch**: `map -> odom -> pelvis -> <all URDF robot links> -> mid360_link -> livox_frame`.
- **RViz Configuration**: Fixed frame updated to `map`.

#### 2. Interactive Keyboard Teleoperation Node (`keyboard_teleop.py`)
- **Package**: `g1_nav`
- **Script**: [`keyboard_teleop.py`](file:///home/thakk100/Projects/thesis/g1_perception_ws/src/g1_nav/g1_nav/keyboard_teleop.py)
- **Key Bindings**:
  - `W` / `S`: Walk Forward / Backward ($\pm v_x$)
  - `A` / `D`: Strafe Left / Right ($\pm v_y$)
  - `Q` / `E`: Turn Left / Right ($\pm \omega_z$)
  - `SPACE`: Instant Brake / Hold stationary stance ($0.0\text{ m/s}$)
  - `Z` / `C`: Decrease / Increase linear speed ($\pm 0.05\text{ m/s}$)
  - `U` / `O`: Decrease / Increase yaw rate ($\pm 0.10\text{ rad/s}$)
  - `H`: Reset stance height to nominal $0.74\text{ m}$
- **Topics**: Publishes synchronized `Twist` messages to `/g1/cmd_vel` and `/cmd_vel` at 20 Hz.

#### 3. Whole-Body Locomotion & Balancing Behavior
- **File**: [`wbc_node.py`](file:///home/thakk100/Projects/thesis/g1_perception_ws/src/g1_wbc/g1_wbc/wbc_node.py)
- **Stationary Stance**: Idle speed $\|\text{cmd}\| \le 0.05\text{ m/s}$ forces the ONNX `GR00T-WholeBodyControl-Balance` policy to maintain a firm, upright standing posture without forward drift.
- **Watchdog**: 0.5s deadman timer zeros commanded velocities if input streams stop.

#### 4. 3D SLAM Exception Resilience (`plain_slam_ros2`)
- **File**: [`lio_3d_node.cpp`](file:///home/thakk100/Projects/thesis/g1_perception_ws/src/plain_slam_ros2/src/lio_3d_node.cpp)
- **Fix**: Wrapped point cloud processing (`SetScanCloud`) in `try-catch` blocks to catch and log non-convergent iterations gracefully without crashing the LIO pipeline.

#### 5. Perception & Detection Backend
- **File**: [`livox_detection_node.py`](file:///home/thakk100/Projects/thesis/g1_perception_ws/src/livox_detection/livox_detection/livox_detection_node.py)
- **Backend**: VoxelNeXt architecture (`pt/voxelnext_nuscenes.pth`) loading on CUDA device.
- **Detections**: Generates 3D bounding boxes and pedestrian classification markers for warehouse human actors.

---

## Operating Instructions

### Launching the Simulation Stack

#### **Terminal 1: Start Simulation, Perception, 3D SLAM, and RViz**
```bash
cd ~/Projects/thesis/g1_perception_ws
source setup_g1_env.sh
source install/setup.bash
ros2 launch g1_bringup sim_teleop.launch.py rviz:=true headless:=false
```

#### **Terminal 2: Run WASDQE Keyboard Teleoperation**
```bash
cd ~/Projects/thesis/g1_perception_ws
source setup_g1_env.sh
source install/setup.bash
ros2 run g1_nav keyboard_teleop
```
