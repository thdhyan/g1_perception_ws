# REPO.md — g1_perception_ws Node & Package Registry

> **Maintenance**: when adding a node, append a row to the relevant package section and add its launch file entry.  
> Last updated: 2026-09-02

---

## Quick Reference

| Package | Nodes | Launch files |
|---------|-------|-------------|
| `g1_perception` | 14 | 11 |
| `livox_detection` | 8 | 9 |
| `g1_bringup` | — | 8 |
| `g1_control` | 4 | 2 |
| `g1_arm_control` | 2 | 2 |
| `g1_nav` | 2 | 2 |
| `g1_isaac_slam` | 1 | 2 |
| `g1_livox_pose` | 2 | 1 |
| `g1_voice` | 4 | 1 |
| `g1_wbc` | — | 1 |
| `plain_slam_ros2` | — | 2 |

---

## `g1_perception` — Core perception stack

**Path**: `src/g1_perception/g1_perception/`

### Nodes

| Node | Entry point | Description |
|------|-------------|-------------|
| `smpl_hmr_node` | `smpl_hmr_node.py` | **[PRIMARY REID NODE]** Subscribes to PointCloud2 + Detection3DArray (ApproxTimeSynced). Crops per-person point clouds, runs LiDAR-HMR (VoteHMR/PMG, humanm3 ckpt, 490MB) for SMPL β (10-d body shape) + θ (72-d pose). BetaTracker assigns stable person IDs via cosine-similarity lookup table (threshold 0.85). Publishes: `/g1/smpl/mesh` (TRIANGLE_LIST), `/g1/smpl/joints` (SPHERE_LIST), `/g1/smpl/skeleton` (LINE_LIST), `/g1/smpl/tracks` (JSON String). |
| `lidar_bridge` | `lidar_bridge.py` | Bridges Livox SDK custom messages to standard PointCloud2. Republishes `/livox/lidar` in ROS2 format. |
| `detection_bridge` | `detection_bridge.py` | Bridges detection output between robot and laptop ROS domains. |
| `reid_enroll_node` | `reid_enroll_node.py` | Enrolls a target person embedding into the ReID server. Subscribes to detection, crops the target, sends to `reid_embed_server` (port 8767). |
| `reid_matcher_node` | `reid_matcher_node.py` | Matches incoming detections against enrolled identity. Uses cosine similarity on 128-d embeddings (legacy PointNet). |
| `human_selector_node` | `human_selector_node.py` | Interactive human selection node; publishes selected target person to `/g1/target`. |
| `ccvnorm_node` | `ccvnorm_node.py` | Colour + contrast + value normalisation node for camera images. |
| `cmd_pose_bridge` | `cmd_pose_bridge.py` | Bridges command pose messages to robot locomotion API. |
| `robot_bridge` | `robot_bridge.py` | High-level robot state bridge (battery, mode, etc.). |
| `lidar_odometry_node` | `lidar_odometry_node.py` | Wraps LIO-SAM / FAST-LIO for LiDAR-based odometry. |
| `nav_goal_node` | `nav_goal_node.py` | Converts high-level nav goals to Nav2 PoseStamped. |
| `scan_restamper` | `scan_restamper.py` | Restamps laser scan with ROS clock (resolves sim-time drift). |
| `move_to_xy` | `move_to_xy.py` | Sends robot to absolute X/Y waypoint via Nav2 simple commander. |
| `nav2point` | `nav2point.py` | Nav2 client for point-to-point navigation. |

### Launch files

| Launch file | Description |
|-------------|-------------|
| `smpl_csv_playback.launch.py` | **[CSV PLAYBACK]** livox_csv_player → VoxelNeXt detection → smpl_hmr_node → RViz2. No robot needed. Default CSV: `~/Downloads/2026-07-29_17-21-48_points.csv` |
| `smpl_full_stack.launch.py` | **[LIVE FULL STACK]** lidar_bridge → VoxelNeXt → smpl_hmr_node → optional ros2 bag record -a |
| `smpl_hmr.launch.py` | smpl_hmr_node only (assumes cloud + detections running separately) |
| `reid.launch.py` | Legacy ReID stack: reid_enroll_node + reid_matcher_node |
| `perception.launch.py` | Core perception: lidar_bridge + detection + human_selector |
| `full_stack.launch.py` | Full robot stack: perception + control + nav |
| `laptop_stack.launch.py` | Laptop-side stack for on-robot ROS domain relay |
| `navigation.launch.py` | Nav2 stack + slam/odom |
| `mapping.launch.py` | SLAM mapping only |
| `lidar_odometry.launch.py` | LIO only |
| `ccvnorm.launch.py` | Camera normalisation pipeline |
| `g1pilot_navigation.launch.py` | Full autonomy stack for G1Pilot |

---

## `livox_detection` — LiDAR person detection

**Path**: `src/livox_detection/livox_detection/`

### Nodes

| Node | Entry point | Description |
|------|-------------|-------------|
| `livox_detection_node` | `livox_detection_node.py` | **[MAIN DETECTOR]** VoxelNeXt 3D detector on PointCloud2. Publishes Detection3DArray to `/g1/detections/livox` and `/g1/detection_markers/livox` (MarkerArray for RViz). Transforms boxes to `target_frame` (default: `pelvis`) via TF2. |
| `livox_csv_player_node` | `livox_csv_player_node.py` | Streams a recorded Livox CSV file as PointCloud2 at 10Hz. Supports two schemas: Livox Viewer native (≥11 cols) and our recorder format (header: `x,y,z,reflectivity,timestamp_ns,frame_idx`). Loops. Publishes static TF `odom→base_link→pelvis`. |
| `human_distance_sorter_node` | `human_distance_sorter_node.py` | Sorts detected humans by distance to robot. Publishes nearest human. |
| `human_keyboard_selector_node` | `human_keyboard_selector_node.py` | Keyboard-based interactive selection of a detected human target. |
| `human_loco_approach_node` | `human_loco_approach_node.py` | Locomotion approach controller — steers robot toward selected human. |
| `livox_front_filter_node` | `livox_front_filter_node.py` | Filters point cloud to front-facing sector only. |
| `livox_snapshot_pipeline_node` | `livox_snapshot_pipeline_node.py` | Captures single-frame snapshots of detections for offline processing. |
| `person_namer_node` | `person_namer_node.py` | Assigns text labels to detected persons (used with ReID enrollment). |
| `voxelnext_model` | `voxelnext_model.py` | VoxelNeXt model loader and inference wrapper (not a ROS node; imported by livox_detection_node). |

### Launch files

| Launch file | Description |
|-------------|-------------|
| `livox_detection.launch.py` | Bare detection node (assumes lidar running) |
| `voxelnext_detection.launch.py` | VoxelNeXt detector only with config |
| `livox_csv_playback.launch.py` | CSV player + detection only (no HMR) |
| `human_follow_pipeline.launch.py` | Detection + distance sorter + approach controller |
| `snapshot_pipeline.launch.py` | Snapshot capture pipeline |
| `snapshot_human_follow.launch.py` | Snapshot + follow combined |
| `person_namer.launch.py` | Detection + person namer |
| `g1_detection_sim.launch.py` | Sim-mode detection (Gazebo) |
| `view_robot_lidar.launch.py` | RViz viewer for robot LiDAR |

---

## `g1_bringup` — Robot bringup orchestration

**Path**: `src/g1_bringup/launch/`  
No custom nodes (launch-only package).

| Launch file | Description |
|-------------|-------------|
| `real.launch.py` | Minimal real-robot bringup: sensors + state publishers |
| `full_real.launch.py` | Full real-robot stack: sensors + perception + control |
| `real_live_detection.launch.py` | Real robot + live detection running |
| `real_human_follow.launch.py` | Real robot + human follow pipeline |
| `g1_sensors.launch.py` | Sensor stack only: Livox + cameras + IMU |
| `sensors_only.launch.py` | Bare sensor publishing (no processing) |
| `sim.launch.py` | Gazebo simulation bringup |
| `sim_teleop.launch.py` | Sim + teleop keyboard control |

---

## `g1_control` — Robot locomotion control

**Path**: `src/g1_control/g1_control/`

| Node | Entry point | Description |
|------|-------------|-------------|
| `cmd_vel_bridge` | `cmd_vel_bridge.py` | Converts Twist (cmd_vel) to Unitree G1 SportModeCmd. |
| `cmd_pose_bridge` | `cmd_pose_bridge.py` | Bridges pose commands to G1 API. |
| `human_follower_node` | `human_follower_node.py` | PID-based human follower: subscribes to target pose, outputs cmd_vel. |
| `robot_bridge` | `robot_bridge.py` | Robot state bridge: high_state, low_state → ROS topics. |

| Launch file | Description |
|-------------|-------------|
| `control.launch.py` | Full control stack: bridges + follower |
| `cmd_pose_bridge.launch.py` | Pose bridge only |

---

## `g1_arm_control` — Arm gestures and follow-and-greet

**Path**: `src/g1_arm_control/g1_arm_control/`

| Node | Entry point | Description |
|------|-------------|-------------|
| `g1_arm_controller_node` | `g1_arm_controller_node.py` | Arm trajectory controller — executes pre-programmed arm gestures via Unitree API. |
| `human_follow_and_greet_node` | `human_follow_and_greet_node.py` | Combines human following with greeting gesture on approach. |

| Launch file | Description |
|-------------|-------------|
| `arm_control.launch.py` | Arm controller standalone |
| `snapshot_follow_and_greet.launch.py` | Human follow + greet (snapshot mode) |

---

## `g1_nav` — Navigation utilities

**Path**: `src/g1_nav/g1_nav/`

| Node | Entry point | Description |
|------|-------------|-------------|
| `autonomous_mapper` | `autonomous_mapper.py` | Autonomous frontier exploration + SLAM mapping. |
| `keyboard_teleop` | `keyboard_teleop.py` | Keyboard teleoperation (WASD → cmd_vel). |

| Launch file | Description |
|-------------|-------------|
| `navigation.launch.py` | Nav2 stack (planner, controller, BT) |
| `slam.launch.py` | SLAM + map server |

---

## `g1_isaac_slam` — Isaac ROS SLAM integration

**Path**: `src/g1_isaac_slam/`

| Node | Entry point | Description |
|------|-------------|-------------|
| `depth_to_uint16` | `depth_to_uint16.py` | Converts float32 depth images to uint16 for Isaac ROS visual SLAM. |

| Launch file | Description |
|-------------|-------------|
| `isaac_slam.launch.py` | Isaac ROS visual SLAM pipeline |
| `_container_isaac_slam.launch.py` | Component container version (GPU-accelerated) |

---

## `g1_livox_pose` — LiDAR human pose estimation (legacy)

**Path**: `src/g1_livox_pose/g1_livox_pose/`

| Node | Entry point | Description |
|------|-------------|-------------|
| `human_pose_node` | `human_pose_node.py` | Estimates 3D human pose from LiDAR point cloud (legacy backend). |
| `pose_sequence_assembler_node` | `pose_sequence_assembler_node.py` | Assembles pose keyframe sequences for temporal smoothing. |

| Launch file | Description |
|-------------|-------------|
| `pose_pipeline.launch.py` | Full pose estimation pipeline |

---

## `g1_voice` — Voice interaction

**Path**: `src/g1_voice/g1_voice/`

| Node | Entry point | Description |
|------|-------------|-------------|
| `audio_bridge_node` | `audio_bridge_node.py` | Bridges microphone audio to ROS2 audio topics. |
| `dialog_node` | `dialog_node.py` | LLM-based dialog node — responds to voice commands. |
| `mic_node` | `mic_node.py` | Raw microphone capture and publishing. |
| `audio_backend` | `audio_backend.py` | Audio backend abstraction (not a ROS node; imported by above). |

| Launch file | Description |
|-------------|-------------|
| `voice.launch.py` | Full voice pipeline: mic → dialog → TTS |

---

## `g1_wbc` — Whole-body control

**Path**: `src/g1_wbc/`

No custom Python nodes (C++ only or external).

| Launch file | Description |
|-------------|-------------|
| `wbc.launch.py` | Whole-body controller for loco-manipulation |

---

## `plain_slam_ros2` — LiDAR SLAM

**Path**: `src/plain_slam_ros2/`

Wraps [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) / LIO-SAM for ROS2.

| Launch file | Description |
|-------------|-------------|
| `slam_3d.launch.py` | 3D LiDAR SLAM (map + odom) |
| `lio_3d.launch.py` | LIO-only (no loop closure) |

---

## `livox_laser_simulation_RO2` — Gazebo LiDAR plugin

Simulates Livox Mid-360 point patterns in Gazebo. No ROS2 nodes.

| Launch file | Description |
|-------------|-------------|
| `livox_harmonic_example.launch.py` | Example Gazebo world with simulated Livox |

---

## Topic Map (key topics)

```
/livox/lidar            PointCloud2          — raw LiDAR (mid360_link frame)
/g1/detections/livox    Detection3DArray     — VoxelNeXt boxes (pelvis frame)
/g1/detection_markers/livox  MarkerArray     — RViz boxes + labels
/g1/smpl/mesh           MarkerArray          — SMPL body mesh (TRIANGLE_LIST)
/g1/smpl/joints         MarkerArray          — 24 joint spheres
/g1/smpl/skeleton       MarkerArray          — skeleton edges (LINE_LIST)
/g1/smpl/tracks         std_msgs/String      — JSON: [{id, beta[10], x, y, z}]
/g1/target              Detection3D          — selected follow target
/cmd_vel                Twist                — velocity command to robot
```

---

## External Servers (off-ROS)

| Port | Server | Purpose |
|------|--------|---------|
| 8767 | `reid_embed_server.py` | SMPL β embedding + cosine ReID (active) |
| 8766 | `reid_tracks_server` | Legacy track store (inactive) |
| 8765 | `reid_web_server` | Pair annotator web UI |

---

## Development Notes

- **Python venv**: `.venv/` (Python 3.12). Activate before running scripts.
- **ROS2**: Jazzy (`/opt/ros/jazzy/`). Source before ROS commands.
- **Build**: `colcon build --symlink-install` from workspace root.
- **LiDAR-HMR**: lives at `./LiDAR-HMR/`. Needs `os.chdir(HMR_DIR)` + `site.addsitedir(venv_site)` for `.egg` deps (`pointops`).
- **SMPL models**: `LiDAR-HMR/smplx_models/smpl/SMPL_NEUTRAL.pkl`
- **VoxelNeXt checkpoint**: `pt/voxelnext_nuscenes.pth`
- **HMR checkpoint**: `LiDAR-HMR/ckpts/humanm3/lidar_hmr_mesh.pth` (490 MB)
