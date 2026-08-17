# 2-Pass Snapshot Point Cloud Collection, Detection, & Human Follow Pipeline

## 1. Overview & Motivation

When streaming inference directly on continuous live Livox LiDAR streams, point clouds in individual frames are often sparse, causing 3D bounding box detection jitter and fluctuating CLI selection menus (inference runs at 3–7 Hz while LiDAR publishes at 10–20 Hz).

The **2-Pass Snapshot Pipeline** decouples continuous streaming from detection:
1. **Pass 1 (Point Cloud Accumulation)**: Gathers $N$ frames (e.g. 10 frames) or $T$ seconds (e.g. 2.0s) of raw Livox LiDAR data into a single dense, high-resolution point cloud and latches it to `/livox/collected_points` with **Transient Local (Latched) QoS**.
2. **Pass 2 (Single-Pass 3D Detection & Ranking)**: Executes CenterPoint / PointPillars 3D detection **once** on the accumulated dense cloud, transforms all bounding boxes into the robot's `pelvis` frame, and ranks detected humans by distance (#1 closest, #2 next).
3. **Stable Interactive CLI Selection**: Displays a non-flickering static CLI menu in the terminal. The operator can choose a target human (e.g., `1`, `2`) or press `R` to capture a fresh 2-second snapshot.
4. **Locomotion Walk-Up**: Computes an approach waypoint stopping **60 cm in front of the selected human** and commands the robot via `/g1/cmd_pose` or direct socket RPC to `robot_bridge.py`.
5. **Post-Walkup Greeting**: Upon reaching the 60 cm standoff, triggers arm gestures (**shake hands** or **low wave / face wave**) before releasing arms to neutral.

---

## 2. System Architecture

```
                                  [ PASS 1: ACCUMULATION ]
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              │                                                               │
  Live LiDAR (/livox/lidar)                                    CSV Replay / Rosbag
              │                                                               │
              └───────────────────────────────┬───────────────────────────────┘
                                              ▼
                             livox_snapshot_pipeline_node
                                              │
                    Accumulates 10 frames / 2.0s into dense cloud
                                              │
                                              ▼
                                   [ PASS 2: INFERENCE ]
                             CenterPoint / PointPillars (ONCE)
                                              │
                   Transforms boxes to 'pelvis' & ranks by distance
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
             /livox/collected_points                        /g1/sorted_humans
          (Latched PointCloud2)                      (Ranked Vision Bounding Boxes)
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                               Interactive CLI Menu (Terminal)
                                      [1] Human #1 (1.45m)
                                      [2] Human #2 (2.80m)
                                      [R] Re-take Snapshot
                                              │ (User types '1')
                                              ▼
                                     /g1/selected_human
                                              │
                                              ▼
                                human_follow_and_greet_node
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              ▼                                                               ▼
   [ PHASE 1: LOCOMOTION ]                                         [ PHASE 2: GREETING ]
  Calculates 60cm Standoff Waypoint                               Arrived at 60cm standoff
  Commands /g1/cmd_pose or Socket Move                            Triggers Shake Hand / Low Wave
```

---

## 3. Nodes & ROS 2 Topics

### Key Nodes
- **`livox_snapshot_pipeline_node`** (`livox_detection`):
  Accumulates LiDAR frames, runs 3D detection, publishes latched dense cloud and sorted human detections, and runs the interactive CLI terminal selector.
- **`human_loco_approach_node`** (`livox_detection`) / **`human_follow_and_greet_node`** (`g1_arm_control`):
  Receives `/g1/selected_human`, computes the 60 cm standoff waypoint, executes locomotion approach, and triggers the greeting arm action upon arrival.
- **`g1_arm_controller_node`** (`g1_arm_control`):
  Exposes ROS 2 services (`/g1/arm/shake_hand`, `/g1/arm/low_wave`, `/g1/arm/wave`) and manages arm gestures via `robot_bridge.py`.
- **`robot_bridge.py`** (`g1_control`):
  Standalone background RPC bridge owning the Unitree SDK `LocoClient` and `G1ArmActionClient` DDS connections.

### Topics Summary
| Topic | Type | QoS | Description |
|---|---|---|---|
| `/livox/lidar` | `sensor_msgs/PointCloud2` | Volatile | Raw streaming LiDAR input |
| `/livox/collected_points` | `sensor_msgs/PointCloud2` | **Transient Local** | 10-frame accumulated dense point cloud |
| `/g1/sorted_humans` | `vision_msgs/Detection3DArray` | Transient Local | 3D bounding boxes ranked by distance |
| `/g1/sorted_human_markers` | `visualization_msgs/MarkerArray` | Transient Local | Distance badge numbers in 3D RViz |
| `/g1/selected_human` | `geometry_msgs/PoseStamped` | Reliable | Target human chosen by operator |
| `/g1/selected_human_marker`| `visualization_msgs/MarkerArray`| Transient Local | High-visibility green target beacon |
| `/g1/approach_markers` | `visualization_msgs/MarkerArray` | Transient Local | 60cm standoff stopping pad & approach line |
| `/g1/greeting_markers` | `visualization_msgs/MarkerArray` | Transient Local | Post-walkup greeting status billboard & gestures |
| `/g1/cmd_pose` | `geometry_msgs/Twist` | Reliable | Relative body-frame displacement command |

---

## 4. How to Launch & Operate

### Step 1: Source the Environment
```bash
cd ~/Projects/thesis/g1_perception_ws
source setup_g1_env.sh
```

### Step 2: Start Robot Bridge (When on Real Robot)
```bash
python3 src/g1_control/g1_control/robot_bridge.py enp2s0
```

### Step 3: Launch the Integrated Snapshot & Greeting Pipeline
```bash
# On Real Robot:
ros2 launch g1_arm_control snapshot_follow_and_greet.launch.py

# For Offline / Standalone Simulation (Publishes Static TFs):
ros2 launch g1_arm_control snapshot_follow_and_greet.launch.py publish_tf:=true
```

### Step 4: Interactive Workflow
1. When launched, the pipeline gathers 10 frames (~2.0 seconds) of LiDAR data:
   ```text
   =======================================================
    [●] PASS 1: COLLECTING POINT CLOUDS FROM '/livox/lidar'...
        Target: 10 frames (or 2.0s)
   =======================================================
     -> Captured frame #2/10 (5,720 points, 0.4s elapsed)
     -> Captured frame #4/10 (11,440 points, 0.8s elapsed)
     -> Captured frame #6/10 (17,160 points, 1.2s elapsed)
     -> Captured frame #8/10 (22,880 points, 1.6s elapsed)
     -> Captured frame #10/10 (28,600 points, 2.0s elapsed)

   =======================================================
    [●] PASS 2: RUNNING 3D DETECTION ON DENSE POINT CLOUD
        Total Accumulated Points: 28,600 across 10 frames
        Model Backend: CENTERPOINT | Target Frame: 'pelvis'
   =======================================================
    [✓] Published dense cloud to topic '/livox/collected_points' (Latched)
    [✓] Detection finished in 142.3ms. Found 2 human(s).

   =======================================================
     [G1 Snapshot Detection Menu] - Frame: 'pelvis'
   =======================================================
     [1]  Distance: 1.45 m (Closest) | Position: (x=+1.40, y=+0.30, z=+0.10) | Conf: 0.88
     [2]  Distance: 2.80 m           | Position: (x=+2.70, y=-0.65, z=+0.05) | Conf: 0.79
   -------------------------------------------------------
     [R]  Re-trigger: Collect a new 2.0s snapshot
     [0]  Clear current selection
   =======================================================
   Enter human number to approach [1-2] or 'R' to re-collect: 
   ```

2. **Select Human**:
   Type `1` and press `Enter`.
   - RViz places a green beacon over Human #1 and draws the approach trajectory to the 60 cm standoff pad.
   - The robot walks up to the target.
   - Upon arriving at the 60 cm mark, the robot executes the arm greeting (**shake hands** or **low wave**), displays visual confirmation in RViz, and releases its arm when done.

3. **Re-take Snapshot**:
   Type `R` and press `Enter` (or invoke service `ros2 service call /g1/trigger_snapshot std_srvs/srv/Trigger`).
