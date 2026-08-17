# Real Robot Workflow & Operation Guide: Unitree G1

Complete guide for operating the full-stack perception, 3D detection, human following, and greeting pipeline on the real Unitree G1 humanoid robot.

---

## 1. System Architecture Overview

```
 [ REAL G1 ROBOT (192.168.123.164) ]
   ├── Livox Mid-360 LiDAR ───────────────> /livox/lidar (PointCloud2 / CustomMsg, 10Hz)
   ├── RealSense D435i Camera ────────────> /camera/... (RGB, Depth, Points)
   └── Unitree LowState/Loco DDS ─────────> Domain 0 / CycloneDDS

                             ▲ Ethernet (enp2s0, 192.168.123.220)
                             │ ROS_DOMAIN_ID=0 / CycloneDDS
                             ▼

 [ LAPTOP WORKSPACE (Ubuntu 24.04, ROS 2 Jazzy) ]
   ├── Terminal 1: Locomotion Bridge (robot_bridge.py)
   │     └─ Owns Unitree LocoClient, exposes Unix Socket: /tmp/g1_robot_bridge.sock
   │
   └── Terminal 2: Perception, Detection & Follow Pipeline (real_human_follow.launch.py)
         ├─ robot_state_publisher & TF Tree
         ├─ 2-Pass Snapshot Pipeline Node:
         │    ├─ Pass 1: Accumulate dense point cloud (10 frames / 2.0s)
         │    └─ Pass 2: 3D Detection (VoxelNeXt / CenterPoint / PointPillars)
         ├─ Distance Sorter & Non-Flicker Human Selection
         ├─ human_follow_and_greet_node:
         │    ├─ Approaching: Computes standoff goal in pelvis frame
         │    ├─ Locomotion: Streams /cmd_vel -> robot_bridge.py -> G1 walk
         │    └─ Arrival: Executes arm gesture (Handshake / Wave)
         └─ RViz2 3D Live Visualizer
```

---

## 2. Network & Preflight Configuration

### Network Setup
- Robot Ethernet Interface: `192.168.123.164`
- Laptop Ethernet Interface (`enp2s0`): Static IP `192.168.123.220`, Netmask `255.255.255.0`

### Verify Connectivity
```bash
# Test ping to robot
ping -c 3 192.168.123.164

# Test SSH access
ssh unitree@192.168.123.164
```

---

## 3. Step-by-Step Operation Workflow

### Step 1: Start Sensors on the Real Robot
From your laptop, launch the robot sensor drivers remotely:

```bash
cd /home/thakk100/Projects/thesis/g1_perception_ws
./scripts/start_robot_sensors.sh
```

*(Alternatively, SSH into the robot and run: `bash /home/unitree/Projects/g1_start_sensors.sh`)*

**Verify topics on laptop:**
```bash
source setup_g1_env.sh
ros2 topic hz /livox/lidar
```

---

### Step 2: Start the Locomotion Control Bridge (Terminal 1)
Due to the DDS co-existence constraint (`rclpy` + `unitree_sdk2py` in the same process causes segfaults), the robot locomotion client runs in a lightweight, non-ROS bridge process:

```bash
cd /home/thakk100/Projects/thesis/g1_perception_ws
source setup_g1_env.sh
python3 src/g1_control/g1_control/robot_bridge.py
```
*Keep this terminal open.* It will log DDS connection state and listen on `/tmp/g1_robot_bridge.sock`.

---

### Step 3: Launch Full Human Perception & Follow Stack (Terminal 2)

#### Option A: Run with **VoxelNeXt** (State-of-the-Art Fully-Sparse 3D Detector - Recommended)
```bash
cd /home/thakk100/Projects/thesis/g1_perception_ws
source setup_g1_env.sh
ros2 launch g1_bringup real_human_follow.launch.py \
  algorithm:=voxelnext \
  checkpoint_path:=pt/voxelnext_nuscenes.pth \
  standoff_distance:=0.80 \
  greeting_action:=shake_hand
```

#### Option B: Run with **CenterPoint**
```bash
cd /home/thakk100/Projects/thesis/g1_perception_ws
source setup_g1_env.sh
ros2 launch g1_bringup real_human_follow.launch.py \
  algorithm:=centerpoint \
  checkpoint_path:=/home/thakk100/Projects/Thesis/livox_detection/pt/livox_model_1.pt
```

#### Option C: Run with **PointPillars** (Geometry Clustering Fallback)
```bash
cd /home/thakk100/Projects/thesis/g1_perception_ws
source setup_g1_env.sh
ros2 launch g1_bringup real_human_follow.launch.py \
  algorithm:=pointpillar
```

---

## 4. Key Launch Parameters

| Parameter | Default | Options / Range | Description |
|---|---|---|---|
| `algorithm` | `voxelnext` | `voxelnext`, `centerpoint`, `pointpillar` | 3D detection model backend |
| `checkpoint_path` | `pt/voxelnext_nuscenes.pth` | File path (.pth / .pt) | Model checkpoint path |
| `standoff_distance` | `0.80` | `0.4` - `2.0` (meters) | Distance to stop in front of target human |
| `greeting_action` | `shake_hand` | `shake_hand`, `high_wave`, `low_wave`, `none` | Arm gesture executed on arrival |
| `linear_speed` | `0.20` | `0.05` - `0.40` (m/s) | Safe robot walking velocity |
| `collect_frames` | `10` | `5` - `20` | Number of LiDAR frames accumulated in Pass 1 |
| `collect_duration_sec` | `2.0` | `1.0` - `5.0` | Max duration for snapshot accumulation |
| `score_threshold` | `0.15` | `0.05` - `0.90` | Detection confidence threshold |
| `auto_execute` | `true` | `true`, `false` | Automatically begin approach upon selection |
| `auto_greet` | `true` | `true`, `false` | Automatically greet upon reaching standoff |
| `rviz` | `true` | `true`, `false` | Launch RViz2 visualizer |

---

## 5. Operator Interaction & Human Selection

1. The snapshot pipeline collects 10 Livox LiDAR scans (accumulating a dense 360° point cloud).
2. The 3D detection model runs inference on the dense frozen cloud.
3. Detected humans are sorted by distance from the robot pelvis and color-coded in RViz.
4. If `auto_start:=true` (default in `real_human_follow.launch.py`), the nearest detected human is automatically selected and followed.
5. The robot rotates toward the human, walks at `0.20 m/s` until reaching the `0.80m` standoff boundary, halts, and extends its arm to perform the handshake/greeting.

---

## 6. Safety & Emergency Stop

- **Keyboard Interrupt**: Press `Ctrl+C` in Terminal 2 (stops walking commands immediately; deadman timer halts robot within 0.5s).
- **Physical E-Stop**: Use the Unitree remote controller (B button / Damp mode) if physical intervention is needed.
- **Bridge Kill**: `pkill -f robot_bridge.py` terminates DDS communication immediately.
