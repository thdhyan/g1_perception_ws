# SETUP.md — g1_perception_ws

Quick-start for two targets:

| Machine | ROS2 | GPU | Role |
|---------|------|-----|------|
| **Demo laptop** (`g1_demo_laptop`) | Humble | RTX, 16 GB VRAM | Perception + HMR inference |
| **Robot onboard** | Foxy | CPU / embedded | Bringup, DDS relay |

---

## 0 — Git commands per device

### First time (fresh clone)

**Demo laptop:**
```bash
git clone --recurse-submodules https://github.com/thdhyan/g1_perception_ws.git
cd g1_perception_ws
```

**Robot onboard:**
```bash
git clone --recurse-submodules https://github.com/thdhyan/g1_perception_ws.git
cd g1_perception_ws
```

Same command — `--recurse-submodules` pulls LiDAR-HMR on both.

---

### Pulling updates (already cloned)

**Demo laptop:**
```bash
cd ~/Projects/thesis/g1_perception_ws   # or wherever you cloned
git pull
git submodule update --recursive
```

**Robot onboard:**
```bash
cd ~/g1_perception_ws                   # adjust to your path on robot
git pull
git submodule update --recursive
```

`git submodule update --recursive` is required every pull — it advances
LiDAR-HMR to the pinned commit if the submodule pointer changed.

---

## 1 — Clone (detail)

`--recurse-submodules` pulls **LiDAR-HMR** automatically.

If you cloned without it:

```bash
git submodule update --init --recursive
```

---

## 2 — Model files via SSD

All checkpoints live on the **SSD** at:

```
/Storage/models/g1_perception_ws/
├── voxelnext_nuscenes.pth      31 MB   VoxelNeXt detector
├── voxelkp_waymo.pth         1.3 GB   VoxelKP (backup detector)
├── lidar_hmr_mesh.pth         491 MB   LiDAR-HMR humanm3 mesh head
├── prn_pct.pth                 21 MB   LiDAR-HMR humanm3 PCT backbone
├── SMPL_NEUTRAL.pkl           236 MB   SMPL body model (not in any repo)
├── reid_model.pt              690 KB   PointNet ReID (triplet-trained)
└── reid_model_identity.pt     570 KB   PointNet ReID (identity-trained)
```

**Connect SSD, then on demo laptop:**

```bash
WS=~/Projects/thesis/g1_perception_ws
SSD=/media/$(whoami)/$(ls /media/$(whoami)/ | head -1)/models/g1_perception_ws
# ↑ adjust SSD mount path if needed

mkdir -p $WS/pt \
         $WS/LiDAR-HMR/ckpts/humanm3 \
         $WS/LiDAR-HMR/smplx_models/smpl \
         $WS/reid_data

cp $SSD/voxelnext_nuscenes.pth    $WS/pt/
cp $SSD/voxelkp_waymo.pth         $WS/pt/
cp $SSD/lidar_hmr_mesh.pth        $WS/LiDAR-HMR/ckpts/humanm3/
cp $SSD/prn_pct.pth               $WS/LiDAR-HMR/ckpts/humanm3/
cp $SSD/SMPL_NEUTRAL.pkl          $WS/LiDAR-HMR/smplx_models/smpl/
cp $SSD/reid_model.pt             $WS/reid_data/model.pt
cp $SSD/reid_model_identity.pt    $WS/reid_data/model_identity.pt
```

**On robot onboard** (only needs VoxelNeXt + SMPL if running HMR on robot):

```bash
WS=~/g1_perception_ws   # adjust to clone path on robot
SSD=/media/$(whoami)/$(ls /media/$(whoami)/ | head -1)/models/g1_perception_ws

mkdir -p $WS/pt $WS/LiDAR-HMR/smplx_models/smpl

cp $SSD/voxelnext_nuscenes.pth  $WS/pt/
cp $SSD/SMPL_NEUTRAL.pkl        $WS/LiDAR-HMR/smplx_models/smpl/
```

---

## 2b — Session data via SSD

LiDAR recordings and annotation data live on the **SSD** at:

```
/Storage/data/
├── downloads/
│   ├── 2026-07-29_17-20-14_points.csv    133 MB   short session
│   ├── 2026-07-29_17-21-48_points.csv    1.6 GB   main session (default for playback)
│   ├── 20260511_181110/
│   │   └── rgb_07_joint_positions.csv
│   ├── results_v2/                        model eval JSONs
│   ├── compare/                           timing report JSONs
│   └── *.json                             veo prompts, analysis, camera params
└── reid_data/
    ├── identity_map_2026-07-29_17-21-48.json
    ├── sameperson_2026-07-29_17-21-48.json
    ├── sameperson_2026-08-05_16-38-40.json
    ├── tracks_smpl_2026-07-29_17-21-48.json
    └── train_log.csv
```

**Connect SSD, then on demo laptop:**

```bash
WS=~/Projects/thesis/g1_perception_ws
SSD=/media/$(whoami)/$(ls /media/$(whoami)/ | head -1)/data

# LiDAR session CSVs → Downloads (default playback path)
mkdir -p ~/Downloads
cp "$SSD/downloads/2026-07-29_17-20-14_points.csv" ~/Downloads/
cp "$SSD/downloads/2026-07-29_17-21-48_points.csv" ~/Downloads/

# ReID annotation data
mkdir -p $WS/reid_data
cp $SSD/reid_data/*.json $WS/reid_data/
cp $SSD/reid_data/*.csv  $WS/reid_data/
```

Robot onboard does **not** need session data (inference only, no playback).

---

## 3 — Python environment (demo laptop — Humble)

Requires **uv** (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync          # reads uv.lock — exact reproducible install, CUDA 12.1 torch
```

For fresh systems without the venv yet:

```bash
uv venv          # creates .venv (python 3.12)
uv sync
```

Activate when running scripts directly (not needed for `ros2 launch`):

```bash
source .venv/bin/activate
```

### LD_LIBRARY_PATH (VoxelKP CUDA ops)

Add to `~/.bashrc` or source before launch:

```bash
export LD_LIBRARY_PATH=$HOME/Projects/thesis/g1_perception_ws/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
```

---

## 4 — ROS2 build

### Demo laptop (Humble)

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select \
    g1_perception livox_detection g1_bringup
source install/setup.bash
```

### Robot onboard (Foxy)

```bash
source /opt/ros/foxy/setup.bash
colcon build --symlink-install --packages-select \
    g1_bringup livox_ros2_driver
source install/setup.bash
```

---

---

## 6 — Launch: LiDAR + HMR pipeline

### Demo laptop (live robot connected)

```bash
# Terminal 1 — sourced env
ros2 launch g1_perception smpl_full_stack.launch.py

# Optional: trajectory viewer (port 8767)
python3 reid_embed_server.py --smpl-mode --session <session-tag>
# Open http://localhost:8767
```

### Demo laptop (CSV playback — no robot needed)

```bash
ros2 launch g1_perception smpl_csv_playback.launch.py \
    csv:=$HOME/Downloads/<session>_points.csv
```

### Robot onboard (Foxy) — bringup only

```bash
source /opt/ros/foxy/setup.bash
source install/setup.bash
ros2 launch g1_bringup livox_mid360.launch.py   # LiDAR driver
```

---

## 7 — ROS2 domain bridge (laptop ↔ robot, different distros)

Humble and Foxy share DDS (CycloneDX / FastRTPS) — topics bridge automatically
on the same LAN **if `ROS_DOMAIN_ID` matches**:

```bash
# Both machines:
export ROS_DOMAIN_ID=42
```

Verify on laptop:

```bash
ros2 topic list   # should show /livox/mid360/points from robot
```

If behind NAT or different subnets, use `ros2-domain-bridge` package:

```bash
sudo apt install ros-humble-domain-bridge
ros2 run domain_bridge domain_bridge config.yaml
```

---

## 8 — Notes

- **Storage**: large sessions → `/Storage` or `/generalssd`, not `/home` (see memory).
- **torch.compile** warm-up: first inference batch is slow (kernel compilation). Normal.
- **TF32**: enabled automatically on RTX Ampere+ (3080/4090/etc). Safe for detection.
- **LiDAR-HMR ckpt**: `humanm3` checkpoint (~490 MB) — copy from SSD (see §2).
