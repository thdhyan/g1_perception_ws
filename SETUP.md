# SETUP.md — g1_perception_ws

Quick-start for two targets:

| Machine | ROS2 | GPU | Role |
|---------|------|-----|------|
| **Demo laptop** (`g1_demo_laptop`) | Humble | RTX, 16 GB VRAM | Perception + HMR inference |
| **Robot onboard** | Foxy | CPU / embedded | Bringup, DDS relay |

---

## 1 — Clone

```bash
git clone --recurse-submodules https://github.com/<your-org>/g1_perception_ws.git
cd g1_perception_ws
```

`--recurse-submodules` pulls **LiDAR-HMR** automatically.

If you cloned without it:

```bash
git submodule update --init --recursive
```

---

## 2 — SMPL model file (NEUTRAL.pkl)

Not in the upstream LiDAR-HMR repo (SMPL license). Copy from robot or another machine:

```bash
scp robot:/path/to/g1_perception_ws/LiDAR-HMR/smplx_models/smpl/SMPL_NEUTRAL.pkl \
    LiDAR-HMR/smplx_models/smpl/SMPL_NEUTRAL.pkl
# or from another local path:
cp /Storage/models/SMPL_NEUTRAL.pkl LiDAR-HMR/smplx_models/smpl/SMPL_NEUTRAL.pkl
```

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

## 5 — VoxelNeXt checkpoint

Checkpoint not in repo (large binary). Copy from robot or download:

```bash
scp robot:/path/to/g1_perception_ws/pt/voxelnext_ped.pth pt/
```

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
- **LiDAR-HMR ckpt**: `humanm3` checkpoint (~490 MB) lives in `LiDAR-HMR/ckpts/`.
  Copy from robot if not present: `scp robot:.../LiDAR-HMR/ckpts/ LiDAR-HMR/ckpts/ -r`
