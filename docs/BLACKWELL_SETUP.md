# Running on RTX 5090 Blackwell Laptop

## Prerequisites

### 1. CUDA Driver ≥ 12.8
Blackwell (sm_100) requires driver 12.8+. Verify:
```bash
nvidia-smi   # check driver version in top-right corner
```

### 2. NVIDIA Container Toolkit
```bash
# Test first — if this works, skip the install
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi

# Install if not working
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Setup

### Clone and pull images
```bash
git clone https://github.com/thdhyan/g1_perception_ws.git
cd g1_perception_ws
docker compose -f docker/compose.yaml pull
```

### Checkpoint locations
Checkpoints are NOT baked into images — mount at runtime.

| File | Source |
|------|--------|
| `voxelnext_nuscenes.pth` | `/Storage/models/voxelnext/` on recording laptop |
| `LiDAR-HMR/ckpts/` | `~/Downloads/LiDAR-HMR ckpts/ckpts/` on recording laptop |
| `SMPL_NEUTRAL.pkl` | `LiDAR-HMR/body_models/smpl/` (licensed — request from MPI-IS) |

## Running

### Full perception stack (no loco control)
```bash
ROS_DOMAIN_ID=42 \
  DETECTION_CKPT=/path/to/voxelnext_nuscenes.pth \
  HMR_CKPTS=/path/to/LiDAR-HMR/ckpts \
  SMPL_PKL=/path/to/SMPL_NEUTRAL.pkl \
  docker compose -f docker/compose.yaml up sensors detection hmr reid
```

### With bag recording
```bash
BAG_DIR=/path/to/bags ROS_DOMAIN_ID=42 \
  docker compose -f docker/compose.yaml --profile record \
  up sensors detection hmr reid record
```

### Individual containers (manual)
```bash
# Detection
docker run --rm --gpus all --network host \
  -v /path/to/voxelnext_nuscenes.pth:/checkpoints/voxelnext_nuscenes.pth:ro \
  -e ROS_DOMAIN_ID=42 \
  thdhyan/g1-detection:latest

# HMR
docker run --rm --gpus all --network host \
  -v /path/to/LiDAR-HMR/ckpts:/ws/LiDAR-HMR/ckpts:ro \
  -v /path/to/SMPL_NEUTRAL.pkl:/ws/LiDAR-HMR/body_models/smpl/SMPL_NEUTRAL.pkl:ro \
  -e ROS_DOMAIN_ID=42 \
  thdhyan/g1-hmr:latest

# Sensors (lidar_bridge — connects to Mid-360)
docker run --rm --network host \
  -e ROS_DOMAIN_ID=42 \
  thdhyan/g1-sensors:latest

# ReID server
docker run --rm --network host \
  -e ROS_DOMAIN_ID=42 \
  thdhyan/g1-reid:latest
```

## Blackwell (sm_100) Notes

The GPU images are built with:
```
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
```

The `9.0+PTX` embeds PTX (intermediate representation). On the RTX 5090, the
CUDA 12.8+ driver JIT-compiles this PTX to native sm_100 on **first use of each
kernel** (~30 s one-time overhead per process). Compiled kernels are cached at
`/root/.cache/torch/` inside the container.

To persist the PTX cache across container restarts:
```bash
mkdir -p ~/.cache/torch_docker
docker run --rm --gpus all --network host \
  -v ~/.cache/torch_docker:/root/.cache/torch \
  -v /path/to/voxelnext_nuscenes.pth:/checkpoints/voxelnext_nuscenes.pth:ro \
  -e ROS_DOMAIN_ID=42 \
  thdhyan/g1-detection:latest
```

## Image Summary

| Image | GPU | Purpose |
|-------|-----|---------|
| `thdhyan/g1-sensors:latest` | ❌ | lidar_bridge (Mid-360 → `/livox/lidar`) |
| `thdhyan/g1-detection:latest` | ✅ | VoxelNeXt pedestrian detection |
| `thdhyan/g1-hmr:latest` | ✅ | LiDAR-HMR body mesh + BetaTracker |
| `thdhyan/g1-reid:latest` | ❌ | ReID server (cosine EMA matching) |
| `thdhyan/g1-control:latest` | ❌ | cmd_vel_bridge + human_follower |

Base: `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` + ROS Jazzy
torch 2.5.1+cu124 · spconv-cu124 · PyG torch-2.5.0+cu124
