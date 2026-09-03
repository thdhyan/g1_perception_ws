# Docker containers — g1_perception_ws

Each container has its own ROS distro + Python. They communicate over DDS
(`--network host`, `ROS_DOMAIN_ID=42`), exactly like the native stack.

## Containers

| Image | Nodes | GPU | Platform |
|---|---|---|---|
| `g1_detection` | livox_detection_node (VoxelNeXt) | required | x86 or ARM64 Jetson |
| `g1_hmr` | smpl_hmr_node, reid_server_node | required | x86 or Jetson Orin ≥16GB |

## x86_64 (this laptop)

```bash
# Detection
docker build -f docker/Dockerfile.detection -t g1_detection:jazzy .

# HMR
docker build -f docker/Dockerfile.hmr -t g1_hmr:jazzy .

# Run both (robot must be publishing /livox/lidar on ROS_DOMAIN_ID=42)
docker run --rm --network host --gpus all \
  -v $(pwd)/pt/voxelnext_nuscenes.pth:/checkpoints/voxelnext_nuscenes.pth:ro \
  -e ROS_DOMAIN_ID=42 g1_detection:jazzy &

docker run --rm --network host --gpus all \
  -v $(pwd)/LiDAR-HMR/ckpts:/models/ckpts:ro \
  -v /path/to/SMPL_NEUTRAL.pkl:/ws/LiDAR-HMR/body_models/smpl/SMPL_NEUTRAL.pkl:ro \
  -e ROS_DOMAIN_ID=42 g1_hmr:jazzy
```

## Jetson Orin (ARM64, JetPack 6 / CUDA 12.2)

Prerequisite: `nvidia-container-runtime` installed on Jetson.

```bash
# Detection — build on Jetson (or cross-compile with buildx)
docker build \
  --build-arg BASE=dustynv/ros:humble-ros-base-l4t-r36.2.0 \
  --build-arg TORCH_URL=https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+6ddf5cf85e.nv24.04-cp310-cp310-linux_aarch64.whl \
  -f docker/Dockerfile.detection -t g1_detection:orin .

# HMR — note BUILD_PYG=1 compiles torch-scatter from source (~30 min)
docker build \
  --build-arg BASE=dustynv/ros:humble-ros-base-l4t-r36.2.0 \
  --build-arg TORCH_URL=https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0a0+6ddf5cf85e.nv24.04-cp310-cp310-linux_aarch64.whl \
  --build-arg BUILD_PYG=1 \
  -f docker/Dockerfile.hmr -t g1_hmr:orin .

docker run --rm --network host --runtime nvidia \
  -v /path/to/checkpoints:/checkpoints:ro \
  -e ROS_DOMAIN_ID=42 g1_detection:orin

docker run --rm --network host --runtime nvidia \
  -v /path/to/ckpts:/ws/LiDAR-HMR/ckpts:ro \
  -e ROS_DOMAIN_ID=42 g1_hmr:orin
```

## Jetson Xavier / Orin JetPack 5 (ARM64, CUDA 11.4)

```bash
docker build \
  --build-arg BASE=dustynv/ros:humble-ros-base-l4t-r35.4.1 \
  --build-arg TORCH_URL=https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl \
  --build-arg BUILD_PYG=1 \
  -f docker/Dockerfile.detection -t g1_detection:xavier .
```

## Feasibility notes

| Concern | Status |
|---|---|
| spconv on ARM64 | Build from source — works, ~15 min |
| torch-scatter ARM64 | No PyPI wheel; build from source — ~30 min |
| SMPL_NEUTRAL.pkl | Licensed file, not in image — mount at runtime with `-v` |
| GPU access Jetson | Use `--runtime nvidia` (not `--gpus all`) on JetPack ≤5 |
| HMR on Xavier | Technically feasible; 11ms/person with fp32 reported on Orin, Xavier ~3× slower |
| AMP/fp16 on Jetson | Same issue as laptop (addmm_sparse_cuda); TF32 only on Ampere (Orin), not Xavier |
