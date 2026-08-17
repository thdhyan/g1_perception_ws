# PointPillar Detection Backend

## Overview

PointPillar is a fast 3D object detector based on pillar-based voxelization (Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds", CVPR 2019). It voxelizes the point cloud into vertical pillars (2D grid cells), encodes each pillar's points into a feature vector, scatters these features back onto a pseudo-image, and runs a 2D CNN backbone + SSD detection head.

**Key characteristics:**
- **Speed**: ~10-20 ms inference on modern GPUs (vs ~50 ms for CenterPoint), suitable for edge/onboard robotics
- **Accuracy**: Competitive with CenterPoint for vehicle/pedestrian detection with proper calibration
- **Simplicity**: No sparse convolutions required; pure dense Conv2d operations
- **Voxel-based**: Inherently handles variable point density better than point-wise approaches

## Architecture

### Components

1. **PillarFeatureNet** (`pointpillar_model.py:PillarFeatureNet`)
   - Encodes points in each pillar into a single feature vector
   - Computes per-point features: coordinates, offsets from pillar center, mean statistics
   - Max-pools over all points in the pillar to produce a pillar-level descriptor

2. **Backbone2D** (`pointpillar_model.py:Backbone2D`)
   - Multi-scale 2D CNN: 4 stride-2 blocks downsample to 1/16 resolution
   - 4 transposed convolution layers upsample and fuse features back to input resolution
   - Creates a dense feature map from the sparse pillar pseudo-image

3. **SSDHead** (`pointpillar_model.py:SSDHead`)
   - Shared conv layers followed by task-specific heads
   - Heatmap head: K anchors × C classes (vehicle, pedestrian, cyclist)
   - Regression head: K anchors × 7 (x, y, z, dx, dy, dz, yaw in world coordinates)

### Voxelization

```
Point cloud (x, y, z, intensity)
    ↓
Grid pillars in XY plane (voxel_size=0.2m, 1120×448 grid)
    ↓
Encode each pillar → feature vector (64 channels)
    ↓
Scatter to pseudo-image: (1, 64, 448, 1120)
    ↓
Backbone + SSD head → detections
```

Detection range: (0.0, -44.8, -2.0) to (224.0, 44.8, 4.0) meters in sensor frame.

## Usage

### Prerequisites

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install rclpy sensor-msgs-py vision-msgs
```

### Running the Node

```bash
# Launch with default parameters
ros2 run g1_perception pointpillar_node

# Custom checkpoint
ros2 run g1_perception pointpillar_node --checkpoint /path/to/model.pt

# Adjust inference rate (default 10 Hz, suitable for onboard inference)
ros2 run g1_perception pointpillar_node --max-hz 15.0

# Fall back to clustering if checkpoint missing
ros2 run g1_perception pointpillar_node --checkpoint /nonexistent/path.pt
```

### Via Launch File

```bash
ros2 launch g1_perception pointpillar.launch.py checkpoint:=/custom/path/model.pt max_hz:=15.0
```

### Topic Interface

**Input:**
- `/livox/mid360/points` (`sensor_msgs/PointCloud2`): LiDAR point cloud from simulator or real robot (bridged by `lidar_bridge` node)

**Output:**
- `/g1/detections/pointpillar` (`vision_msgs/Detection3DArray`): Detected 3D boxes with class and score
- `/g1/detection_markers/pointpillar` (`visualization_msgs/MarkerArray`): RViz visualization (cube markers, color-coded by class)

### ROS2 Parameters

- `checkpoint` (string): Path to model checkpoint. Default: `G1_sim/detection/pt/pointpillar_model.pt`
- `device` (string): Torch device (cuda, cpu). Default: `cuda`
- `score_threshold` (float): Minimum detection score (0–1). Default: `0.4`
- `input_topic` (string): Input PointCloud2 topic. Default: `/livox/mid360/points`
- `max_hz` (float): Maximum inference frequency (Hz). Default: `10.0`
- `frame` (string): Override frame_id for published detections (empty = use cloud's frame). Default: (empty)

## Fallback Behavior

If the model checkpoint is not found or fails to load, the node automatically falls back to a simple **Euclidean clustering** backend:
- Filters points by height (-1.6 to 1.2 m) and radial distance (0.5 to 25 m)
- Voxelizes in XY plane (0.2 m voxels)
- Fits bounding boxes to clusters with person-like proportions (0.8–2.2 m tall, < 1.2 m wide)
- Publishes with confidence 0.5 (placeholder)

This allows the node to run end-to-end for validation and integration testing without requiring model weights.

## Comparison with CenterPoint

| Aspect | PointPillar | CenterPoint |
|--------|-------------|------------|
| **Voxelization** | Pillar (2D grid) | Voxel (3D grid) |
| **Backbone** | Dense 2D CNN | Sparse 3D convolution |
| **Inference time** | ~10–20 ms | ~50–100 ms |
| **Accuracy** | Good for vehicles/pedestrians | Excellent; handles diverse objects |
| **Dependencies** | PyTorch (pure dense ops) | PyTorch + SPCONV or MinkowskiNet |
| **Edge suitability** | Excellent (light) | Good (heavier) |
| **Tuning** | Simpler (fewer hyperparams) | More complex (sparse conv tuning) |

**When to use PointPillar:**
- Onboard/edge inference where latency is critical
- Vehicle and pedestrian detection on 360° LiDAR (Livox Mid-360)
- When GPU memory is limited
- Fast iteration / prototyping

**When to use CenterPoint:**
- Maximum accuracy required
- Diverse object categories or scale ranges
- Sufficient computational budget

## Model Checkpoint Format

PointPillar checkpoints are PyTorch state dicts (`.pt` files). To save or convert a model:

```python
import torch
from g1_perception.pointpillar_model import PointPillar

model = PointPillar(num_classes=3, num_anchors=2)
# ... train or load from another format ...
torch.save(model.state_dict(), "pointpillar_model.pt")

# Later, load in the node:
state = torch.load("pointpillar_model.pt", map_location="cuda", weights_only=False)
model.load_state_dict(state)
```

## Debugging

**No detections published:**
1. Check ROS topics: `ros2 topic list | grep -E 'livox|detections'`
2. Monitor: `ros2 topic hz /livox/mid360/points`
3. Node logs: `ros2 run g1_perception pointpillar_node 2>&1 | grep -i "warn\|error"`

**High latency:**
- Reduce `max_hz` if GPU is saturated
- Check GPU memory: `nvidia-smi`
- Verify CUDA is enabled: `python3 -c "import torch; print(torch.cuda.is_available())"`

**Misaligned markers in RViz:**
- Ensure lidar_bridge is republishing with correct frame_id (`mid360_link` for sim)
- Check frame offset in the frame broadcaster

## References

- Lang, A. H., Deng, S., Levine, S., & Feiszli, M. (2019). PointPillars: Fast Encoders for Object Detection from Point Clouds. CVPR.
- [OpenPCDet Implementation](https://github.com/open-mmlab/OpenPCDet)
