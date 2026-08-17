# CenterPoint Detection Node

The `centerpoint_node` runs the ported CenterPoint model from `G1_sim/detection/livox_centerpoint.py` directly in this ROS2 workspace, subscribing to `/livox/mid360/points` and publishing 3D bounding box detections.

## Architecture

**CenterPoint** differs from **PointPillar** (another agent is adding a PointPillar node) in how it encodes and decodes the point cloud:

| Aspect | CenterPoint | PointPillar |
|--------|-------------|------------|
| **Input** | 30-channel Bird's-Eye-View (BEV) binary occupancy map (z-binned) | BEV pseudo-image from pillar encoding |
| **Backbone** | Multi-scale FPN (5 stride-2 blocks, 32→128 channels) with channel-attention fusion | Typically simpler conv stack |
| **Detection Head** | CenterPoint heatmap + 7 regression targets (center_xy, center_z, dim_xyz, yaw) | Anchor-based or center-based detection |
| **NMS** | Greedy BEV-space NMS on axis-aligned footprints | Usually 3D IoU-based |
| **Classes** | 3: car, pedestrian, cyclist | Often includes more classes |

CenterPoint's strength is in dense pedestrian detection via its heatmap-based approach; the 30-layer z-binning captures height information without explicit 3D convolution. The fixed BEV occupancy encoding is memory-efficient and runs on any PyTorch installation without spconv or custom CUDA ops.

## Usage

### Launch with defaults

```bash
ros2 launch g1_perception centerpoint.launch.py
```

This will:
1. Load the checkpoint from `G1_sim/detection/pt/livox_model_1.pt` (resolved relative to G1_sim if checkpoint_path is relative)
2. Subscribe to `/livox/mid360/points`
3. Publish detections to `/g1/detections/centerpoint` (Detection3DArray)
4. Publish RViz markers to `/g1/detection_markers/centerpoint` (MarkerArray)
5. Limit inference to 5 Hz (configurable with `max_hz` parameter)

### Launch with custom parameters

```bash
ros2 launch g1_perception centerpoint.launch.py \
  checkpoint_path=/path/to/custom_model.pt \
  max_hz:=10.0 \
  score_threshold:=0.5 \
  device:=cpu
```

**Parameters:**
- `checkpoint_path` (default: `pt/livox_model_1.pt`) — Model weights. If relative, resolved relative to `G1_sim/detection`.
- `max_hz` (default: `5.0`) — Maximum inference frequency in Hz. Frames arriving faster than this are dropped to prevent GPU/CPU overload.
- `score_threshold` (default: `0.4`) — Detection confidence threshold in [0, 1]; lower scores are suppressed.
- `device` (default: `cuda`) — PyTorch device (`cuda` or `cpu`).
- `input_topic` (default: `/livox/mid360/points`) — Input PointCloud2 topic.
- `frame_override` (default: `""`) — Optional frame_id override; if empty, uses the incoming cloud's frame_id.

### With simulated data (Isaac Sim)

Isaac Sim publishes `/livox/mid360/points` directly. Simply launch the node and it will subscribe automatically:

```bash
ros2 launch g1_perception centerpoint.launch.py
```

### With real robot data

The `lidar_bridge` (in the same package) republishes the real robot's native LiDAR topic onto `/livox/mid360/points`. Run both bridges together:

```bash
ros2 launch g1_perception lidar_bridge.launch.py  # if it exists
ros2 launch g1_perception centerpoint.launch.py
```

## Checkpoint Loading

The node gracefully handles missing checkpoints:
- **Checkpoint found:** Loads it at startup, runs inference normally.
- **Checkpoint missing:** Logs a warning, skips inference, publishes empty Detection3DArray and MarkerArray every frame so subscribers don't block.

This allows testing the topic graph and RViz visualization without model weights. To enable inference later, update the `checkpoint_path` parameter and relaunch.

## Output Topics

### `/g1/detections/centerpoint` (Detection3DArray)

Standard ROS2 detection message. Each detection contains:
- **class_id** — string: `"car"`, `"pedestrian"`, or `"cyclist"`
- **score** — float in [0, 1], the model's confidence
- **bbox** — 3D axis-aligned box in the sensor frame (not rotated; see below)
  - **center** — position (x, y, z) + quaternion rotation encoding yaw only
  - **size** — dimensions (dx, dy, dz) in meters

The quaternion is yaw-only (roll and pitch are zero) because the LiDAR ground plane is flat and the model doesn't regress those angles. The yaw is decoded from sine/cosine outputs.

### `/g1/detection_markers/centerpoint` (MarkerArray)

RViz visualization. Each frame publishes a DELETEALL marker followed by one CUBE per detection. Cubes are translucent (alpha=0.4) so the point cloud remains visible. Colors:
- **Car:** cyan (0, 1, 1)
- **Pedestrian:** yellow (1, 1, 0)
- **Cyclist:** green (0, 1, 0)

## Performance

On a Livox Mid360 at ~150k points/frame:
- Inference: ~100–150 ms per frame (NVIDIA GPU, torch 2.7, CUDA 12.6)
- Memory: ~2 GB GPU (model + BEV tensor buffer)
- Rate limiting: Defaults to 5 Hz to keep latency under 200 ms; increase `max_hz` if inference is faster

The node processes frames sequentially (busy flag) and drops incoming clouds while processing, so the effective rate is bounded by both `max_hz` and inference latency.

## Troubleshooting

### "Checkpoint not found at ..."
Set `checkpoint_path` to an absolute path or place `pt/livox_model_1.pt` in `G1_sim/detection/`.

### No detections or very few
Check `score_threshold` (default 0.4). Lower it to see more candidates, or increase it to be more selective. Also verify that the input point cloud is populated (RViz point cloud display, or `ros2 topic echo /livox/mid360/points`).

### GPU OOM or slow inference
Reduce `max_hz` to lower the inference frequency, or set `device:=cpu` for slower but lower-memory fallback.

### Frame_id mismatch
Use `frame_override:=<frame>` to force a specific frame name in the output, or rely on the incoming cloud's frame_id if it's already correct.
