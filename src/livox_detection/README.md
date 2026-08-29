# livox_detection

3D object detect on Livox Mid-360 point clouds, G1 perception stack.
One node (`livox_detection_node`), single detection backend:
**VoxelNeXt** (see [VOXELNEXT.md](../../VOXELNEXT.md) at repo root for
architecture/class mapping detail). README covers run + tune node.

## Quick start (sim)

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch g1_bringup sim_teleop.launch.py rviz:=true detection_algorithm:=voxelnext slam:=false
```

Detections publish on `/g1/detections/livox` (and alias
`/g1/detections/voxelnext`) as `vision_msgs/Detection3DArray` in the
`pelvis` frame, and markers on `/g1/detection_markers/livox` +
`/g1/detection_markers/voxelnext` (`MarkerArray` for RViz —
`g1_sim_mapping.rviz` already has a display for the `/livox` names, so
no RViz config change needed to see boxes).

## Backend checkpoint

VoxelNeXt needs its own checkpoint — default is
`pt/voxelnext_nuscenes.pth` (workspace root), OpenPCDet format
(`model_state` key). Pass `checkpoint_path` explicitly to override.

If the model fails to load (e.g. `pcdet` not built from `VoxelNeXt/`,
or `spconv-cu121` missing), the node logs an error and publishes
**empty** detections — you will see a constant zero-detection log;
that means the backend is not active, not a clean 0-detection scene.

## Key parameters

| Param | Default | Notes |
|---|---|---|
| `algorithm` | `voxelnext` | only `voxelnext` is supported (older values are ignored with a warning) |
| `checkpoint_path` | `~/Projects/thesis/g1_perception_ws/pt/voxelnext_nuscenes.pth` | OpenPCDet `.pth` |
| `voxelnext_cfg` | `VoxelNeXt/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml` | model config |
| `voxelnext_dir` | `VoxelNeXt/` (workspace root) | cloned VoxelNeXt repo (provides `pcdet`) |
| `class_filter` | `pedestrian` | comma-separated `CLASS_NAMES` to keep (`car`, `pedestrian`, `cyclist`); empty string disables filtering |
| `score_threshold` | `0.10` | per-box confidence cutoff |
| `max_hz` | `10.0` (`5.0` in sim launch files) | inference rate cap — node skips a cycle rather than queue if previous is still running (`_busy` guard), so the cap is not a guarantee |
| `accumulate_frames` | `4` (`2` in sim launch files) | how many LiDAR scans are concatenated into one inference call |
| `max_distance` | `25.0` | drop detections beyond this 2D range (meters) |
| `target_frame` | `pelvis` | detections TF-transform into this frame before publish |

### Match `accumulate_frames` to `max_hz`

Mid-360 `gpu_lidar` sensor in sim runs 10Hz (100ms/scan). At
`max_hz=5.0` (200ms cadence), `accumulate_frames` should be `2` (200ms
of data per inference) — not more, else each inference call re-processes
stale points from the previous cycle window. Sim launch files set both
together for this reason.

### Achieved rate vs `max_hz`

`max_hz` is a cap, not a guarantee. Measured on this dev box with
VoxelNeXt: ~4.0-4.2Hz achieved against the 5.0Hz cap — real inference
latency occasionally exceeds the 200ms budget, and the node correctly
skips a cycle (via `_busy`) rather than backing up the queue.
Need full 5Hz? The actual lever is inference cost (smaller voxel grid,
fewer `accumulate_frames`, TensorRT), not the `max_hz` param.

## Verified behavior (sim, this repo's `g1_warehouse.sdf`)

- `class_filter: "pedestrian"` correctly drops `car`/`cyclist` — confirmed
  zero non-pedestrian detections across 100+ consecutive frames.
- Bounding boxes render at correct positions on the LiDAR point cloud in
  RViz, with per-box `pedestrian (score) [distance in pelvis]` labels.
- Known false-positive source: the stock nuScenes-pretrained VoxelNeXt
  checkpoint occasionally flags warehouse shelving/geometry as `car` when
  `class_filter` is disabled — expected, the checkpoint is not fine-tuned
  for this scene; filtering to `pedestrian` sidesteps it.
- Multiple overlapping `pedestrian` boxes on the same person are possible
  at low score thresholds (no NMS-across-classes tuning done yet) —
  raise `score_threshold` or add NMS if it matters for your use case.
