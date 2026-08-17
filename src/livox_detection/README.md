# livox_detection

3D object detection on Livox Mid-360 point clouds for the G1 perception
stack. One node (`livox_detection_node`), three swappable backends:
CenterPoint, PointPillars, and **VoxelNeXt** (recommended — see
[VOXELNEXT.md](../../VOXELNEXT.md) at the repo root for architecture/class
mapping details). This README covers running and tuning the node itself.

## Quick start (sim)

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch g1_bringup sim_teleop.launch.py rviz:=true detection_algorithm:=voxelnext slam:=false
```

Detections publish on `/g1/detections/voxelnext` (`vision_msgs/Detection3DArray`,
in the `pelvis` frame) and `/g1/detection_markers/voxelnext` (`MarkerArray`
for RViz — also mirrored to `/g1/detection_markers/livox`, which is what
`g1_sim_mapping.rviz` already has a display for, so no RViz config changes
are needed to see boxes).

## Backend checkpoints

Each backend needs its own checkpoint — they are **not** interchangeable:

| Algorithm | Checkpoint | Format |
|---|---|---|
| `voxelnext` | `pt/voxelnext_nuscenes.pth` (workspace root) | OpenPCDet (`model_state` key) |
| `centerpoint` / `pointpillar` | `checkpoint_path` param, default `~/Projects/thesis/G1_sim/detection/pt/livox_model_1.pt` | custom (raw state dict) |

`sim.launch.py`/`sim_teleop.launch.py` select the right one automatically
based on `detection_algorithm` — if you're launching the node standalone
(not through those launch files), pass `checkpoint_path` explicitly or it
will default to the CenterPoint/PointPillar path and VoxelNeXt will fail to
load (`load_params_from_file` raises `KeyError('model_state')`), silently
falling back to the PointPillar clustering heuristic. Symptom: a suspiciously
constant `max_score` across every frame in the log — that means it isn't
actually running.

## Key parameters

| Param | Default | Notes |
|---|---|---|
| `algorithm` | `centerpoint` | `centerpoint` \| `pointpillar` \| `voxelnext` |
| `checkpoint_path` | see above | overridden per-algorithm by the sim launch files |
| `class_filter` | `pedestrian` | comma-separated `CLASS_NAMES` to keep (`car`, `pedestrian`, `cyclist`); empty string disables filtering |
| `score_threshold` | `0.10` | per-box confidence cutoff |
| `max_hz` | `10.0` (`5.0` in sim launch files) | inference rate cap — the node skips a cycle rather than queuing if the previous one is still running (`_busy` guard), so this is a ceiling, not a guarantee |
| `accumulate_frames` | `4` (`2` in sim launch files) | how many LiDAR scans get concatenated into one inference call |
| `max_distance` | `25.0` | drop detections beyond this 2D range (meters) |
| `target_frame` | `pelvis` | detections are TF-transformed into this frame before publishing |

### Matching `accumulate_frames` to `max_hz`

The Mid-360 `gpu_lidar` sensor in sim runs at 10Hz (100ms/scan). At
`max_hz=5.0` (200ms cadence), `accumulate_frames` should be `2` (200ms of
data per inference) — not more, or each inference call re-processes stale
points from the previous cycle's window. The sim launch files set both
together for this reason.

### Achieved rate vs `max_hz`

`max_hz` is a cap, not a guarantee. Measured on this dev box with VoxelNeXt:
~4.0-4.2Hz achieved against a 5.0Hz cap — real inference latency
occasionally exceeds the 200ms budget, and the node correctly skips that
cycle (via `_busy`) rather than backing up a queue. If you need the full
5Hz, the actual lever is inference cost (smaller voxel grid, fewer
`accumulate_frames`, TensorRT, or a lighter backend), not the `max_hz` param.

## Verified behavior (sim, this repo's `g1_warehouse.sdf`)

- `class_filter: "pedestrian"` correctly drops `car`/`cyclist` — confirmed
  zero non-pedestrian detections across 100+ consecutive frames.
- Bounding boxes render correctly positioned on the LiDAR point cloud in
  RViz, with per-box `pedestrian (score) [distance in pelvis]` labels.
- Known false-positive source: the stock nuScenes-pretrained VoxelNeXt
  checkpoint occasionally flags warehouse shelving/geometry as `car` when
  `class_filter` is disabled — expected, since the checkpoint isn't
  fine-tuned for this scene; filtering to `pedestrian` sidesteps it.
- Multiple overlapping `pedestrian` boxes on the same person are possible at
  low score thresholds (no NMS-across-classes tuning done yet) — raise
  `score_threshold` or add NMS if this matters for your use case.
