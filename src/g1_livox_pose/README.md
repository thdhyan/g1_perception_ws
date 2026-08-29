# g1_livox_pose

3D human pose estimation from Livox LiDAR point clouds, assembled into
**time-continuous per-person skeleton sequences** in a stable reference frame
— designed for logging and future skeleton-token transformer training
(next-action prediction / intent classification).

Phase 1 ships the full plumbing with a zero-ML **debug backend**: it projects
a deterministic standing figure into each detection box so crop → infer →
TF-at-capture-stamp → tracking → sequences can be verified end-to-end before
any real network is wired in (VoxelKP / DAPT come in Phase 2/3).

## Architecture

```
[livox_detection snapshot pipeline] ──► /livox/collected_points (latched)
                                   └──► /g1/sorted_humans        (latched)
                                                │
                                  human_pose_node (backend: debug|dapt|voxelkp)
                                                │  crops ROI cloud per human,
                                                │  TF @ capture stamp
                                                ▼
                                     /g1/human_poses (PersonPose3DArray)
                                                │
                        pose_sequence_assembler_node
                          • nearest-neighbor association → track ids
                          • re-transform pelvis@t_i → odom@t_i (per-frame stamp)
                          • ring buffer per track
                          ► /g1/human_pose_sequences (SkeletonSequenceArray, latched)
                          ► /g1/skeleton_markers     (LINE_LIST + root trail, latched)
                          ► optional JSONL log (log_path param)
```

### Why `odom` as the sequence frame

Poses in `pelvis` live in a *moving* frame — successive frames would encode
robot ego-motion instead of human motion. The assembler re-transforms every
pose into `sequence_frame` (default `odom`) using TF at that pose's own
stamp, so the root-joint trajectory captures real approach/retreat/walking.
Requires SLAM/odometry publishing TF; poses are dropped (with throttled
warnings) when the transform is unavailable rather than corrupting the
sequence with mixed frames.

## Messages

- `PersonPose3D` — one person, one instant: header (capture stamp), track_id,
  `joints` = flattened K×3, `valid` per-joint flags, pose_score.
- `PersonPose3DArray` — multi-person at one instant.
- `SkeletonSequence` — N time-ordered frames of ONE tracked person in the
  stable sequence frame; `window` = newest − oldest.
- `SkeletonSequenceArray` — all current tracks.

Joint convention: **Waymo 14** (`nose, left_shoulder, left_elbow, left_wrist,
left_hip, left_knee, left_ankle, right_shoulder, right_elbow, right_wrist,
right_hip, right_knee, right_ankle, head`) — matches LPFormer/DAPT/VoxelKP;
defined once in `common.py::JOINT_NAMES`, bones in `BONES`.

## Running (sim)

```bash
# 1. Snapshot pipeline (produces collected_points + sorted_humans)
ros2 launch livox_detection snapshot_pipeline.launch.py

# 2. Pose pipeline (debug backend by default)
ros2 launch g1_livox_pose pose_pipeline.launch.py backend:=debug

# Optional: log to JSONL for training data
ros2 launch g1_livox_pose pose_pipeline.launch.py log_path:=/tmp/poses.jsonl

# Inspect sequences
ros2 topic echo /g1/human_pose_sequences --once
```

RViz: add a MarkerArray display on `/g1/skeleton_markers` (fixed frame =
your SLAM world/odom frame) to see skeletons + colored root trajectories.

## Key parameters

### human_pose_node

| Param | Default | Notes |
|---|---|---|
| `backend` | `debug` | pluggable; registry in `backends/__init__.py` |
| `input_cloud_topic` | `/livox/collected_points` | latched snapshot cloud |
| `input_detections_topic` | `/g1/sorted_humans` | pedestrian-filtered Detection3DArray |
| `target_frame` | `pelvis` | joints transformed here via TF **at the cloud stamp** |
| `crop_margin` | `0.30` | extra meters around bbox for point cropping |
| `min_crop_points` | `20` | skip humans with fewer cropped points |

### pose_sequence_assembler_node

| Param | Default | Notes |
|---|---|---|
| `sequence_frame` | `odom` | stable gravity-aligned output frame |
| `max_frames` | `150` | ring-buffer length per track (150 @5Hz ≈ 30 s window) |
| `gate_speed_mps` | `1.5` | association gate = speed × dt (capped below) |
| `max_gate_m` | `3.0` | absolute association cap |
| `track_timeout_sec` | `0.8` | drop tracks unseen this long |
| `log_path` | `""` | append-only JSONL of accepted poses (empty = off) |

## Status

- [x] Phase 1: scaffold + messages + debug backend + assembler + markers + JSONL
- [ ] Phase 2: single-frame verification harness (reprojection residual, bone-length stability, jitter)
- [ ] Phase 2/3: real backends (VoxelKP whole-scene via OpenPCDet, DAPT on crops)
- [ ] Phase 4: temporal smoothing, deskew, fine-tuning
