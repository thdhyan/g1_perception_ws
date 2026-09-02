# Research: Indoor LiDAR 3D Object Detectors

**Date**: 2026-09-01
**Context**: G1 perception — Livox Mid-360 (200k pts/s, 360°×59°), indoor ≤15m, target Jetson Orin 16GB
**Scope**: Detect humans + common indoor objects (chairs, tables, shelves, robots)

---

## 1. Key Finding: The Indoor Domain Gap

Most published LiDAR detectors are trained on outdoor driving data (KITTI, nuScenes, Waymo). This creates a **significant domain gap** for indoor use:

| Challenge | Impact |
|-----------|--------|
| Point density | Outdoor datasets (32+ beam, 30-100k pts) → much denser than Mid-360 indoor (~200k pts in 360°×59° at ≤15m) |
| Object scale | Cars/trucks (4-8m) vs. people (0.4-0.8m wide) → 10x smaller targets |
| Occlusion patterns | Vehicles on open roads vs. cluttered indoor (furniture, shelves) |
| Scan pattern | Velodyne/Ouster concentric rings vs. Livox non-repetitive |
| Range | Detectors trained for 30-100m; indoor objects at 0.5-15m |

**Critical paper** (arXiv 2106.11239): CenterPoint trained on nuScenes achieves only **9.9% AP_box** on JRDB (indoor robot dataset). After fine-tuning on JRDB: **58.2% AP_box**. With fine-grained voxelization + nuScenes pretraining: **70.0% AP_box**.

**Implication**: Pretrained models transfer *some* useful representations, but indoor fine-tuning is mandatory. Training from scratch on indoor data gets comparable results (60.0% AP_box on JRDB).

---

## 2. Livox Mid-360 Scan Pattern: What It Means for Detectors

The Mid-360 uses **non-repetitive scanning** (spirograph pattern). Key implications:

### 2.1 Voxelization / Pillarization Compatibility

- **PointPillars / PillarNet**: Uses vertical "pillars" in BEV space. These assign points to grid cells in X-Y regardless of Z distribution. **Works with Livox** — points just land in different pillars. No issue.
- **SECOND / VoxelNeXt**: 3D sparse voxelization. Points are assigned to 3D voxel grid. **Works with Livox** — the voxelization is coordinate-based, not scan-line-based. No issue.
- **PillarNeXt**: Same as PointPillars. **Works**.
- **PointRCNN**: Operates directly on points (PointNet++). **Works** — scan-agnostic.

**Bottom line**: All these detectors operate on (x,y,z) coordinates, not on "scan lines." The non-repetitive pattern **does not break voxelization or pillarization**. The only effect is the spatial *distribution* of points differs — some pillars/voxels get more points, some get fewer. This is handled naturally by sparse convolutions.

### 2.2 Evidence from Literature

- **"Which LiDAR scanning pattern is better for roadside perception?" (arXiv 2025)**: Evaluated PointPillars, PointRCNN, PV-RCNN, DSVT on Livox Avia (non-repetitive) vs. 16/64/128-line (repetitive). Found: non-repetitive LiDAR + PointPillars = **91.37 AP** vs 128-line + PointPillars = **92.43 AP** (comparable!). Non-repetitive LiDAR + PV-RCNN = **92.58 AP** vs 128-line = **93.29 AP**. **Conclusion**: Non-repetitive pattern is NOT a blocker. All standard detectors work.

- **MMDetection3D Issue #3105**: User asking about MID-360 + PointPillars. Response confirms no architectural issue — just need to adjust point cloud range params.

- **OpenPCDet**: Uses standard `.bin` format (x,y,z,intensity,timestamp). Works with any coordinate-based LiDAR.

### 2.3 Practical Notes for Mid-360

- **Point range**: Set detection range to ~±15m (not ±75m like Waymo). Smaller range → better voxel resolution → faster inference → better indoor detection.
- **Voxel size**: Use 0.05m or 0.075m (not 0.1m+). Indoor objects are small.
- **Vertical range**: -2m to +3m covers standing human + furniture.
- **Point density**: 200k pts at 10Hz = 20M pts/s. For indoor ≤15m, expect ~5-20k pts per frame depending on scene complexity.

---

## 3. Per-Detector Analysis

### 3.1 PointPillars

| Aspect | Assessment |
|--------|-----------|
| Indoor suitability | **Good** — lightweight, fast, works at various scales. Trained on car-size objects but can be retuned for humans. |
| Livox compatible | **Yes** — coordinate-based pillarization. |
| Jetson Orin | **Yes** — 42 FPS TensorRT FP16 on AGX Orin (DKrishna trained on "custom indoor LiDAR dataset"). 18.27ms FP16, 14.77ms INT8 on Orin (from 2026 TensorRT benchmarks). |
| RTX 4060 | **Excellent** — 36-37 FPS FP16 (from Jetson benchmarks scaled) |
| GPU req | ~2-4 GB VRAM. Fine on 8GB RTX 4060. |
| ROS2 | **Yes** — `NVIDIA-AI-IOT/ros2_tao_pointpillars` (TensorRT), `DKrishna007/pointpillars-3d-detection-jetson`, `ragibarnab/ros2-lidar-object-detection` (PyTorch wrapper) |
| Drawback | Lower accuracy than voxel methods for small objects. Designed for BEV detection → poor height/size regression for small objects. |
| Key paper | Lang et al. 2019 "End-to-end 3D Object Detection from Point Clouds" |
| **Recommendation** | **Best for Jetson deployment.** Fastest, most battle-tested, best ROS2 support. Accept 5-10% AP tradeoff vs SECOND/VoxelNeXt. |

**Jetson Orin numbers (measured)**:
- FP32: 32.91ms (30 FPS)
- FP16: **18.27ms (55 FPS)**  ← TensorRT
- INT8: **14.77ms (68 FPS)**  ← TensorRT
- From DKrishna: **42 FPS** (TensorRT FP16, indoor dataset, 78.4% MOTA with ByteTrack)

### 3.2 SECOND

| Aspect | Assessment |
|--------|-----------|
| Indoor suitability | **Very good** — sparse 3D convolution captures full 3D structure. 95% AP on solid-state LiDAR pedestrian detection (Basile et al. 2024, Jetson AGX). |
| Livox compatible | **Yes** — 3D sparse voxelization, coordinate-based. |
| Jetson Orin | **Yes** — 10.3 FPS on Jetson AGX Xavier (Basile et al. 2024). On Orin (2x perf): expect **~15-20 FPS** with TensorRT. |
| RTX 4060 | **Good** — ~20-30 FPS (from OpenPCDet benchmarks on consumer GPUs) |
| GPU req | ~4-6 GB VRAM. Fine. |
| ROS2 | **Limited** — no official package. Requires custom wrapper around OpenPCDet or mmdet3d. |
| Drawback | Higher compute than PointPillars. Slower. More VRAM. |
| Key strength | Best accuracy/speed tradeoff for indoor among voxel methods. Robust to point sparsity. |
| **Recommendation** | **Best accuracy for indoor.** Choose if 10-20 FPS is sufficient and accuracy is priority. |

**Indoor evidence**: Basile et al. 2024 (Fraunhofer) tested SECOND on solid-state LiDAR person detection: **95% AP, 10.3 FPS on Jetson AGX**. O-LiPeDeT (Aalto 2026): SECOND was "most reliable backbone" for overhead indoor person detection.

### 3.3 VoxelNeXt

| Aspect | Assessment |
|--------|-----------|
| Indoor suitability | **Very good** — end-to-end, no NMS. Best for <3m range (O-LiPeDeT 2026): "VoxelNeXt is best for <3m range as observed by its respective recall, F1 and AP values." |
| Livox compatible | **Yes** — same voxelization as SECOND. |
| Jetson Orin | **Yes** — 9.8 FPS on Jetson AGX (Basile et al. 2024). On Orin: expect **~12-18 FPS** with TensorRT. |
| RTX 4060 | **Good** — ~15-25 FPS (spconv is compute-heavy; ONNX export may have issues with SpConv) |
| GPU req | ~5-7 GB VRAM (SpConv kernels). May be tight on 8GB with other processes. |
| ROS2 | **No** — SpConv makes ONNX/TensorRT export difficult. Requires custom PyTorch inference on Jetson. |
| Drawback | SpConv → difficult ONNX export → hard to TensorRT-optimize. Jetson deployment needs PyTorch + cuDNN (slower than TensorRT). |
| **Recommendation** | **Best indoor accuracy (especially close range).** But hardest to deploy on Jetson due to SpConv. Good for dev on RTX 4060. |

### 3.4 CenterPoint

| Aspect | Assessment |
|--------|-----------|
| Indoor suitability | **Moderate** — designed for large objects. Pedestrian detection is secondary. Domain gap paper shows: 9.9% AP → 58.2% after JRDB fine-tuning → 70% with pretrain+fine-tune. |
| Livox compatible | **Yes** — center-point detection, coordinate-based. |
| Jetson Orin | **Yes** — 4.41 FPS (Xavier AGX, PyTorch) → 18.4 FPS with TensorRT (Xavier AGX). On Orin: expect **~25-40 FPS** with TensorRT. |
| RTX 4060 | **Good** — ~30-50 FPS (TensorRT) |
| GPU req | ~4-6 GB. Fine. |
| ROS2 | **Limited** — no official. CenterPointTensorRT exists (GitHub). D-Robotics has CenterPoint for their platform. |
| Drawback | Two-stage (center detection + refinement) → more compute. Pedestrian is not the primary target. |
| **Recommendation** | **Viable but not optimal for humans.** Best if you want to detect multiple object classes. PointPillar backbone is lighter than Voxel. |

### 3.5 IRBGHR-PIXOR (Indoor-specific!)

| Aspect | Assessment |
|--------|-----------|
| Indoor suitability | **Purpose-built for indoor** — 97.17% AP@0.5 on JRDB indoor dataset |
| Architecture | PIXOR (BEV-based CNN) + Inverted Residual Blocks + Gaussian Heatmap Regression + Modified Focal Loss |
| Livox compatible | **Likely yes** — BEV-based, coordinate-agnostic |
| Speed | **Fast** — ~30+ FPS (BEV-CNN architecture, much lighter than voxel) |
| GPU req | Lightweight (~1-2 GB). MobileNet-like backbone. |
| ROS2 | **No** — research code only |
| Drawback | Single-class (pedestrian only). Not multi-object. |
| **Recommendation** | **Best AP for indoor pedestrian detection.** Lightweight. Ideal for Jetson. But: no ROS2, single-class, research code (need to reimplement). |

### 3.6 DBSCAN + Bounding Box (Classical)

| Aspect | Assessment |
|--------|-----------|
| Indoor suitability | **Moderate** — works for rough object segmentation. No classification. |
| Approach | 1) Remove ground (RANSAC) → 2) Voxel downsample → 3) DBSCAN cluster → 4) AABB per cluster → 5) Filter by height (1.5-2m = person) |
| Livox compatible | **Yes** — operates on raw points |
| Speed | **Very fast** — <1ms per frame. No GPU needed. |
| GPU req | **None** — CPU only |
| ROS2 | **Easy** — few lines with PCL/Open3D |
| Drawback | No classification (can't tell person from chair). Merged/split clusters. No heading. Not scale-invariant. |
| **Recommendation** | **Good as a baseline + fallback.** Use for initial validation, quick deployment, or as a safety net when GPU is busy. NOT a production solution alone. |

### 3.7 MinkowskiEngine (MinkowskiConvolution)

| Assessment |
|------------|
| General sparse convolution library from NVIDIA. Can implement SECOND-like or VoxelNet-like backbones. |
| NOT a detector itself — a library you build a detector with. |
| Heavier than SpConV (more flexible, slower). |
| NOT recommended for this use case — SpConv or standard 3D conv is simpler. |
| Jetson: poor CUDA/aarch64 support for MinkowskiEngine. |
| **Recommendation: Skip.** Use SpConV (included in OpenPCDet) or plain 3D conv. |

### 3.8 Other notable approaches

- **PillarNeXt (CVPR 2023)**: Improved PointPillars. Better for small objects. ~same speed. Good alternative to standard PointPillars.
- **PV-RCNN**: Voxel + Point hybrid. Two-stage. Higher accuracy but 2.5-5 FPS on Jetson (too slow).
- **DSVT** (sparse transformer): 71.93 AP on non-repetitive LiDAR (roadside). Heavy. Not for edge.
- **VoteNet / TR3D** (from RGB-D, not LiDAR): Used for indoor object detection in 2026 thesis (TR3D+VoteNet on Jetson AGX Orin: 4-8.7 FPS). Only for RGB-D, not LiDAR.
- **O-LiPeDeT** (Aalto 2026): Open source. Overhead LiDAR indoor. Fine-tunes PointPillars, SECOND, VoxelNeXt on custom dataset. **Most relevant prior work.**

---

## 4. ROS2 Ecosystem for 3D LiDAR Detection

| Package | Source | Framework | Notes |
|---------|--------|-----------|-------|
| `NVIDIA-AI-IOT/ros2_tao_pointpillars` | GitHub | TensorRT | **Best option.** Input: PointCloud2. Output: Detection3DArray. 10 FPS on Orin. |
| `DKrishna007/pointpillars-3d-detection-jetson` | GitHub | TensorRT FP16 | 42 FPS on AGX Orin. ByteTrack 3D MOT. |
| `ragibarnab/ros2-lidar-object-detection` | GitHub | PyTorch | PointPillars wrapper. Easier dev. |
| `ros-perception/perception_open3d` | GitHub | Open3D | NOT detection. Conversion utilities. |
| `mmdetection3d` + custom ROS2 node | — | PyTorch | Full toolkit. Complex setup. |
| `OpenPCDet` + custom ROS2 node | GitHub | PyTorch/SpConv | Research-grade. No official ROS2. |
| `laser_odometry_velodyne` / PCL | — | C++ | Clustering-based. ROS-native. |
| `livox_ros_driver2` | GitHub | — | **REQUIRED** for Mid-360 on ROS2. |

**Key gap**: No turnkey ROS2 package for SECOND or VoxelNeXt. Would need to write a node wrapping OpenPCDet inference.

**ROS2 + Livox stack**: `livox_ros_driver2` (pub `/livox/lidar` as PointCloud2) → detection node → detection output.

---

## 5. Comparison Table

| Model | Indoor AP (est.) | Jetson Orin FPS (TensorRT) | RTX 4060 FPS | GPU VRAM | ROS2 | Mid-360 OK | Recommendation |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **PointPillars** | 85-95% (fine-tuned) | **40-55** | 60-100 | 2-4 GB | ✅ Yes | ✅ | **Primary deploy choice** |
| **SECOND** | 90-97% (indoor) | **~15-20** | 20-30 | 4-6 GB | ⚠️ No | ✅ | **Best accuracy/speed balance** |
| **VoxelNeXt** | 90-97% (indoor) | **~12-18** | 15-25 | 5-7 GB | ❌ | ✅ | Best close-range, hardest deploy |
| **CenterPoint-Pillar** | 70-85% (fine-tuned) | **~25-40** | 30-50 | 4-6 GB | ⚠️ Partial | ✅ | Multi-class, two-stage |
| **IRBGHR-PIXOR** | **97%** (JRDB) | **~30-50** (est.) | 50-80 (est.) | 1-2 GB | ❌ | ✅ (likely) | **Best indoor pedestrian AP** |
| **DBSCAN + AABB** | N/A (no class) | **>>100** (CPU) | **>>100** | 0 GB | ✅ Yes | ✅ | Baseline / fallback |
| PILLa rNeXt | ~90% (est.) | ~35-50 | 50-80 | 3-4 GB | ⚠️ Partial | ✅ | PointPillars upgrade |
| PV-RCNN | ~92% | 2-5 | 10-15 | 6-8 GB | ❌ | ✅ | Too slow for Jetson |

All FPS are estimates unless marked with measured data. Jetson numbers derived from published benchmarks.

---

## 6. Recommended Path

### Phase 0: Baseline (Week 1)
1. **DBSCAN + AABB + height filter** to validate data pipeline works
2. `livox_ros_driver2` → PointCloud2 → Open3D/PCL → DBSCAN → publish boxes
3. Measure: point counts on objects at 1m, 5m, 10m, 15m
4. This gives immediate "something works" on any hardware

### Phase 1: PointPillars (Week 2-3)
1. Collect indoor data on G1 robot (human positions + common objects), 5-10 hours
2. Annotate 3D boxes (use O-LiPeDeT approach or manual in OpenPCDet viewer)
3. Train PointPillars on custom dataset (OpenPCDet or mmdet3d)
4. Export ONNX → TensorRT
5. Deploy on RTX 4060 first, then Jetson Orin
6. **Target**: >10 FPS on Orin, AP >85% for person

### Phase 2: SECOND or VoxelNeXt (Week 4-5)
1. If PointPillars accuracy insufficient → train SECOND on same data
2. Compare AP @ 5m, 10m, 15m
3. SECOND likely 5-10% AP higher. Cost: 2-3x compute.
4. If accuracy still not enough → try IRBGHR-PIXOR architecture (BEV-CNN, lightweight)

### Phase 3: Tracking (Week 6)
1. AB3D-MOT or SimpleTrack (see O-LiPeDeT repo)
2. Or ByteTrack 3D (see DKrishna repo)
3. Track identity → downstream navigation

---

## 7. Key References

1. **arXiv 2106.11239** — "Bridging the Domain Gap for Pedestrian Detection" (CenterPoint nuScenes→JRDB transfer)
2. **Basile et al. 2024** (Fraunhofer) — "Evaluation of 3D-LiDAR person detection for edge computing" (SECOND 10.3 FPS, 95% AP on Jetson AGX)
3. **O-LiPeDeT** (Aalto 2026) — Overhead LiDAR indoor. PointPillars/SECOND/VoxelNeXt. **Open source.**
4. **IRBGHR-PIXOR** (IEEE 2024) — 97.17% AP on JRDB. Indoor-specific.
5. **arXiv 2511.00060** (2025) — "Which LiDAR scanning pattern is better" — Non-repetitive = comparable to 128-line
6. **DKrishna007/pointpillars-3d-detection-jetson** — 42 FPS TensorRT on AGX Orin, indoor
7. **NVIDIA TAO PointPillars ROS2** — Official ROS2 node
8. **Basile et al. 2024 (KIT)** — Jetson AGX benchmarks: SECOND 10.3 FPS, VoxelNeXt 9.8 FPS
9. **Jetson 3D detection benchmark (MDPI 2023)** — All Jetson platforms tested. TensorRT 4x speedup.
10. **JRDB Leaderboard** (jrdb.erc.monash.edu) — Indoor detection benchmark. RPEA 39 FPS, PiFeNet 26 FPS
11. **Frontiers 2023** (warehouse) — Jetson Xavier AGX + L515, human 3D localization
12. **MMDetection3D Issue #3105** — Livox MID-360 + detection (community experience)
13. **HULI-Track (2026)** — ROS2 LiDAR dataset for human tracking in industrial envs
14. **Enhancing Indoor Mobility (2024)** — Indoor pedestrian tracking, clustering + LiDAR-camera fusion

---

## 8. Open Questions

1. **How many indoor data points needed?** JRDB has 54 sequences. O-LiPeDeT used custom dataset (unknown size). Expect: 500-2000 annotated sequences minimum for good generalization.
2. **Can we use synthetic data?** CARLA has indoor environments. NVIDIA DRIVE Sim has indoor. 2026 paper shows: 44% AP improvement from mixing real + synthetic (CARLA) data for LiDAR persons.
3. **Mid-360 vertical FOV**: -7° to +52°. If mounted at 1m height, covers 0-2m floor to 3m+ ceiling for nearby objects. Good for standing people. Mounting height/tilt critical.
4. **Multi-object classes**: If need to detect chairs/tables/shelves, need custom training data. Pretrained outdoor models won't help.
5. **SpConv on Jetson**: Does SpConv (used by SECOND/VoxelNeXt) support Jetson aarch64? (It does — PyPI builds exist, but SpConv kernels are compiled for specific CUDA arch. May need local build.)
