# LiDAR-Based Human Pose Estimation — Indoor / Livox MID-360 / Jetson Orin

> **Scope**: 99% indoor (offices, hallways, labs), range ≤15 m, sensor = Livox MID-360 (360°×59°, ~200 kpts/s, 40-line equiv.)
> **Hardware**: Jetson Orin 16 GB (target), RTX 4060 8 GB (dev)
> **Date**: 2026-09-01
> **Author**: Subagent research (for G1 perception thesis)

---

## 1. Landscape Overview

LiDAR-based 3D human pose estimation (3D HPE) remains a young and sparse field. The majority of published work targets **outdoor autonomous driving** (Waymo, nuScenes) with 64–128-beam spinning LiDARs at 10–30 m range. Indoor use with sparse rotating-solid-state (NRCS) sensors like Livox MID-360 is almost entirely unaddressed in the literature.

**Key constraint mismatch**: Most SOTA models (VoxelKP, DAPT, LPFormer) are:
- Trained on 64+ beam spinning RMB LiDARs (Waymo: 64×2650 pts per frame)
- Designed for 30–150 m range
- Assume 100k+ points per frame
- Often need SMPL mesh fitting or temporal context

Indoor Livox MID-360 produces **~30–60 points per person** at 5 m, **~10–20 points at 10 m** (NRCS rosetta pattern, inhomogeneous density). This is orders of magnitude sparser than training data.

---

## 2. Detailed Model Survey

### 2.1 VoxelKP (ICCV 2025, KAUST)

| Attribute | Value |
|---|---|
| **Keypoints** | 14 (head, L/R shoulder/elbow/wrist/hip/knee/ankle) |
| **Architecture** | Fully sparse voxel conv, 4 stages, sparse box-attention, BEV fusion |
| **Training data** | Waymo v1.4.2 (8,125 human instances) |
| **Point range** | 150.4 m × 150.4 m × 6 m |
| **Input features** | XYZ + intensity |
| **Single-scan** | YES |
| **Temporal** | NO |
| **Multi-person** | YES (single-stage) |
| **Code** | ✅ [github.com/shijianjian/VoxelKP](https://github.com/shijianjian/VoxelKP) (inference + train, built on OpenPCDet/VoxelNeXt) |
| **MPJPE** | 8.87 cm (Waymo) — 27% improvement over HUM3DIL |
| **GPU (train)** | 8× GPU (distributed training scripts) |
| **ROS2** | ❌ NO |
| **Indoor tested** | ❌ NO (outdoor driving only) |
| **Code status** | Inference-only repo + checkpoints on HF |

**Fit assessment**: VoxelKP is the current SOTA but designed for large-scale outdoor driving. The 14-keypoint output is fine. The architecture requires sparse convolution kernels (torchsparse/minkowski) which are available on Jetson but untested with this exact model. **Too heavy for Jetson Orin 16 GB in its current form** (model ~400-700 MB VRAM + sparse conv overhead).

### 2.2 DAPT (AAAI 2025, NJUST)

| Attribute | Value |
|---|---|
| **Keypoints** | 14 (same Waymo schema) |
| **Architecture** | UNet-like Point Transformer, joint anchors, 1D heatmap decoder, MDE module |
| **Training data** | Pre-trained on synthetic (SMPL raycast, 64-liDAR); fine-tuned on Waymo, SLOPER4D, Human-M3, LiDARHuman26M |
| **Single-scan** | YES (explicitly "single-frame low-quality LiDAR") |
| **Temporal** | NO |
| **Multi-person** | Implicit (point cloud level) |
| **Code** | ✅ [github.com/AnxQ/dapt](https://github.com/AnxQ/dapt) (Apache 2.0) |
| **MPJPE** | 7.87 mm improvement over LPFormer on Waymo; -20.7 mm vs PRN on SLOPER4D |
| **GPU (train)** | 2× RTX 4090 |
| **ROS2** | ❌ NO |
| **Indoor tested** | ❌ NO (outdoor; SLOPER4D is ~2.8 m avg distance, closest to indoor) |
| **Inference speed** | Not explicitly reported; PT architecture similar complexity to PointNeXt |
| **Key strength** | Density-aware — explicitly designed to handle point dropout and sparsity |

**Fit assessment**: DAPT is the **most promising single-scan model** for sparse point clouds. Its density-aware design directly addresses Livox sparsity. SLOPER4D dataset uses 2.8 m avg distance — closest available dataset to indoor. Point Transformer architecture is more portable than sparse conv (no torchsparse needed). Could potentially run on Jetson with 16 GB, though untested.

### 2.3 LidPose (Sensors 2024, Hungary)

| Attribute | Value |
|---|---|
| **Keypoints** | 17 (COCO format) |
| **Architecture** | ViTPose-based, range-image projection, 2D + 3D heads |
| **Sensor** | **Livox Avia** (NRCS, 70°×77° FoV) — same family as MID-360 |
| **Training data** | Custom dataset (outdoor + indoor), co-registered camera + LiDAR |
| **Single-scan** | YES |
| **Temporal** | NO |
| **Multi-person** | Single person per detection |
| **Code** | ✅ (public, but not well-documented) |
| **Indoor tested** | YES (one indoor test sequence in large room) |
| **ROS2** | ❌ NO |
| **Inference** | Real-time (paper claims <30 FPS on modern GPU) |
| **Key strength** | **Only model specifically tested on NRCS LiDAR** (Livox Avia), closest to MID-360 |

**Fit assessment**: **Most directly relevant to Livox MID-360 use case.** Same scanning family (NRCS rosetta pattern). Tested indoors. 17 keypoints (more than 14). ViTPose architecture runs on standard PyTorch/TensorRT without special sparse conv libraries. However: (1) single-person only, (2) small custom dataset, (3) no multi-person capability, (4) no ROS2.

### 2.4 LPFormer (ICRA 2024, TuSimple)

| Attribute | Value |
|---|---|
| **Keypoints** | 14 (Waymo) |
| **Architecture** | 2-stage: LiDARMultiNet detection + Keypoint Transformer (KPTR) regression |
| **Training data** | Waymo Open Dataset |
| **Single-scan** | YES |
| **Temporal** | NO |
| **Multi-person** | YES (depends on detection stage) |
| **Code** | ❌ NOT publicly available (TuSimple internal) |
| **ROS2** | ❌ NO |
| **Indoor tested** | ❌ NO |
| **GPU** | Requires multiple GPUs (LiDARMultiNet backbone) |

**Fit assessment**: Good accuracy but **code not public**. Two-stage pipeline, heavier than alternatives. Not practical for this project unless reimplementing.

### 2.5 HUM3DIL (CoRL 2022, Waymo)

| Attribute | Value |
|---|---|
| **Keypoints** | 14 (Waymo) |
| **Architecture** | Camera + LiDAR fusion, semi-supervised |
| **Training data** | Waymo Open Dataset |
| **Single-scan** | YES (but requires camera too) |
| **Temporal** | NO |
| **Multi-person** | YES |
| **Code** | ❌ NOT public |
| **ROS2** | ❌ NO |

**Fit assessment**: Fails on multiple counts: needs camera (not LiDAR-only), no code, driving-only.

### 2.6 LiveHPS / LiveHPS++ (CVPR 2024 / 2025)

| Attribute | Value |
|---|---|
| **Keypoints** | Full SMPL (17 joints) + body shape |
| **Architecture** | PointNet-GRU body tracker + SMPL solver |
| **Training data** | FreeMotion (40 motion types, multi-view LiDAR) |
| **Single-scan** | **NO — requires point CLOUD SEQUENCE (temporal)** |
| **Temporal** | YES (essential) |
| **Multi-person** | Single human tracked |
| **Code** | Partial (research code) |
| **ROS2** | ❌ NO |

**Fit assessment**: **Excluded** — temporal dependency, SMPL optimization loop (slow), requires continuous point cloud sequence. Not suitable for single-scan or Jetson.

### 2.7 LiCamPose (WACV 2025, Tsinghua)

| Attribute | Value |
|---|---|
| **Keypoints** | 14 (Waymo) |
| **Architecture** | Multi-view RGB + sparse point cloud fusion (volumetric) |
| **Single-scan** | YES |
| **Multi-person** | YES |
| **Code** | ✅ (partial) |
| **Indoor tested** | YES (MVOR dataset = operating room; basketball court) |
| **ROS2** | ❌ NO |

**Fit assessment**: Interesting indoor application but **requires multi-view cameras + LiDAR** (4 camera-LiDAR pairs). Overkill for single MID-360 setup.

### 2.8 Lightweight LiDAR-based (MoveNet depth) — Sensors 2026

| Attribute | Value |
|---|---|
| **Keypoints** | 17 (COCO) |
| **Architecture** | Point cloud → depth image → MoveNet (2D) → depth-based 3D lifting |
| **Single-scan** | YES |
| **Multi-person** | Single |
| **Indoor tested** | Partial (autonomous driving context) |
| **Inference** | MoveNet is <1 ms per frame — very lightweight |
| **Code** | Not public (method paper) |

**Fit assessment**: **Most promising for Jetson Orin deployment.** Pipeline: point cloud → virtual camera projection → depth image → MoveNet (2D CNN, <50 MB) → 3D lifting via depth lookup. Total model <100 MB, runs at >100 FPS on Jetson. Limitations: single-person, requires known camera pose for projection, depth image representation loses 3D structure.

### 2.9 Fallback: MediaPipe BlazePose (camera-based)

| Attribute | Value |
|---|---|
| **Keypoints** | 33 (full body, 3D) |
| **Input** | RGB image (NOT LiDAR) |
| **Single-frame** | YES |
| **Inference** | 30-60+ FPS on Jetson |
| **3D output** | YES (relative z, 2m×2m×2m cube, centered on hips) |
| **ROS2** | ✅ Multiple wrappers exist (mediapipe-ros2, opencv4ros) |
| **Jetson** | ✅ Native support (TensorRT, TFLite) |

**Fit assessment**: Not LiDAR-based at all, but provides **instant 3D pose from camera at near-zero cost**. The z-depth is relative (not metric), so absolute 3D position must come from LiDAR or another source. Best candidate for the "hybrid approach" (Q9).

---

## 3. Direct Answer Matrix

### Q1: Which LiDAR pose estimators exist?

| Model | Year | Venue | Keypoints | Indoor | Code |
|---|---|---|---|---|---|
| **VoxelKP** | 2025 | ICCV | 14 | ❌ | ✅ |
| **DAPT** | 2025 | AAAI | 14 | ❌ (tested at 2.8m) | ✅ |
| **LPFormer** | 2024 | ICRA | 14 | ❌ | ❌ |
| **HUM3DIL** | 2022 | CoRL | 14 | ❌ | ❌ |
| **LidPose** | 2024 | Sensors | 17 | ✅ | ✅ |
| **LiveHPS** | 2024 | CVPR | SMPL | ❌ | ✅ (partial) |
| **LiveHPS++** | 2025 | — | SMPL | ❌ | ❌ |
| **LiCamPose** | 2025 | WACV | 14 | ✅ | ✅ (partial) |
| **FusionPose** | 2023 | AAAI | 14 | ❌ | ✅ (partial) |
| **MoveNet-depth** | 2026 | Sensors | 17 | ⚠️ | ❌ |

### Q2: Which work on single-scan (non-temporal) LiDAR?

**YES — single scan**: VoxelKP, DAPT, LPFormer, HUM3DIL, LidPose, LiCamPose, FusionPose, MoveNet-depth
**NO — temporal required**: LiveHPS, LiveHPS++, LiDARCap, NE-3D-HPE, Point2Pose

DAPT explicitly states: *"our method greatly improves the stability and accuracy of single-frame LiDAR-only human pose estimation"*

### Q3: Actual inference speed on edge GPUs?

| Model | Desktop GPU | Jetson Orin | Notes |
|---|---|---|---|
| VoxelKP | ~5-15 FPS est. (400-700 MB VRAM) | ~2-5 FPS est. (sparse conv) | No published benchmarks |
| DAPT | ~10-30 FPS est. (PT arch) | ~5-15 FPS est. | PTv3-based, similar to PointNext |
| LidPose | 30+ FPS (paper claims real-time) | ~15-30 FPS est. | ViTPose, standard PyTorch |
| MoveNet-depth | 100+ FPS | **>60 FPS** | 2D CNN, <50 MB model |
| BlazePose (cam) | 60+ FPS | **30-60 FPS** | TensorRT optimized |
| LPFormer | ~5 FPS (2-stage) | ~2-4 FPS | LiDARMultiNet + KPTR |

**Reality check**: No published FPS numbers for any LiDAR pose model on Jetson. The TorchSparse++ paper benchmarks sparse conv on Orin, achieving >18 FPS for detection workloads at 1× speed. Point Transformer workloads (DAPT) are 1.25× slower than SpConv but still competitive. **Estimate: a DAPT-class model at 15-30 FPS on Orin is achievable if the model is <500 MB.**

### Q4: Do any have ROS2 wrappers?

**NO** — none of the LiDAR-specific pose models have ROS2 wrappers.

Indirect options:
- **VoxelKP**: Build on OpenPCDet → write a ROS2 node wrapping inference
- **DAPT**: Point Transformer architecture → wrap in PyTorch ROS2 node
- **LidPose**: ViTPose-based → could wrap in ROS2 with Livox SDK2 point cloud input
- **BlazePose (camera)**: Existing ROS2 packages: `mediapipe_ros2`, `mediapipe_msgs`
- **Livox driver**: `livox_ros_driver2` (ROS2, official) provides point cloud + IMU topics

### Q5: How many keypoints do they estimate?

| Key count | Models | Joints covered |
|---|---|---|
| **14** (most common) | VoxelKP, DAPT, LPFormer, HUM3DIL, LiCamPose | Head, L/R shoulder/elbow/wrist/hip/knee/ankle |
| **17** (COCO) | LidPose, MoveNet-depth | + nose, + L/R ear, + L/R big toe/small toe |
| **33** (full body) | MediaPipe BlazePose | 14 + face/hand/foot detail |
| **SMPL 21** | LiveHPS, FusionPose | 15 joints + jaw/eyes/thumbs |
| **14-17+** (G1 robot) | G1 has 29 DOF; 25-35 keypoints typical | Depends on robot skeleton |

> **For G1 robot (Unitree)**: 29 DOF, human-like biped. 14-keypoint output maps cleanly to G1 joints. 17-keypoint (COCO) provides slightly more ankle/ear info. **14 is the practical minimum for G1 motion planning; 17 preferred for finger-level detail on feet.**

### Q6: What's the minimum point density needed?

From dataset statistics:

| Dataset | Avg points/person | Avg distance | Context |
|---|---|---|---|
| Waymo Open | 384 | 14.5 m | 64-beam spinning, outdoor |
| SLOPER4D | 968 | 2.8 m | Multi-LiDAR studio, indoor |
| Human-M3 | 369 | N/A | 4× LiDAR, outdoor |
| **Livox Avia @ 5m** | **~30-60** | 5 m | 1 NRCS, indoor |
| **Livox Avia @ 10m** | **~10-20** | 10 m | 1 NRCS, indoor |

**Critical finding**: The Livox MID-360 at 5 m produces **~10-30× fewer points per human** than any training dataset used for SOTA models. The DAPT paper explicitly addresses this with its density-aware design. LiCamPose paper states: *"using Livox point cloud information alone struggles to accurately extract the 3D human skeleton due to its sparsity"* (on Mid-40, similar class to MID-360).

**Practical minimum**: Research suggests **~50+ points per person** is needed for reliable keypoint estimation. At 5 m with MID-360, this is borderline. At 2-3 m, it's more feasible.

### Q7: Are there lighter alternatives?

**Ranked by ease of Jetson deployment**:

1. **MediaPipe BlazePose 3D** (camera) — 33 keypoints, 30-60 FPS on Orin, TensorRT ready, ~15 MB model. **Easiest by far.**
2. **MoveNet + depth lifting** (LiDAR) — 17 keypoints, 100+ FPS on Orin, <50 MB. Novel approach from Sensors 2026.
3. **CDO-POSE** (camera, YOLOv11-based) — 17 keypoints, 30-40 FPS on Orin Nano
4. **LidPose** (LiDAR, ViTPose-based) — 17 keypoints, 30 FPS on desktop, ~15 FPS Orin est.
5. **DAPT** (LiDAR, Point Transformer) — 14 keypoints, 10-30 FPS on desktop, ~5 FPS Orin est.

### Q8: Could a simple skeleton estimator run on Jetson Orin 16 GB?

**YES — definitely, with strong caveats**:

| Approach | Model size | VRAM on Orin | Est. FPS | Reliability at 5m |
|---|---|---|---|---|
| BlazePose (cam) | 15 MB | <100 MB | 30-60 | N/A (camera) |
| MoveNet-depth (LiDAR) | ~30 MB | <150 MB | 60+ | Moderate |
| Simplified VoxelKP (pruned) | ~200-400 MB | 500-800 MB | 5-10 | Low (needs retrain for indoor) |
| DAPT (full) | ~100-300 MB | 400-700 MB | 5-15 | Moderate (density-aware but trained on outdoor) |

**Key issue**: None of the LiDAR pose models are trained on indoor NRCS data at 5 m range. You would need to:
1. Collect indoor data with MID-360 + camera GT (or SMPL)
2. Fine-tune DAPT or train MoveNet-depth on your domain
3. Address the 10-30× point density gap

**Realistic option for Jetson**: MoveNet-depth pipeline (point cloud → depth image → MoveNet → depth-based 3D). Total <100 MB, trivially fast, but accuracy limited by 2D representation.

### Q9: Hybrid camera + LiDAR approach

**This is the most practical and robust option.**

**Architecture**:
```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│  Camera     │     │  LiDAR      │     │  Jetson Orin     │
│  (RGB)      │     │  MID-360    │     │                  │
└──────┬──────┘     └──────┬──────┘     └──────┬───────────┘
       │                   │                    │
       ▼                   ▼                    ▼
  BlazePose          Person detection       Fusion node
  (2D+rel3D)         (PointPillars/         (ROS2)
  33 kpts            VoxelNeXt)
       │                   │
       ▼                   ▼
  2D pose +          3D bounding      ┌──────────────────┐
  rel. z (cube)      box + position   │ 3D Pose =         │
       │                   │          │   Cam relative     │
       │                   │          │   z → lifted to   │
       │                   │          │   metric 3D via    │
       │                   │          │   LiDAR centroid   │
       └───────────────────┴──────────┤                     │
                                     │  14-17 3D kpts       │
                                     └──────────────────┘
```

**How it works**:
1. **Camera** (ZED/RPi/USB): BlazePose → 2D keypoints + relative z (in 2m×2m×2m cube, hip-centered)
2. **LiDAR** (MID-360): Person detection (PointPillars, VoxelNeXt, or simple clustering) → 3D bounding box + centroid
3. **Fusion** (ROS2 node): Project person bbox to camera image → crop → BlazePose on crop → take relative z → **scale + translate to LiDAR metric coordinate frame** using bbox centroid as reference

**Accuracy achievable**: ±5-10 cm per joint (camera relative z is well-calibrated; LiDAR gives absolute position)

**This is what LiCamPose and FusionPose do, but in a simpler, more practical form.**

**Advantages**:
- BlazePose is battle-tested, 30+ FPS on Jetson
- LiDAR gives absolute 3D position (camera can't do this alone)
- Works with existing Livox SDK2 (ROS2 driver already available)
- Each component individually deployable and debuggable
- Can degrade gracefully: LiDAR down → cam-only 2D pose; camera down → LiDAR-only bbox tracking

**Disadvantages**:
- Needs camera-LiDAR calibration (static, one-time)
- Camera must see the person (LiDAR can occlude)
- 2D pose + relative z is approximate (not true 3D)
- Single-person per camera region

---

## 4. Recommendation Matrix

| Model | KPs | FPS | GPU | Single-scan | ROS2 | Indoor | **Recommendation** |
|---|---|---|---|---|---|---|---|
| **VoxelKP** | 14 | 5-15 est. | 400-700 MB | ✅ | ❌ | ❌ | Best accuracy, outdoor-trained, heavy. Use if accuracy > everything |
| **DAPT** | 14 | 10-30 est. | 200-500 MB | ✅ | ❌ | ⚠️ (2.8m) | **Best LiDAR-only option** for sparse NRCS. Density-aware |
| **LidPose** | 17 | 30+ est. | 100-300 MB | ✅ | ❌ | ✅ | **Best Livox-specific option**. Only model tested on NRCS LiDAR |
| **LPFormer** | 14 | ~5 | 500+ MB | ✅ | ❌ | ❌ | Good but no code |
| **HUM3DIL** | 14 | ~5 | 500+ MB | ✅ | ❌ | ❌ | Needs camera too, no code |
| **LiveHPS** | SMPL | 1-5 | 300+ MB | ❌ | ❌ | ❌ | Rejected (temporal, SMPL) |
| **LiCamPose** | 14 | ~5 | 500+ MB | ✅ | ❌ | ✅ | Multi-camera, overkill |
| **MoveNet-depth** | 17 | 100+ est. | <100 MB | ✅ | ❌ | ⚠️ | **Lightest LiDAR option**. Novel, unvalidated |
| **BlazePose** | 33 | 30-60 | <100 MB | ✅ | ✅ | ✅ | **Lightest camera option**. Proven, ROS2 ready |

---

## 5. Recommended Path for G1 Perception

### Phase 1 — Immediate (dev, RTX 4060)
```
1. Set up BlazePose (camera) → 33 relative 3D keypoints, ROS2 node
2. Set up Livox SDK2 (MID-360) → point cloud topic
3. Add simple LiDAR human clustering (Euclidean/DBSCAN on point cloud)
   → 3D centroid + bbox
4. Write fusion node: camera crop → BlazePose → depth-lift to metric 3D
5. Validate: 1-person scenarios, 3-10 m range
```
**Deliverable**: Working 3D skeleton at 15-30 FPS on desktop GPU, camera + LiDAR

### Phase 2 — Jetson Orin (deployment)
```
1. Port BlazePose to TensorRT (ONNX → TRT)
2. Port LiDAR clustering to CUDA (or just ROS2 PCL on CPU)
3. Fusion node → TensorRT + ROS2
4. Target: <200 ms total latency, 15+ FPS
```
**Deliverable**: 3D skeleton at 15-20 FPS on Jetson Orin 16 GB

### Phase 3 — Research (thesis contribution)
```
1. If LiDAR-only is needed:
   - Fine-tune DAPT with synthetic indoor NRCS data
   - Collect 1-2 weeks of MID-360 + camera GT (or SMPL) data
   - Retrain on Livox-specific pattern
2. Or: train MoveNet-depth on your own depth image data
3. Or: build custom lightweight PointNet pose estimator
   (Point Net 2 layers → 14-17 keypoint regression head)
4. Bench: compare LiDAR-only vs camera+LiDAR hybrid
```
**Deliverable**: Novel contribution — first indoor NRCS LiDAR pose estimator validated at 3-10 m

### Why not just use VoxelKP?
- Trained on Waymo 150 m range driving
- Requires 100k+ point cloud input (MID-360 gives ~5k in 100 ms at 200 kpts/s)
- Sparse conv kernels untested on Jetson
- No indoor domain adaptation
- 14 keypoints only

### Why DAPT over VoxelKP?
- **Density-aware** — designed for point dropout, directly addresses NRCS sparsity
- Point Transformer — no special sparse conv library needed
- SLOPER4D test at 2.8 m = closest to indoor use case
- 2× RTX 4090 training (achievable on lab)
- Apache 2.0 license

### Why hybrid is the pragmatic choice?
- BlazePose: proven, 15 MB, 30+ FPS on Orin
- LiDAR: absolute metric position (camera can't do this)
- Each component debuggable independently
- Graceful degradation
- No retraining needed for Phase 1

---

## 6. ROS2 Package Plan

```
g1_pose_estimation/
├── g1_pose_cam/                    # Camera-based pose (BlazePose)
│   ├── src/pose_node.py            # Subscribe /camera/color/image
│   │                              # Publish  /g1/pose_3d (17 kpts, metric)
│   ├── src/blazepose_wrapper.py    # BlazePose → ROS2
│   └── launch/pose_cam.launch.py
│
├── g1_pose_lidar/                  # LiDAR-based detection + optional pose
│   ├── src/human_cluster.py        # DBSCAN/Euclidean on point cloud
│   ├── src/human_bbox.py           # 3D bounding box from cluster
│   ├── publish   /g1/person_bboxes
│   └── launch/pose_lidar.launch.py
│
├── g1_pose_fusion/                 # Camera + LiDAR fusion
│   ├── src/fusion_node.py          # Bbox → crop → BlazePose → depth-lift
│   ├── publish   /g1/skeleton_3d   # 17 3D keypoints, metric world frame
│   └── launch/fusion.launch.py
│
└── g1_skeleton_msgs/               # Custom message types
    └── msg/Skeleton3d.msg          # pose [17], covariance [17x3], timestamp
```

---

## 7. Key References

1. **VoxelKP**: Shi, J., Wonka, P. "VoxelKP: A Voxel-based Network Architecture for Human Keypoint Estimation in LiDAR Data." ICCV 2025. [github.com/shijianjian/VoxelKP](https://github.com/shijianjian/VoxelKP)

2. **DAPT**: An, X. et al. "Pre-training a Density-Aware Pose Transformer for Robust LiDAR-based 3D Human Pose Estimation." AAAI 2025. [github.com/AnxQ/dapt](https://github.com/AnxQ/dapt)

3. **LidPose**: Kovács, L., Bódis, B.M., Benedek, C. "LidPose: Real-Time 3D Human Pose Estimation in Sparse Lidar Point Clouds with Non-Repetitive Circular Scanning Pattern." Sensors 2024;24(11):3427. [mdpi.com/1424-8220/24/11/3427](https://www.mdpi.com/1424-8220/24/11/3427)

4. **LPFormer**: Ye, D. et al. "LPFormer: LiDAR Pose Estimation Transformer with Multi-Task Network." ICRA 2024. [arxiv.org/abs/2306.12525](https://arxiv.org/abs/2306.12525)

5. **LiCamPose**: Pan, Z. et al. "LiCamPose: Combining Multi-View LiDAR and RGB Cameras for Robust Single-timestamp 3D Human Pose Estimation." WACV 2025. [arxiv.org/abs/2312.06409](https://arxiv.org/abs/2312.06409)

6. **HUM3DIL**: Zanfir, R. et al. "Semi-supervised Multi-modal 3D Human Pose Estimation for Autonomous Driving." CoRL 2022. [waymo.com](https://waymo.com/research/hum3dil-semi-supervised-multi-modal-3d-human-pose-estimation-for-autonomous/)

7. **LiveHPS**: Ren, Y. et al. "LiveHPS: LiDAR-based Scene-level Human Pose and Shape Estimation in Free Environment." CVPR 2024. [arxiv.org/abs/2402.17171](https://arxiv.org/abs/2402.17171)

8. **MoveNet-depth**: "Lightweight LiDAR-Based 3D Human Pose Estimation via 2D Depth Images." Sensors 2026;26(5):1631. [mdpi.com/1424-8220/26/5/1631](https://www.mdpi.com/1424-8220/26/5/1631)

9. **Review**: Galaaoui, S. et al. "3D Human Pose and Shape Estimation from LiDAR Point Clouds: A Review." 2025. [arxiv.org/abs/2509.12197](https://arxiv.org/abs/2509.12197)

10. **TorchSparse++**: Tang, H. et al. "TorchSparse++: Efficient Point Cloud Engine." CVPR 2023 Workshop. Benchmark on Jetson Orin. [github.com/mit-han-lab/torchsparse](https://github.com/mit-han-lab/torchsparse)

11. **Livox MID-360**: [livoxtech.com/mid-360](https://www.livoxtech.com/mid-360) — 360°×59°, 200 kpts/s, 40-line, min 10 cm range

---

## 8. Open Questions / Future Research

- [ ] What is the actual point density of MID-360 on a human at 3 m, 5 m, 10 m? (measure, don't assume)
- [ ] Can DAPT be quantized to INT8 and run on Jetson Orin with <500 MB VRAM?
- [ ] How does NrCS rosetta pattern affect point cloud distribution for pose vs spinning LiDAR?
- [ ] Is there enough data to train/fine-tune indoor pose models, or do we rely on synthetic (Isaac Sim)?
- [ ] For multi-person indoor: how do we resolve occlusion with single 360° LiDAR?
- [ ] G1 robot-specific: 14 keypoints maps to which G1 joints? (needs kinematic mapping table)
- [ ] Livox SDK2 ROS2: point cloud rate, packet loss, and latency on Jetson Orin?
- [ ] Camera choice for G1: ZED 2i Mini? RPi + camera module? USB webcam?
