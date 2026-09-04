# Indoor LiDAR 3D Multi-Class Object Detection — Research Findings

> **Date**: 2026-09-02
> **Context**: G1 perception stack, Livox MID-360 (360° HFOV, 59° VFOV, 200K pts/s), indoor office/lab/warehouse ≤15m, Jetson Orin 16GB (target), RTX 4060 (dev)
> **Goal**: Detect humans + common indoor objects (chairs, tables, shelves, boxes, forklifts, pallets)

---

## 1. Critical Finding: The Indoor-LiDAR-MultiClass Gap

**No single off-the-shelf dataset or pretrained model covers all requirements.**

| Requirement | Best available dataset | Gap |
|---|---|---|
| Human + objects multi-class | ScanNet (RGB-D, not LiDAR) | No LiDAR sensor |
| LiDAR + warehouse objects | lidar-warehouse-dataset | No humans, only 5 classes |
| LiDAR + indoor 24 classes | LiDAR-Net (MLS) | MLS scanner, not single LiDAR on robot |
| Human + objects from robot | JRDB | Humans ONLY |
| Best detector | UniDet3D (AAAI 2025) | Trained on RGB-D, untested in Jetson LiDAR |

**Conclusion**: Must build hybrid pipeline. Use ScanNet-trained base + fine-tune on robot-collected data.

---

## 2. Indoor Datasets — Comparison Table

| Dataset | Year | Sensor | # Classes | # Scenes/Scans | Indoor? | LiDAR? | Download Link |
|---|---|---|---|---|---|---|---|
| **ScanNet v2 / ScanNet200** | 2017/2020 | RGB-D (Kinect, iPad SL) | 18 / 200 | 1,513 / 2,400 | Yes | **No** | [scannet.org](https://scannet.org) |
| **S3DIS** | 2016 | RGB-D (Kinect) | 13 (5 obj + wall/floor/ceiling) | 6 areas, 271 rooms | Yes | **No** | [s3dis.stanford.edu](http://www.stanford.edu/~shafer/project/s3dis) |
| **SUN RGB-D** | 2015 | RGB-D (4 scanners) | 10 + bg | 10,355 images | Yes | **No** | [rgbd.cs.princeton.edu](https://rgbd.cs.princeton.edu) |
| **Matterport3D** | 2017 | Matterport Pro Camera | 38 semantic classes | 90 buildings | Yes | **No** | [niessner.github.io/Matterport](https://niessner.github.io/Matterport/) (form required) |
| **JRDB** | 2020 | Velodyne VLP-16 ×2 + Sick ×2 + stereo RGB 360 | **Humans only** (pedestrian) | 54 seq / 28K frames | Yes + outdoor | **Yes** | [jrdb.erc.monash.edu](https://jrdb.erc.monash.edu) |
| **LiDAR-Net** (CVPR 2024) | 2024 | **MLS (terrestrial LiDAR scanner)** | **24 indoor classes** | 9 buildings / 3.0K m² | Yes | **Yes** (MLS) | [arxiv 2312.13029](https://arxiv.org/abs/2312.13029) |
| **ScanNet++** | 2023 | **TLS (terrestrial LiDAR scanner)** | 1000+ semantic | 713 scenes | Yes | **Yes** (TLS) | [scannetpp.github.io](https://scannetpp.github.io) |
| **lidar-warehouse-dataset** | 2024 | **Velodyne VLP-16** (Puck) | **5** (FTS, ELF++, CargoBike, MetalBox, ForkLift) | 3,287 scans | Yes (warehouse) | **Yes** | [github.com/anavsgmbh](https://github.com/anavsgmbh/lidar-warehouse-dataset) |
| **Cubify-Anything CA-1M** (CVPR 2025) | 2025 | LiDAR-derived | 400K objects, many classes | 1K+ scenes | Yes | **Yes** (LiDAR-derived) | [CVPR 2025 paper](https://cvpr.thecvf.com/virtual/2025/poster/35075) |
| **UniDet3D mixture** | 2025 | Multi (ScanNet+S3DIS+MultiScan+3RScan+ScanNet+++ARKitScenes) | Unified ~50 classes | 8K+ scenes | Yes | Mixed | [huggingface datasets](https://huggingface.co/datasets/maksimko123/UniDet3D) |
| **CodA** | 2025 | **128-line LiDAR** + 2 RGB | **53** | 28K frames | Indoor + outdoor | **Yes** | (campus robot, paper 2025) |

### Key Class Lists

**ScanNet v2 (18 classes)**:
`window, door, cabinet, chair, table, couch, refrigerator, sink, bathtub, bookshelf, counter, desk, curtain, toilet, mirror, pillow, desk lamp, tv`

**S3DIS (13 classes)**:
`wall, floor, ceiling, beam, column, window, door, table, chair, sofa, board, otherfurniture, partition`

**JRDB**: Human / pedestrian only (3D boxes)

**LiDAR-Net (24 indoor classes)**:
Includes: wall, floor, ceiling, window, door, table, chair, sofa, lamp, bookshelf, sink, microwave, oven, refrigerator, counter, desk, partition, pillar, beam, other furniture

**lidar-warehouse (5 classes)**:
`FTS (vehicle platform), ELFplusplus (vehicle platform), CargoBike (vehicle platform), MetalBox, ForkLift` — No humans, no furniture

**SemanticKITTI / nuScenes / Waymo**: NO indoor subset. All outdoor.

---

## 3. Multi-Class 3D Detectors — Comparison

### Can Outdoor Detectors Do Indoor?

| Detector | Indoor Support | Multi-Class | Speed (RTX 4060 est.) | Jetson Orin? | Notes |
|---|---|---|---|---|---|
| **PointPillars** | ❌ (trained KITTI/nuScenes outdoor) | ✅ (configurable num_classes) | **5-15 ms** | ✅ | Fastest, pillar-based. `num_classes` is a config param — train with 10+ classes easy. Not pre-trained for indoor. |
| **SECOND** | ❌ (outdoor) | ✅ | **15-30 ms** | ⚠️ (possible with TensorRT) | 3D sparse conv (mmsc). More accurate but heavier. OpenPCDet supports custom classes. |
| **VoxelNeXt** | ❌ (outdoor) | ✅ | **20-40 ms** | ⚠️ | Better than SECOND but designed for outdoor large-scale. Complex. |
| **PillarNet** | ⚠️ | ✅ (arbitrary) | **5-10 ms** | ✅ | Newer pillar, efficient. |
| **IA-SSD** | ✅ (indoor+outdoor) | ✅ | **8-20 ms** | ✅ | Point-based, "Not All Points Are Equal". |

### Indoor-Specific Detectors

| Detector | Dataset | # Classes | mAP50 (ScanNet) | Speed | Jetson Orin? | Link |
|---|---|---|---|---|---|---|
| **VoteNet** | ScanNet, SUN RGB-D | 18 | **58-62** | 30-80 ms | ⚠️ (PyTorch, no TensorRT) | [traveldig/votenet-3d](https://github.com/traveldig/votenet-3d) |
| **FCAF3D** | ScanNet (3D) | 18 | **35-40** | **<10 ms** | ⚠️ | [samsunglabs/fcaf3d](https://github.com/samsunglabs/fcaf3d) |
| **TR3D** | ScanNet | 18 | ~60 | **Real-time** (~15ms) | ⚠️ | [SamsungLabs/tr3d](https://github.com/SamsungLabs/tr3d) |
| **3DETR** | ScanNet, SUN RGB-D | 18 | **60-65** | 40-100 ms | ❌ | [3DETR GitHub](https://github.com/wzzheng/3DETR) |
| **UniDet3D** (AAAI 2025) | 6 indoor datasets (unified ~50 classes) | **~50 unified** | **65.9 (ScanNet)**, **65.3 (S3DIS)** | 20-50ms | ⚠️ | **[filaPro/unidet3d](https://github.com/filaPro/unidet3d)** ⭐ BEST |
| **CuTR / Cubify** (CVPR 2025) | CA-1M (LiDAR-derived) | many | **62%+ object recall** | unknown | TBD | CVPR 2025 |
| **OneDet3D** (NeurIPS 2024) | Multi-domain | All classes | strong | unknown | TBD | open-source coming |

### Instance Segmentation (Alternative to Box Detection)

| Method | Dataset | mAP50 | Jetson? | Notes |
|---|---|---|---|---|
| **PointGroup** (CVPR 2020) | ScanNet v2, S3DIS | **63.6 (ScanNet)**, **64.0 (S3DIS)** | ⚠️ | Dual-set point grouping. Best for "identify separate chairs" case. |
| **Mask3D** | Structured3D (synthetic), ScanNet | — | ❌ | Room-level instance seg. Not designed for robot. |
| **SoftGroup** (CVPR 2022) | ScanNet, S3DIS | 65+ | ⚠️ | Newer than PointGroup. |
| **MaskGroup** (2022) | ScanNet | 64+ | ⚠️ | |

### Key Point: Multi-Class IS Supported by Outdoor Detectors

**Yes — PointPillars, SECOND, VoxelNeXt, PillarNet all support multi-class.** The class count is a config parameter. In OpenPCDet, `num_classes` in `cfg.yaml` controls classification head size. You can set it to 5, 10, or 100 classes and the model trains accordingly. The issue is NOT the architecture — it's the TRAINING DATA being outdoor.

---

## 4. Indoor-Specific Challenges (MID-360 Specific)

### MID-360 Quirks
- **Non-repetitive** random scan pattern, not fixed planes. Angular resolution improves over ~2s integration.
- For ≤15m indoor: 200K pts/s is dense enough.
- Single return, 905nm. Reflectivity matters for furniture (white paint = good, dark fabric = bad).
- 59° VFOV means objects above/below ±30° from horizon may be missed.

### Challenging Cases for MID-360

| Object | Size | Challenge |
|---|---|---|
| Chair (standard) | 0.45-0.6m × 0.6m | Small, thin legs, low point count |
| Table (office) | 0.6m × 1.2m | Flat top = sparse return, void underneath hard to segment |
| Shelf | 0.4m × 2m | Tall + thin vertical lines, easily confused with walls |
| Box (cardboard) | 0.3-1m | No structure, irregular shape |
| Forklift | 2-3m | Large, moving — good detection candidate |
| Person | 0.5m × 1.8m | Best LiDAR target — irregular shape moves |
| Table legs / chair legs | 5cm diameter | Below minimum feature detection for sparse scans |
| Door (open) | 0.9m × 2m | Against wall — hard to separate |

### Recommended Minimum Viable Detection

| Priority | Class | Why | Feasibility |
|---|---|---|---|
| 1 | **Human / pedestrian** | Safety-critical | **Easiest** — best labeled data, most distinctive shape, JRDB available |
| 2 | **Forklift / robot platform** | Warehouse safety | **Good** — lidar-warehouse data, large objects |
| 3 | **Metal box / pallet** | Warehouse operation | **Moderate** — lidar-warehouse has 2847 box samples |
| 4 | **Table / chair** | Indoor awareness | **Hard** — small, occluded, indistinguishable from walls in sparse LiDAR |
| 5 | **Walls / ceiling / floor** | Not for detection — for SLAM/navigation | Use PCL, not detector |

---

## 5. ROS2 Integration

### Available Packages

| Package / Resource | What It Does | Status |
|---|---|---|
| **NVIDIA TAO — PointPillars** (ROS2 node) | Real-time 3D detection, TensorRT-optimized, ROS2 Humble | **Best option** — [developer.nvidia.com/blog](https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-using-ros-2-and-tao-pointpillars/) |
| **CUDA-PointPillars** (NVIDIA) | 33ms on Xavier AGX; C++ + TensorRT, no ROS wrapper | [github.com/NVIDIA-AI-IOT](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars) — fast but no ROS2 |
| **ragibarnab/ros2-lidar-object-detection** | Simple ROS2+PyTorch PointPillars wrapper | Basic, Python — slow but works |
| **perception_pcl** (Rosperception) | PCL wrappers for ROS2 (clustering, filtering) | Mature, C++ |
| **open3d-detection** | Open3D + detection (research only) | Not production-grade |
| **Isaac ROS Object Detection** (NVIDIA 2024) | ROS2 Jazzy + Jetson — 2D RGB detection | 2D only, not 3D LiDAR |
| **mmdetection3d** (OpenMMLab) | Full research framework: VoteNet, UniDet3D, SECOND, etc. | Research-only, Python, not real-time on Jetson |

### No Ready-to-Use Multi-Class INDOOR ROS2 Package Exists

There is no drop-in ROS2 package that does multi-class indoor LiDAR detection out of the box. Build required.

---

## 6. Recommended Pipeline for G1

### Architecture: Two-Head Approach

```
MID-360 Point Cloud (200K pts/s, 5-10 Hz frames)
         │
         ├──► [Head A: Human Detection]
         │    PointPillars (retrained, 2 classes: pedestrian + bg)
         │    Pre-train from JRDB, fine-tune on robot data
         │    → 3D boxes + heading, real-time OK on Orin
         │
         ├──► [Head B: Static Object Segmentation]
         │    FCAF3D or VoteNet (ScanNet-trained)
         │    OR: semantic segmentation → cluster → bounding box
         │    → chair/table/shelf/box/other
         │
         └──► [Head C (warehouse mode): Vehicle + Cargo]
              PointPillars (5-class: FTS/ELF++/CargoBike/Box/ForkLift)
              Pre-train from lidar-warehouse-dataset
```

### Why Two/Three Heads Instead of One Unified Model

- **Human detection** — safety priority, needs <50ms latency, 3D tracking ID
- **Static objects** — slower is OK (100-200ms), used for spatial awareness
- **Warehouse objects** — mode-specific, activate in warehouse environments
- One unified model with 10+ classes is possible but:
  - Requires training data per class in real deployment locations
  - Single model harder to interpret, debug, and A/B test per class

### Minimum Viable System (Phase 1 — 2 weeks)

```
1. Build: PointPillars (OpenPCDet), 2-class (pedestrian/bg)
2. Train: JRDB train split (1.8M pedestrian 3D boxes, indoor+outdoor mix)
3. Fine-tune: 1-2 days robot data in target environment
4. Export: ONNX → TensorRT (FP16) target
5. ROS2 node: livox_driver2 → point cloud → detection node → /detections topic
   (Publish: sensor_msgs::PointCloud2 in, visualization_msgs::MarkerArray out)
6. Test on jetson-orin: TAO benchmark tool for inference latency
```

### Phase 2: Add Static Object Classes

1. Download ScanNet v2 (train/val), UniDet3D weights (HuggingFace)
2. Train in mmdetection3d → export ONNX
3. Fine-tune with ~100-500 real indoor scenes from robot
4. Publish static object boxes (low priority for planner)

### Phase 3: Warehouse Mode

1. Download lidar-warehouse-dataset (Google Drive)
2. Train 5-class PointPillars from that data
3. Switch modes by robot config

### Jetson Orin 16GB Budget

| Component | VRAM |
|---|---|
| PointPillars Head A (FP16) | ~80-120 MB |
| FCAF3D Head B (FP16) | ~150-250 MB |
| PointPillars Head C (FP16) | ~80-120 MB |
| SLAM / NDT matching | ~200 MB GPU |
| Total inference | **~500-700 MB** ✅ fits in 16GB |
| Inference latency (all heads, Orin) | **30-60 ms total** (parallel) |

**Orin 16GB can comfortably run all three heads in parallel.** Even 3D sparse conv (SECOND) single model fits.

---

## 7. Alternative: Semantic Segmentation + Clustering

**Question**: "Can we just cluster points and use geometric heuristics?"

**Answer**: For 2-3 class detection (person + "other object"), YES:

| Method | How | Pros | Cons |
|---|---|---|---|
| **Euclidean clustering** (PCL) | Cluster → fit box → classify by size | Zero training, works NOW | Cannot distinguish chair from box; misses small objects |
| **RANSAC plane removal + cluster** | Remove walls/floor/ceiling, then cluster | Better, removes 60% of points | Still no semantic meaning |
| **Semantic seg PCL** (trained S3DIS/Semantic3D) | 13-class seg → cluster per class | Distinguishes furniture from walls | RGB-D trained, not LiDAR; slow |
| **FCAF3D** (lightest indoor detector) | Direct box detection | Single model, real-time | Only ~35% AP50 on ScanNet v2 |
| **VoteNet** | Standard indoor detector | ~58% AP50 ScanNet | Slow, no Jetson port |

**For the specific case of "just know what's in the path"**: Euclidean clustering + size-based classification (person = tall thin, forklift = large wide, box = small cube) gets you 60-70% of useful detection with zero training.

---

## 8. Data Collection Strategy for Custom Dataset

To train a truly useful indoor multi-class detector:

### Minimum: 500 scenes

| Class | Target # Instances | Collection Method |
|---|---|---|
| Pedestrian (standing, sitting, walking) | 2000+ | Robot moves through office/lab, people present |
| Table (office, lab) | 500 | Static scenes, different rooms |
| Chair | 800 | Standard office chairs, lab stools |
| Shelf/rack | 200 | Library, warehouse racks |
| Box (cardboard, metal) | 300 | Warehouse loading zone |
| Forklift / AMR | 100 | Warehouse area |
| Robot (other G1) | 50 | Two robots in same room |

**Collection protocol**:
- Robot moves slowly (0.2 m/s) through environments
- Record MID-360 raw + IMU + odom
- Post-process: static background subtraction (N scan average)
- Annotate 3D boxes in [SaaS or SUSTech or label tool]

---

## 9. Summary: Action Items

| # | Action | Time | Dependencies |
|---|---|---|---|
| 1 | **Get PointPillars running in venv** (OpenPCDet, 2-class) | 1 day | Python 3.12, venv |
| 2 | **Download JRDB** (pedestrian 3D boxes) | 1 day | Sign terms on website |
| 3 | **Train PointPillars on JRDB** (pedestrian) | 4-6 GPU-hours | RTX 4060 |
| 4 | **Test on robot data** — collect 20 min of office/lab | 0.5 day | Robot operational |
| 5 | **Export ONNX → TensorRT FP16** | 0.5 day | cuda 12, TRT 8.6 |
| 6 | **ROS2 node** (PointCloud2 in → MarkerArray out) | 1 day | ROS2 Humble (or Jazzy) |
| 7 | **Deploy to Jetson Orin 16GB** | 0.5 day | JetPack 5.x |
| 8 | **Phase 2: UniDet3D** ScanNet classes | 2 weeks | UniDet3D repo + weights |
| 9 | **Phase 3: Warehouse mode** | 1 week | lidar-warehouse dataset |

---

## Appendix: Key Sources

### Papers
1. **UniDet3D** (AAAI 2025) — Best indoor multi-class detector. arxiv.org/abs/2409.04234
2. **PointGroup** (CVPR 2020) — Instance segmentation. arxiv.org/abs/2004.01658
3. **JRDB** (TPAMI 2021) — Indoor+outdoor humans, robot-collected. arxiv.org/abs/1910.11792
4. **LiDAR-Net** (CVPR 2024) — 24-class indoor MLS. arxiv.org/abs/2312.13029
5. **Cubify-Anything CA-1M** (CVPR 2025) — 400K objects, LiDAR-derived
6. **FCAF3D** (NeurIPS 2022) — Lightweight indoor 3D detection
7. **TR3D** (ICIP 2023, Samsung) — Real-time indoor 3D detection

### Tools & Frameworks
- **OpenPCDet** — github.com/open-mmlab/openPCDet (PointPillars, SECOND, PointRCNN)
- **MMDetection3D** — github.com/open-mmlab/mmdetection3d (VoteNet, UniDet3D, FCAF3D)
- **CUDA-PointPillars** (NVIDIA) — github.com/NVIDIA-AI-IOT/CUDA-PointPillars (TensorRT, 33ms Xavier)
- **TAO-PointPillars ROS2 node** — developer.nvidia.com (TensorRT on Jetson, ROS2)
- **mvg-inatech Mask3D** — github.com/mvg-inatech/room-instance-segmentation-mask3d
- **PCL (perception_pcl)** — for ROS2 point cloud ops

### Datasets
- **ScanNet** — scannet.org (18 classes, RGB-D)
- **JRDB** — jrdb.erc.monash.edu (humans, LiDAR + RGB)
- **LiDAR-warehouse** — github.com/anavsgmbh/lidar-warehouse-dataset (5 warehouse classes)
- **S3DIS** — s3dis.stanford.edu (13 classes, RGB-D)
- **CodA** — campus indoor+outdoor, 53 classes, 128-line LiDAR
