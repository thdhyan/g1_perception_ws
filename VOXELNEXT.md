# VoxelNeXt 3D Object & Human Detection Backend

[VoxelNeXt](https://github.com/JIA-Lab-research/VoxelNeXt) (Chen et al., CVPR 2023) is a fully-sparse, anchor-free 3D object detector.

---

## 1. Why VoxelNeXt for G1 Humanoid Perception?

Existing 3D detectors suffer from distinct drawbacks when detecting humans from robot-mounted LiDAR:
- **Anchor-based pillar detectors**: 2D pillar footprint. Rely on rigid bounding box aspect ratio priors; struggle to generalize to non-slender humans (e.g. broad builds, heavy coats, or unusual postures).
- **Sparse-to-dense BEV projection detectors**: Often trained with high-mounted automotive LiDAR priors; sensitive to vertical point density changes.
- **VoxelNeXt**: **Directly predicts 3D bounding boxes from sparse 3D voxel features.** It has NO BEV conversion, NO anchors, and NO center proxies. This enables it to maintain full 3D spatial geometry and reliably detect humans regardless of body proportion, stance, or attire.

---

## 2. Architecture & Pipeline

```
 Livox Mid-360 PointCloud2 [x, y, z, intensity]
                    │
                    ▼
 Preprocessing (offset_ground=1.33m, 360° range [-54,-54,-5, 54,54,3])
                    │
                    ▼
 Dynamic Voxelization (0.075m voxel size)
                    │
                    ▼
 Sparse 3D ResNet Backbone (spconv-cu121 3D sparse convolutions)
                    │
                    ▼
 Fully-Sparse Voxel Head (Direct Box & Class Prediction on Sparse Voxels)
                    │
                    ▼
 Post-Processing & Class Mapping (nuScenes 10-Class -> G1 3-Class)
                    │
                    ▼
 Detection3DArray & Visual MarkerArray in Pelvis Frame
```

---

## 3. Directory Layout

- `src/livox_detection/livox_detection/voxelnext_model.py`: OpenPCDet backend wrapper
- `src/livox_detection/launch/voxelnext_detection.launch.py`: Standalone detection launch
- `src/g1_bringup/launch/real_human_follow.launch.py`: Integrated real robot launch
- `VoxelNeXt/`: Cloned upstream repository with compiled CUDA extensions (`pcdet`)
- `pt/`: Pretrained model checkpoints directory

---

## 4. Class Mapping

VoxelNeXt (nuScenes model) outputs 10 classes, mapped to G1 perception conventions:
| nuScenes Label | Output Name | G1 Mapped Class | G1 Class ID |
|---|---|---|---|
| 1 | `car` | `car` | 0 |
| 2 | `truck` | `car` | 0 |
| 3 | `construction_vehicle` | `car` | 0 |
| 4 | `bus` | `car` | 0 |
| 5 | `trailer` | `car` | 0 |
| 6 | `barrier` | *(filtered)* | -1 |
| 7 | `motorcycle` | `cyclist` | 2 |
| 8 | `bicycle` | `cyclist` | 2 |
| 9 | `pedestrian` | `pedestrian` | 1 |
| 10 | `traffic_cone` | *(filtered)* | -1 |

---

## 5. Usage Commands

### Test Backend Standalone:
```bash
source setup_g1_env.sh
python3 -c "from livox_detection.voxelnext_model import VoxelNeXtBackend; b = VoxelNeXtBackend(); print('VoxelNeXtBackend ready!')"
```

### Run Live Detection:
```bash
source setup_g1_env.sh
ros2 launch livox_detection voxelnext_detection.launch.py checkpoint_path:=pt/voxelnext_nuscenes.pth rviz:=true
```
