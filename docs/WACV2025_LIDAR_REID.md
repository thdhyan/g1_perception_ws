# LiDAR Person ReID - WACV 2025 & Latest Research

## Summary

| Model | Venue | Year | Type | Pre-trained | GitHub |
|-------|-------|------|------|-------------|--------|
| **point-cloud-reid** | WACV | 2024 | Object ReID (incl. persons) | ✅ Yes | [bentherien/point-cloud-reid](https://github.com/bentherien/point-cloud-reid) |
| **ReID3D** | CVPR | 2024 | LiDAR Person ReID | ❌ No | [GWxuan/ReID3D](https://github.com/GWxuan/ReID3D) |
| **GaitCloud** | WACV | 2025 | LiDAR Gait Recognition | ❌ No | [seagrgz/GaitCloud-master](https://github.com/seagrgz/GaitCloud-master) |
| **MOJO** | IEEE TIFS | 2025 | 4D LiDAR Person ReID | ❌ No | [O-VIGIA/MOJO](https://github.com/O-VIGIA/MOJO) |
| **CLFormer** | IV | 2025 | Camera-LiDAR Person ReID | ❌ No | Not available |

## Current Download Status

### point-cloud-reid (WACV 2024) - IN PROGRESS

**Server**: https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/

| Model | Expected Size | Current Size | Status |
|-------|---------------|--------------|--------|
| `pts_pointnet_r_nus_det_500e.pth` | 472 MB | 216 MB | ⏳ Downloading |
| `pts_point-transformer_r_nus_det_500e.pth` | 427 MB | 345 MB | ⏳ Downloading |

**Direct Download Links** (if you want to download manually):

```bash
# PointNet model (472 MB)
wget -c "https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/pts_pointnet_r_nus_det_500e.pth"

# Point-Transformer model (427 MB) - STRONGEST
wget -c "https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/pts_point-transformer_r_nus_det_500e.pth"

# Image model (73 MB) - not needed for LiDAR
wget "https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/rgb_deit-tiny_pt_nus_det_200e.pth"
```

**Place files in**: `~/Projects/thesis/g1_perception_ws/point-cloud-reid/pretrained/nuscenes/`

### ReID3D (CVPR 2024)

**Dataset**: https://cloud.tsinghua.edu.cn/d/cdcdab829e184a698b63/

**No pre-trained models available** - must train from scratch on LReID dataset.

**Dependencies** (heavy):
- PyTorch 1.12.1
- pytorch3d 0.7.4 (requires CUDA compilation)

### GaitCloud (WACV 2025)

**Dataset**: SUSTech1K (gait recognition, not person ReID)

**No pre-trained models** - must train on SUSTech1K dataset.

### MOJO (IEEE TIFS 2025)

**Code**: https://github.com/O-VIGIA/MOJO

**Status**: Repo exists but README is empty. No pre-trained models.

---

## Best Option for Your Use Case

### Recommendation: **point-cloud-reid** (WACV 2024)

**Why**:
1. ✅ Pre-trained models available (when download completes)
2. ✅ Works with any point cloud backbone (PointNet, DGCNN, Point-Transformer)
3. ✅ Lightweight RTMM matching head (real-time inference)
4. ✅ Tested on nuScenes (autonomous driving, similar to your use case)
5. ✅ Pure PyTorch inference (no mmdet3d needed)

**Architecture**:
- Backbone: Point-Transformer (strongest) or PointNet
- Matching Head: RTMM (Real-Time Matching Module)
- Input: Point cloud crop (N points × 3 coordinates)
- Output: 128-d L2-normalised embedding

**Performance**:
- Rank-1 accuracy: ~90% (rigid objects), ~85% (persons)
- Inference speed: ~10 Hz (1000 pairwise comparisons)

### Alternative: **ReID3D** (CVPR 2024)

**Why consider**:
1. ✅ First LiDAR-specific person ReID (320 identities, Livox Mid-100)
2. ✅ Uses graph neural network (GNN) for 3D features
3. ✅ Pre-training on synthetic dataset (LReID-sync)

**Why avoid**:
1. ❌ No pre-trained models
2. ❌ Requires pytorch3d (heavy dependency)
3. ❌ Must train from scratch (weeks of training)

---

## Manual Download Instructions

If the automatic download fails, here are the direct links:

### point-cloud-reid Models

```bash
# Navigate to the directory
cd ~/Projects/thesis/g1_perception_ws/point-cloud-reid/pretrained/nuscenes

# Download PointNet (472 MB)
wget -c "https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/pts_pointnet_r_nus_det_500e.pth"

# Download Point-Transformer (427 MB) - RECOMMENDED
wget -c "https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/pts_point-transformer_r_nus_det_500e.pth"

# Verify downloads
ls -lh *.pth
# Expected: 472M and 427M respectively
```

### ReID3D Dataset (for training)

```bash
# Download LReID dataset
cd ~/Projects/thesis/g1_perception_ws/ReID3D
wget -c "https://cloud.tsinghua.edu.cn/d/cdcdab829e184a698b63/files/?dl=1" -O LReID.zip

# Unzip
unzip LReID.zip -d data/
```

---

## Integration Status

### point-cloud-reid
- [x] Repo cloned
- [x] Inference script created (`reid_embed_pointcloudreid.py`)
- [ ] Models downloaded (in progress)
- [ ] Inference tested
- [ ] Integrated with pipeline

### ReID3D
- [x] Repo cloned
- [ ] Dataset downloaded
- [ ] Dependencies installed
- [ ] Model trained
- [ ] Inference tested

---

## Next Steps

### Immediate (Today)
1. **Wait for point-cloud-reid downloads** (~5-10 min at current speed)
2. **Test inference** with downloaded models
3. **Compare** with existing PointNet model

### Short-term (This Week)
1. **Fine-tune** on indoor data if needed
2. **Evaluate** RTMM matching vs raw cosine similarity
3. **Benchmark** inference speed on RTX 4060

### Medium-term (Next Week)
1. **Consider ReID3D** if point-cloud-reid underperforms
2. **Train on LReID dataset** for outdoor generalization
3. **Deploy** to Jetson Orin

---

## Files Reference

| File | Purpose |
|------|---------|
| `point-cloud-reid/pretrained/nuscenes/*.pth` | Pre-trained models |
| `reid_embed_pointcloudreid.py` | Inference script |
| `reid_model.py` | Existing PointNet model |
| `reid_data/model.pt` | Existing trained checkpoint |
| `docs/REID_INFERENCE_STATUS.md` | Inference guide |

---

**Last updated**: 2026-09-02
**Status**: ⏳ point-cloud-reid models downloading
