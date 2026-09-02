# ReID Model Comparison for LiDAR Point Clouds

## Context

We need person re-identification (ReID) from LiDAR point clouds to associate detections across frames and assign persistent identities. Current setup: simple PointNet (256 points → 128-d embedding) trained on our own data.

## Options Evaluated

### 1. ReID3D (CVPR 2024)
- **Architecture**: GCN backbone + Complementary Feature Extractor
- **Dataset**: LReID (320 identities, Livox Mid-100)
- **Pre-training**: LReID-sync (600 synthetic pedestrians)
- **Performance**: 94.0% rank-1 on LReID
- **Pros**: State-of-the-art, LiDAR-specific
- **Cons**: Requires PCDet + spconv v1.0 (legacy), LReID dataset uses Livox Mid-100 (similar to our MID-360), needs custom training pipeline
- **Verdict**: High effort, uncertain gain over simpler alternatives

### 2. point-cloud-reid (WACV 2024) ⭐ RECOMMENDED
- **Architecture**: PointNet / DGCNN / Point-Transformer + RTMM matching head
- **Dataset**: nuScenes ReID + Waymo ReID
- **Performance**: 90%+ rigid, 85%+ deformable (Waymo)
- **Inference**: 10 Hz (thousands of pairwise comparisons/sec)
- **Pre-trained models**: Available for download
- **Pros**: Ready to use, pluggable backbones, RTMM learned matching, tested on driving LiDAR
- **Cons**: Trained on outdoor driving data, may need fine-tuning for indoor
- **Verdict**: Lowest friction, immediate gains

### 3. MOJO (IEEE TIFS 2025)
- **Architecture**: Motion patterns + 3D joint graph
- **Input**: 4D LiDAR (requires temporal sequences)
- **Pros**: Uses motion information
- **Cons**: Requires 4D LiDAR, not compatible with single-scan MID-360
- **Verdict**: Not applicable

### 4. Our Current PointNet
- **Architecture**: 3-layer shared MLP + global max-pool → 128-d
- **Training**: Our own labelled data (59 annotations)
- **Performance**: Unknown (not benchmarked)
- **Pros**: Simple, fast, already integrated
- **Cons**: No pre-training, small dataset, no learned matching head
- **Verdict**: Functional but improvable

## Recommendation

**Adopt point-cloud-reid** (WACV 2024):
1. Download pre-trained Point-Transformer model
2. Extract embeddings on our data
3. Compare with current PointNet embeddings
4. Fine-tune on our data if needed

The RTMM matching head is the key advantage — it learns to compare two observations rather than relying on raw cosine similarity.

## Integration Plan

See `plans/PLAN_REID_UPGRADE.md`
