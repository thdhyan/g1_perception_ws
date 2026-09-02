# 3D Object Detection Models Comparison

## Context

We use VoxelNeXt for pedestrian detection from LiDAR point clouds. Need to evaluate if upgrading is beneficial, considering indoor use, MID-360 LiDAR, and Jetson Orin deployment.

## Current: VoxelNeXt (CVPR 2023)

| Metric | Value |
|---|---|
| mAP L2 (Waymo) | 70.7 |
| mAPH L2 (Waymo) | 69.6 |
| Latency | 38.7ms (~26 FPS) |
| Architecture | Fully sparse, no dense conversion |
| Indoor performance | Unknown (trained on outdoor driving) |

## Comparison Table

| Model | Year | mAP (Waymo) | Latency (ms) | FPS | Indoor | Jetson | ROS2 | Notes |
|---|---|---|---|---|---|---|---|---|
| VoxelNeXt | 2023 | 70.7 | 38.7 | 26 | ❓ | ✅ | ✅ | Current model |
| DSVT(Pillar) | 2023 | 73.2 | 67 | 15 | ❓ | ⚠️ | ❌ | Slower, more accurate |
| DSVT(Pillar)+TRT | 2023 | 73.2 | 37 | 27 | ❓ | ✅ | ❌ | Same speed, +2.5 mAP |
| CenterPoint | 2021 | 66.0 | 35 | 29 | ❓ | ✅ | ✅ | Fast, less accurate |
| FastPillars | 2023 | 70.6 | 31 | 32 | ❓ | ✅ | ❌ | Fastest |
| PillarNeXt | 2023 | 70.3 | 103 | 10 | ❓ | ⚠️ | ❌ | Too slow |
| UniMamba | 2025 | 70.2 (nuScenes) | TBD | TBD | ❓ | ❓ | ❌ | New architecture |

## Indoor Considerations

### Key Issues with Outdoor Trained Models
1. **Scale difference**: Indoor humans are 1-10m away, outdoor 10-80m
2. **Point density**: MID-360 at 5m has ~500 points/human, at 15m ~100 points
3. **Background**: Indoor has walls, furniture, ceiling — different from roads
4. **Scan pattern**: MID-360 has 360° FoV with Livox non-repetitive scanning

### Indoor-Specific Approaches
1. **DBSCAN clustering** + bounding box fitting — simple, no training needed
2. **PointPillars** — fast, works on any point cloud
3. **SECOND** — sparse convolution, good for indoor
4. **MinkUNet** — fully convolutional, good for segmentation

### Recommendation for Indoor
- **Keep VoxelNeXt** for now — it's fast and already integrated
- **Add DBSCAN clustering** as a fallback for indoor-specific scenarios
- **Consider fine-tuning** VoxelNeXt on indoor data if needed

## Jetson Orin Deployment

| Model | FP32 (GB) | INT8 (GB) | FP32 FPS | INT8 FPS |
|---|---|---|---|---|
| VoxelNeXt | ~2 | ~1 | ~15 | ~25 |
| DSVT(Pillar) | ~3 | ~1.5 | ~8 | ~15 |
| CenterPoint | ~1.5 | ~0.8 | ~18 | ~30 |

**Note**: These are estimates based on GPU memory and compute. Actual performance depends on point cloud size and post-processing.

## See Also
- `docs/research_indoor_detectors.md` — detailed indoor detector analysis (subagent)
- `plans/PLAN_DETECTION_UPGRADE.md` — upgrade plan
