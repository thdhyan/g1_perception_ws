# Plan: Detection Model Evaluation for Indoor Use

## Objective

Evaluate and potentially upgrade 3D object detection for indoor pedestrian detection with MID-360 LiDAR.

## Current State

- **Model**: VoxelNeXt (PCDet framework)
- **Performance**: 70.7 mAP on Waymo (outdoor)
- **Speed**: 26 FPS on RTX 4060
- **Indoor performance**: Unknown

## Phase 1: Baseline Evaluation (NOW)

- [ ] Run VoxelNeXt on indoor data (2026-07-29_17-21-48 session)
- [ ] Measure detection accuracy on known pedestrians
- [ ] Profile inference speed on RTX 4060
- [ ] Document failure modes (missed detections, false positives)

## Phase 2: Indoor-Specific Approaches (Week 1-2)

- [ ] Test DBSCAN clustering + bounding box fitting
  - No training required
  - Fast (~100 FPS)
  - Good for indoor humans at close range
- [ ] Evaluate PointPillars (simpler, faster)
- [ ] Test SECOND (sparse convolution)

## Phase 3: Fine-tuning (Week 3, if needed)

- [ ] Collect indoor detection data
- [ ] Fine-tune VoxelNeXt on indoor data
- [ ] Compare with outdoor-only model

## Phase 4: Jetson Optimization (Week 4)

- [ ] Export VoxelNeXt to ONNX
- [ ] TensorRT optimization (FP16 → INT8)
- [ ] Benchmark on Jetson Orin

## Decision Tree

```
Is VoxelNeXt detecting pedestrians indoors?
├── YES → Keep VoxelNeXt, optimize for Jetson
└── NO → Is it missing detections or false positives?
    ├── Missing → Try lower min_score threshold
    ├── False positives → Try NMS tuning or fine-tuning
    └── Both → Consider DBSCAN fallback or fine-tuning
```

## Success Criteria

1. Detection recall ≥80% for pedestrians within 15m indoors
2. Inference speed ≥10 FPS on laptop
3. Model fits in 2 GB VRAM on Jetson Orin

## Files to Create

- `test_indoor_detection.py` — evaluate VoxelNeXt on indoor data
- `indoor_detector.py` — DBSCAN-based indoor detector (fallback)
