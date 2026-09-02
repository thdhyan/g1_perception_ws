# Plan: LiDAR Human Pose Estimation

## Objective

Add 3D human pose estimation (skeleton keypoints) from LiDAR point clouds for indoor use.

## Why Pose Estimation?

1. **Complementary features**: Skeleton geometry (limb lengths, joint angles) provides identity information beyond appearance
2. **Activity recognition**: Pose enables understanding human actions
3. **Better tracking**: Skeleton-based features can improve long-term re-identification

## Options Evaluated

See `docs/research_pose_estimators.md` for detailed comparison.

## Recommended: VoxelKP (ICCV 2025)

**Why VoxelKP**:
- Fully sparse architecture (efficient for LiDAR)
- 14 keypoints (nose, shoulders, elbows, wrists, hips, knees, ankles, head)
- Built on sparse convolution (similar to VoxelNeXt)
- 27% improvement over previous SOTA on Waymo
- Code available: https://github.com/shijianjian/VoxelKP

**Concerns**:
- Trained on Waymo (outdoor)
- May need fine-tuning for indoor
- Inference speed unknown on Jetson

## Alternative: Simplified Skeleton Estimator

If VoxelKP is too heavy, consider:
1. **Bounding box + heuristics**: Estimate skeleton from box dimensions
2. **MediaPipe + LiDAR**: Use camera for 2D pose, LiDAR for 3D depth
3. **Lift3D-style**: Lift 2D detections to 3D skeleton

## Phase 1: Research (NOW)

- [ ] Test VoxelKP on indoor data (if code is available)
- [ ] Measure inference speed on RTX 4060
- [ ] Evaluate keypoint accuracy indoors

## Phase 2: Integration (Week 2-3)

- [ ] Create ROS2 node for pose estimation
- [ ] Integrate with tracking pipeline
- [ ] Add skeleton features to ReID embedding

## Phase 3: Optimization (Week 4)

- [ ] TensorRT optimization for Jetson
- [ ] Benchmark end-to-end pipeline

## Decision

**Defer until ReID upgrade is complete**. Pose estimation is a medium-priority enhancement, not a blocker.

## Success Criteria

1. Pose estimation runs at ≥5 FPS on laptop
2. Keypoint localization error <10cm at 5m range
3. Integrates with existing tracking pipeline
