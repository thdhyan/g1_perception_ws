# LiDAR → Person Identification & ReID Pipeline

## Overview

This document shows the complete data flow from raw LiDAR scans to person re-identification.

---

## Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAW LIDAR DATA                                      │
│                                                                              │
│  Source: Livox MID-360 (360° FoV, ~200k points/sec, non-repetitive)        │
│  Format: .lvx2 files → converted to .npy (N × 4: x, y, z, intensity)       │
│  Location: ~/Projects/Thesis/Lidar Data/frames/{session}/frame_XXXXX.npy    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      3D OBJECT DETECTION                                     │
│                                                                              │
│  Model: VoxelNext (or VoxelNeXt)                                            │
│  Framework: OpenPCDet                                                         │
│  Input: Point cloud (N × 3)                                                  │
│  Output: detections.npz with:                                                │
│    - pred_boxes: (K, 7) — [cx, cy, cz, dx, dy, dz, yaw]                   │
│    - pred_scores: (K,) — confidence scores                                   │
│    - pred_labels: (K,) — class IDs (1=car, 2=pedestrian, 3=cyclist)         │
│    - class_names: ['car', 'pedestrian', 'cyclist']                          │
│                                                                              │
│  Output file: {session}_frames_voxelnext.npz                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FILTERING & PREPROCESSING                               │
│                                                                              │
│  1. Label Filter: Keep only label=2 (pedestrian)                            │
│  2. Score Filter: Keep score ≥ 0.2 (configurable)                           │
│  3. Range Filter: Keep detections ≤ 15m from sensor                         │
│  4. Point Count: Keep detections with ≥ 100 points in crop                  │
│                                                                              │
│  Before filtering: ~14,405 detections (session 2026-07-29)                  │
│  After filtering: ~1,193 detections (with score≥0.2, points≥100)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CROP EXTRACTION                                         │
│                                                                              │
│  For each filtered detection:                                                │
│    1. Load frame point cloud (N × 3)                                        │
│    2. Extract points inside 3D bounding box                                  │
│    3. Center at box centroid (cx, cy, cz)                                    │
│    4. De-yaw (rotate by -yaw)                                                │
│    5. Sample 128 points (or 256 for existing model)                         │
│    6. Output: (128, 3) float32 crop                                         │
│                                                                              │
│  Function: extract_crop(pts_xyz, box7, n_pts=128)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EMBEDDING EXTRACTION                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Model A: Existing PointNet (173K params)                           │    │
│  │   Architecture: Conv1d(3→64→128→256) + MaxPool + FC(256→128)      │    │
│  │   Input: (B, 256, 3)                                                │    │
│  │   Output: (B, 128) L2-normalised embedding                         │    │
│  │   Training: Indoor data, 243 classes (person IDs)                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Model B: point-cloud-reid PointNet (109M params)                   │    │
│  │   Architecture: PointNet2 backbone + RTMM matching head            │    │
│  │   Input: (B, 128, 3)                                                │    │
│  │   Output: (B, 128) L2-normalised embedding                         │    │
│  │   Training: nuScenes (outdoor driving), WACV 2024                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Output: emb_{session}_emb.npy (N × 128)                                    │
│          emb_{session}_fi.npy (N,) — frame indices                          │
│          emb_{session}_box.npy (N × 7) — bounding boxes                    │
│          emb_{session}_score.npy (N,) — confidence scores                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PERSON IDENTIFICATION                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Method A: Embedding Clustering (cosine similarity > threshold)     │    │
│  │   - Compute pairwise cosine similarity                             │    │
│  │   - Greedy clustering: assign to existing cluster if sim > 0.7     │    │
│  │   - Result: N unique people identified                             │    │
│  │   - Problem: Both models output 99%+ similar embeddings → 1 person │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Method B: Temporal Tracking (Hungarian assignment)                  │    │
│  │   - Track people across frames using position + appearance         │    │
│  │   - Cost = position_distance + embedding_similarity                │    │
│  │   - Long-gap re-ID: embedding matching when position gate fails   │    │
│  │   - Used by: reid_embed_server.py (port 8767)                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Method C: Manual Annotation (pair annotator)                       │    │
│  │   - Human annotates pairs: same person / different person          │    │
│  │   - Used to train/evaluate ReID models                             │    │
│  │   - Used by: reid_web_server.py (port 8765)                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RE-IDENTIFICATION                                       │
│                                                                              │
│  Given: New detection at frame t with embedding e_t                         │
│                                                                              │
│  1. Compare e_t with all active track embeddings                            │
│  2. Compute similarity: sim = e_t · e_track (dot product)                   │
│  3. If sim > threshold (0.7) → assign to existing track                     │
│  4. If no match → create new track                                          │
│                                                                              │
│  Output: Track ID for each detection across all frames                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION & OUTPUT                                  │
│                                                                              │
│  Web UI (port 8765):                                                        │
│    - Panel A: Frame selector with detection cards                           │
│    - Panel B: 3D visualization with person tracks                          │
│    - Side panel: Controls, playback, status                                 │
│                                                                              │
│  Embed Tracker (port 8767):                                                 │
│    - Real-time tracking with Hungarian assignment                           │
│    - Embedding-based re-identification                                       │
│                                                                              │
│  Analysis Plots (reid_analysis/):                                           │
│    - Trajectory comparison                                                   │
│    - t-SNE embedding visualization                                          │
│    - Similarity heatmaps                                                    │
│    - Detection statistics                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
Raw LiDAR (.lvx2)
    ↓
Point Cloud (.npy) — [N × 4: x, y, z, intensity]
    ↓
VoxelNext Detection — [K × 7 boxes, K scores, K labels]
    ↓
Filter (label=2, score≥0.2, range≤15m, points≥100)
    ↓
Crop Extraction — [128 × 3 per detection]
    ↓
Embedding Model — [128-d L2-normalised vector]
    ↓
Clustering/Tracking — [Person ID per detection]
    ↓
ReID — [Match across frames/sessions]
```

---

## Current Limitations

### Problem: Embeddings Not Discriminative

**Root cause**: Both models trained on **outdoor driving data** (nuScenes/Waymo), not indoor pedestrians.

**Evidence**:
- Intra-frame similarity: 0.995 (different people in same frame are 99.5% similar!)
- Inter-frame similarity: 0.997 (same person across frames)
- Result: All detections map to 1 "person" cluster

**Impact**:
- Cannot distinguish between different people
- Cannot re-identify people after occlusion
- Only useful for detection, not ReID

### Solution Paths

1. **Fine-tune on indoor data** (best but requires labeled data)
2. **Use temporal tracking** (position-based, works now)
3. **Collect labeled data** (20+ people, multiple sessions)
4. **Use RTMM matching head** (learned matching, needs training)

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `reid_embed_server.py` | Embedding extraction + tracking (port 8767) |
| `reid_web_server.py` | Web UI for annotation (port 8765) |
| `reid_embed_pointcloudreid.py` | point-cloud-reid inference |
| `compare_reid_embeddings.py` | Embedding analysis & visualization |
| `reid_model.py` | Existing PointNet architecture |
| `reid_data/model.pt` | Trained checkpoint |
| `reid_data/emb_*.npy` | Pre-computed embeddings |

---

## Thresholds Used

| Parameter | Value | Effect |
|-----------|-------|--------|
| `min_score` | 0.2 | Filter low-confidence detections |
| `min_points` | 100 | Filter detections with too few points |
| `max_range` | 15.0m | Filter distant detections |
| `n_points` | 128 | Points per crop (point-cloud-reid) |
| `n_pts` | 256 | Points per crop (existing model) |
| `sim_threshold` | 0.7 | Cosine similarity for same-person |

---

**Last updated**: 2026-09-02
