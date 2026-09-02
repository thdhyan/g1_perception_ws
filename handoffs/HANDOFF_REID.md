# Handoff: Current ReID Pipeline

## Status: WORKING

The ReID system is fully functional with:
- Pair annotator (port 8765) — 1000 pairs, cosine similarity ranking
- Embedding tracker (port 8767) — 9 tracks, embedding-based tracking
- Position tracker (port 8766) — 7 slots, position-based tracking

## Architecture

```
LiDAR Frame
    ↓
VoxelNeXt (3D detection)
    ↓
Crop extraction (256 points, centered + de-yawed)
    ↓
PointNet embedding (128-d, L2-normalised)
    ↓
Cosine similarity matching
    ↓
Tracking (Hungarian assignment)
```

## Key Files

| File | Purpose |
|---|---|
| `reid_model.py` | PointNet architecture (128-d embedding) |
| `reid_embed_server.py` | Embedding extraction + tracking (port 8767) |
| `reid_web_server.py` | Pair annotator (port 8765) |
| `reid_tracks_server.py` | Position-based tracker (port 8766) |
| `reid_data/model.pt` | Trained model checkpoint |
| `reid_data/emb_*.npy` | Pre-computed embeddings |

## Data Format

### Embeddings
- `emb_{session}_emb.npy`: (N, 128) float32, L2-normalised
- `emb_{session}_fi.npy`: (N,) int32, frame indices
- `emb_{session}_box.npy`: (N, 7) float32, [cx,cy,cz,dx,dy,dz,yaw]
- `emb_{session}_score.npy`: (N,) float32, detection confidence

### Annotations
- `sameperson_{session}.json`: Binary same-person labels

## How to Run

```bash
# Pair annotator
python3 reid_web_server.py --session 2026-07-29_17-21-48 \
  --lidar-dir ~/Projects/Thesis/"Lidar Data" \
  --min-score 0.4 --max-dist 50.0 --gap-max 500 \
  --min-gap 80 --cos-thresh 0.7 --port 8765

# Embedding tracker
python3 reid_embed_server.py --session 2026-07-29_17-21-48 \
  --model reid_data/model.pt --port 8767

# Position tracker
python3 reid_tracks_server.py --session 2026-07-29_17-21-48 \
  --people 7 --port 8766
```

## Upcoming Changes

- Upgrade to point-cloud-reid (WACV 2024) — see `HANDOFF_POINT_CLOUD_REID.md`
- Indoor detection evaluation — see `plans/PLAN_DETECTION_UPGRADE.md`
- Pose estimation — see `plans/PLAN_POSE_ESTIMATION.md`
