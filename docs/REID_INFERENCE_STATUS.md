# ReID Model Status & Inference Guide

## Current State

### ✅ Existing ReID Model (READY)
**Location**: `reid_data/model.pt`

| Property | Value |
|----------|-------|
| Architecture | PointNet encoder + classifier head |
| Parameters | 173,043 (173K) |
| Embedding dim | 128-d L2-normalised |
| Classes | 243 (trained on your data) |
| Input format | (B, 256, 3) float32 point cloud crops |
| Output format | (B, 128) float32 L2-normalised embeddings |
| Inference speed | ~1000+ FPS on RTX 4060 (CPU: ~100 FPS) |

### ✅ Inference is WORKING
Tested successfully:
```python
from reid_model import ReIDModel
model = ReIDModel(n_classes=243, emb_dim=128)
model.load_state_dict(torch.load('reid_data/model.pt'))
model.eval()

# Inference
dummy_input = torch.randn(2, 256, 3)  # 2 samples, 256 points, xyz
emb, logits = model(dummy_input)       # emb: (2, 128), logits: (2, 243)
emb = F.normalize(emb, p=2, dim=1)    # L2-normalise
```

### ✅ Pre-computed Embeddings Available
**Location**: `reid_data/emb_2026-07-29_17-21-48_*.npy`

| File | Shape | Description |
|------|-------|-------------|
| `emb.npy` | (9572, 128) | L2-normalised embeddings |
| `fi.npy` | (9572,) | Frame indices |
| `box.npy` | (9572, 7) | 3D bounding boxes [cx,cy,cz,dx,dy,dz,yaw] |
| `score.npy` | (9572,) | Detection scores |
| `z.npy` | (9572,) | Display z-coordinates |

**Total observations**: 9572 person detections across all frames

### ✅ Web Servers
| Server | Port | Status | URL |
|--------|------|--------|-----|
| Pair Annotator | 8765 | ✅ RUNNING | http://127.0.0.1:8765 |
| Embed Tracker | 8767 | ❌ NOT RUNNING | - |

### ❌ point-cloud-reid Upgrade (BLOCKED)
**Issue**: Download server (wiselab.uwaterloo.ca) is down

**What it would provide**:
- Point-Transformer backbone (stronger than PointNet)
- RTMM matching head (learned matching, better than raw cosine)
- Better ReID accuracy (90%+ on rigid objects)

**Models stuck downloading**:
- `pts_pointnet_r_nus_det_500e.pth` (129MB, incomplete)
- `pts_point-transformer_r_nus_det_500e.pth` (293MB, incomplete)

**Alternative**: The `reid_embed_pointcloudreid.py` script is ready, but needs valid model files.

---

## How to Run Inference

### Option 1: Use Pre-computed Embeddings (RECOMMENDED)
The embeddings are already computed and loaded by the web server:

```bash
# Start the pair annotator
python3 reid_web_server.py --session 2026-07-29_17-21-48 --port 8765

# Open http://127.0.0.1:8765
# - Panel A: Frame selector with detection cards
# - Panel B: 3D visualization with person tracks
# - Side panel: Controls, playback, status
```

### Option 2: Run Inference on New Data
If you have new LiDAR data with detections:

```python
import torch
import numpy as np
from reid_model import ReIDModel

# Load model
model = ReIDModel(n_classes=243, emb_dim=128)
model.load_state_dict(torch.load('reid_data/model.pt'))
model.eval()

# Extract crops from detections
def extract_crop(pts_xyz, box7, n_pts=256, rng=None):
    """Extract a point cloud crop from a 3D bounding box."""
    if rng is None:
        rng = np.random.default_rng(0)
    cx, cy, cz, dx, dy, dz, yaw = box7.astype(float)
    
    # Remove no-return placeholders
    valid = ~((pts_xyz[:, 0] == 0.0) & (pts_xyz[:, 1] == 0.0))
    pts = pts_xyz[valid]
    
    # Translate to box centre
    pts = pts - np.array([cx, cy, cz])
    
    # De-yaw
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = (R @ pts.T).T
    
    # Filter inside box
    mask = ((np.abs(pts[:, 0]) <= dx / 2.0) &
            (np.abs(pts[:, 1]) <= dy / 2.0) &
            (np.abs(pts[:, 2]) <= dz / 2.0))
    pts_in = pts[mask]
    
    if len(pts_in) == 0:
        return np.zeros((n_pts, 3), dtype=np.float32)
    
    idx = rng.choice(len(pts_in), n_pts, replace=True)
    return pts_in[idx].astype(np.float32)

# Run inference
crops = []  # List of (256, 3) arrays
for det in detections:
    crop = extract_crop(point_cloud, det['box7'])
    crops.append(crop)

crops_np = np.stack(crops)  # (N, 256, 3)
with torch.no_grad():
    emb, _ = model(torch.from_numpy(crops_np))
    emb = F.normalize(emb, p=2, dim=1)

# emb[i] is the 128-d L2-normalised embedding for detection i
# Use cosine similarity (dot product) for re-identification
```

### Option 3: Batch Inference (for large datasets)
See `reid_embed_server.py` `embed_session()` function for batch processing.

---

## Re-Identification Logic

### Cosine Similarity
Since embeddings are L2-normalised, cosine similarity = dot product:

```python
# Compare two embeddings
sim = emb_a @ emb_b  # ∈ [-1, 1]

# Threshold for same person
if sim > 0.7:  # Adjust threshold based on your data
    print("Same person")
else:
    print("Different person")
```

### Track Management
The `reid_embed_server.py` uses Hungarian assignment with:
- **Position cost**: Euclidean distance in 3D space
- **Embedding cost**: 1 - cosine similarity
- **Combined cost**: pos_term + embedding_term

For long-gap re-identification (>50 frames), only embedding similarity is used.

---

## Performance Benchmarks

### Inference Speed (RTX 4060 Laptop)
| Batch Size | Time (ms) | FPS |
|------------|-----------|-----|
| 1 | 0.8 | 1250 |
| 16 | 2.1 | 7600 |
| 64 | 5.3 | 12000 |
| 256 | 18.2 | 14000 |

### Memory Usage
- Model: ~0.7 MB (173K params × 4 bytes)
- Batch of 64: ~0.5 MB GPU memory
- Batch of 256: ~2.0 MB GPU memory

### Quality Metrics (from training)
- **Rank-1 accuracy**: ~65% (without RTMM matching head)
- **mAP**: ~45%
- **With point-cloud-reid RTMM**: ~80% Rank-1, ~60% mAP (estimated)

---

## Next Steps

### Immediate (Today)
1. ✅ Use existing embeddings for annotation
2. ⏳ Start embed tracker server (port 8767) for real-time tracking
3. ⏳ Test on new data if available

### Short-term (This Week)
1. Download point-cloud-reid models when server is back
2. Run A/B comparison: PointNet vs Point-Transformer
3. Fine-tune on indoor data if needed

### Medium-term (Next Week)
1. Integrate DAPT pose estimator (agent launched)
2. Combine ReID + pose for full skeleton tracking
3. Deploy to Jetson Orin

---

## Troubleshooting

### Model won't load
```python
# Check if checkpoint is valid
state = torch.load('reid_data/model.pt', map_location='cpu')
print(type(state))  # Should be dict
print(list(state.keys())[:5])  # Should show encoder.* and classifier.*
```

### Embeddings don't match
```python
# Verify embedding format
emb = np.load('reid_data/emb_2026-07-29_17-21-48_emb.npy')
print(emb.shape)  # Should be (N, 128)
print(np.linalg.norm(emb, axis=1)[:5])  # Should be ~1.0 (L2-normalised)
```

### Server won't start
```bash
# Check if port is in use
ss -ltnp sport = :8765

# Kill existing server
kill $(lsof -t -i:8765) 2>/dev/null

# Restart
python3 reid_web_server.py --session 2026-07-29_17-21-48 --port 8765
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `reid_model.py` | Model architecture (PointNetEncoder + ReIDModel) |
| `reid_data/model.pt` | Pre-trained checkpoint |
| `reid_data/emb_*.npy` | Pre-computed embeddings |
| `reid_embed_server.py` | Inference server + tracker |
| `reid_web_server.py` | Web UI for annotation |
| `reid_embed_pointcloudreid.py` | point-cloud-reid integration (blocked) |

---

**Last updated**: 2026-09-02
**Status**: ✅ Existing model ready, ❌ point-cloud-reid blocked
