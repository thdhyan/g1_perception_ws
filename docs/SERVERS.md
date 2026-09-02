# g1_perception_ws — Server Reference

## Running servers

| Port | Script | Status | Purpose |
|------|--------|--------|---------|
| **8767** | `reid_embed_server.py` | ✅ running | Main viewer — SMPL-mode ReID tracker + 3D trajectories + SMPL mesh rendering |
| **8766** | `reid_tracks_server.py` | ⬜ stopped | Legacy slot-based tracker (fixed N people, linear interp, no ReID embeddings) |
| **8765** | `reid_web_server.py` | ⬜ stopped | Pair annotator — side-by-side 3D views to label same/different person pairs |

---

## Port 8767 — reid_embed_server (SMPL mode) ← **use this one**

Main research server. Loads pre-extracted SMPL β vectors, runs Hungarian tracker with β-based ReID, serves 3D trajectory viewer with live SMPL mesh rendering.

**Restart:**
```bash
cd ~/Projects/thesis/g1_perception_ws && source .venv/bin/activate
nohup python3 reid_embed_server.py \
  --smpl-mode \
  --smpl-checkpoint humanm3 \
  --session 2026-07-29_17-21-48 \
  --short-gap 20 \
  --cos-thresh 0.5 \
  --pos-gate 3.0 \
  --max-range 5.0 \
  --port 8767 \
  > /tmp/reid8767.log 2>&1 &
```

**Key flags:**
| Flag | Value | Meaning |
|------|-------|---------|
| `--smpl-mode` | — | Use β (10-d shape) instead of 128-d PointNet embeddings |
| `--smpl-checkpoint` | `humanm3` | Which LiDAR-HMR checkpoint produced the `.npy` files |
| `--session` | `2026-07-29_17-21-48` | Session to load (must have matching `reid_data/smpl_humanm3_<session>_*.npy`) |
| `--short-gap` | 20 | Frames ≤ this → pos+β cost; beyond → β-only re-ID |
| `--cos-thresh` | 0.5 | Cosine similarity threshold for long-gap re-ID |
| `--pos-gate` | 3.0 | Max XY distance (m) for position match |
| `--max-range` | 5.0 | Drop detections > 5 m from sensor |

**API endpoints:**
- `GET /` — 3D viewer UI
- `GET /api/tracks` — all track data + anchor β per track (JSON)
- `GET /api/frame?fi=N` — LiDAR point cloud for frame N (base64 float32)
- `GET /api/mesh?tid=T&fi=F` — SMPL mesh vertices at frame F for track T (temporally smoothed)
- `GET /api/meshfaces` — SMPL face index buffer (base64, sent once on init)

**Session results (2026-07-29_17-21-48):**
- 10 tracker fragments → **3 distinct people** (silhouette = 0.624 at k=3)
- Run `python3 cluster_identities.py --port 8767` to recompute identity clusters

---

## Port 8766 — reid_tracks_server (legacy)

Fixed-slot tracker. No ReID embeddings. Useful for comparison / sanity checks.

**Restart:**
```bash
cd ~/Projects/thesis/g1_perception_ws && source .venv/bin/activate
python3 reid_tracks_server.py \
  --session 2026-07-29_17-21-48 \
  --people 5 \
  --port 8766
```

---

## Port 8765 — reid_web_server (pair annotator)

Side-by-side 3D pair labeling tool. Used to build ground-truth same/different-person labels for training the ReID model.

**Restart:**
```bash
cd ~/Projects/thesis/g1_perception_ws && source .venv/bin/activate
python3 reid_web_server.py \
  --session 2026-07-29_17-21-48 \
  --port 8765
```

Controls: `Y` same · `N` different · `Space` skip · `←` previous · `Q` finish  
Annotations saved to `reid_data/sameperson_<session>.json` (resumable).

---

## Available sessions

| Session | Frames | SMPL detections | Notes |
|---------|--------|-----------------|-------|
| `2026-07-29_17-21-48` | ~2000 | 4955 (humanm3) | Main session. 3 people identified. |
| `2026-08-05_16-38-40` | ~500 | 72 (humanm3, min-score 0.1) | Weaker detections; needs `--max-range 7.0` |

## Pre-extracted SMPL data

Files live in `reid_data/smpl_<checkpoint>_<session>_*.npy`:

| Suffix | Shape | Content |
|--------|-------|---------|
| `_beta.npy` | (N, 10) | Body shape vectors |
| `_theta.npy` | (N, 72) | Pose axis-angle |
| `_box.npy` | (N, 7) | Detection boxes `[cx,cy,cz,dx,dy,dz,yaw]` |
| `_fi.npy` | (N,) | Frame indices |
| `_score.npy` | (N,) | VoxelNext confidence scores |

Re-extract with:
```bash
python3 extract_smpl.py --session 2026-07-29_17-21-48 --checkpoint humanm3
```

## Live ROS2 SMPL node (smpl_hmr_node)

Subscribes to live robot topics → runs LiDAR-HMR → publishes SMPL meshes + skeleton to RViz2.

**Start:**
```bash
# Terminal 1 — source workspace + venv
source /opt/ros/humble/setup.bash   # or iron/noetic
source ~/Projects/thesis/g1_perception_ws/install/setup.bash
source ~/Projects/thesis/g1_perception_ws/.venv/bin/activate

# Terminal 2 — node
ros2 launch g1_perception smpl_hmr.launch.py

# Terminal 3 — RViz2
rviz2 -d ~/Projects/thesis/g1_perception_ws/src/g1_perception/config/smpl_hmr.rviz
```

**Override options:**
```bash
ros2 launch g1_perception smpl_hmr.launch.py \
  checkpoint:=humanm3 \
  device:=cuda \
  min_score:=0.15 \
  max_range:=6.0 \
  show_boxes:=true
```

**Subscribed topics:**
| Topic | Type | Description |
|-------|------|-------------|
| `/livox/mid360/points` | `PointCloud2` | Raw LiDAR cloud |
| `/g1/detections/livox` | `Detection3DArray` | VoxelNext person boxes |

**Published topics (all MarkerArray):**
| Topic | Marker type | Content |
|-------|-------------|---------|
| `/g1/smpl/mesh` | `TRIANGLE_LIST` | Full SMPL body mesh (6890 verts, 13776 faces) per person |
| `/g1/smpl/joints` | `SPHERE_LIST` | 24 SMPL joints per person |
| `/g1/smpl/skeleton` | `LINE_LIST` | Skeleton edges per person |
| `/g1/smpl/boxes` | `CUBE` | Detection bounding boxes (off by default) |

Each person gets a unique color (teal/amber/coral/sky cycling). Markers expire after 0.5s if detections stop.

**Requires:** VoxelNext detection_bridge running (provides `/g1/detections/livox`).

---

## Utility scripts

| Script | Usage |
|--------|-------|
| `cluster_identities.py` | Cluster track β-anchors → count distinct people |
| `extract_smpl.py` | Run LiDAR-HMR on a session → write `reid_data/*.npy` |
| `compare_reid_embeddings.py` | Compare checkpoint β discrimination metrics |
| `extract_identity_crops.py` | Mine per-person crops for ReID model training |
