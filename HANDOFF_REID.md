# HANDOFF: Person ReID — VoxelKP Embedding Head + LUT-Based Identity

**Date:** 2026-08-27, session 1.
**Repo:** `g1_perception_ws` on `main`.
**Workstream:** Parallel to the SLAM/Isaac ROS stream in `HANDOFF.md`.
This document covers: (1) VoxelKP embedding head design, (2) self-supervised
ReID training from existing LiDAR data, (3) ROS2 plumbing for identity
assignment, (4) data tools.

---

## Goal

Give the G1 robot persistent person identity across frame boundaries:
- Detect person A, assign them an ID (e.g. `db:5`).
- When they walk out of frame and return 30–300 s later, recognise them
  as the same person without any camera or name input.
- Allow an operator to name a person via keyboard at any time; the name
  persists in a JSON lookup table, not in model weights.
- Architecture: VoxelKP sparse-CNN extracts a 128-D L2-normalised embedding
  per detection; a cosine-similarity LUT matches new detections to stored
  identities.

---

## Architecture Overview

```
LiDAR scan
    │
    ▼
VoxelNeXt backbone (frozen, Waymo pretrained)
    │   encoded_spconv_tensor  shape (N_vox, 384)
    ▼
VoxelNeXtHeadKPMerge
    ├── hm branch   → heatmap
    ├── dim branch  → 3-D size
    ├── rot branch  → heading
    ├── kp_* branches → keypoints
    ├── iou branch  → IOU score
    └── embed branch (NEW)  → (N_vox, 128), gathered at detection centres
                               → L2-normalise → per-detection embedding
    │
    ▼
ReIDLUT (Python, runtime, NOT in model weights)
    {person_db_id: EMA_embedding, last_seen, last_pos}
    │  cosine similarity threshold 0.82
    ▼
PersonReIDArray ROS2 message → /g1/detections/reid
    │
    ▼
person_namer_node  (keyboard naming UI)
    └── ~/.ros/person_names.json  {db:5: "Alice", ...}
```

---

## What Was Done This Session

### VoxelKP Installation Fix
- **CUDA ops were already compiled** in `VoxelKP/build/lib.linux-x86_64-cpython-312/`.
  They just hadn't been copied to the importable locations.
- **Fix applied:** copy all `.so` files from `build/` to `pcdet/ops/*/` and
  export `LD_LIBRARY_PATH` to include the venv torch lib dir.
- **Permanent fix in `~/.bashrc`:**
  ```bash
  export LD_LIBRARY_PATH=/home/thakk100/Projects/thesis/g1_perception_ws/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
  ```
- **Verified working:**
  ```bash
  source .venv/bin/activate
  cd VoxelKP/tools
  python3 -c "from pcdet.config import cfg, cfg_from_yaml_file; cfg_from_yaml_file('cfgs/waymo_models/kp_effv2next4_voxelnext_iou_aug_bev_channel.yaml', cfg); print(cfg.CLASS_NAMES)"
  # → ['Pedestrian', 'Cyclist']
  ```
- **DO NOT re-run `python setup.py develop`** — it has crashed the subagent
  process 5 times. The ops are already compiled; just copy + set `LD_LIBRARY_PATH`.

### Data Inventory
LiDAR recordings at `/home/thakk100/Projects/Thesis/Lidar Data/`:

| Session | Frames | Raw LVX2 | CSV (Livox) | NPZ frames | Ped crops |
|---------|--------|----------|-------------|------------|-----------|
| 2026-08-05_16-38-40 | 998 | ✅ 272MB | ✅ | ✅ | **0** (PP scores < 0.25) |
| 2026-08-05_16-59-33 | 193 | ✅ 52MB | ✅ | ✅ | 406 |
| 2026-08-05_17-00-24 | 257 | ✅ 70MB | ✅ | ✅ | 41 |
| 2026-07-29_17-21-48 | — | ✅ **558MB** | ❌ | ❌ | 0 (not yet processed) |
| 2026-07-29_17-20-14 | — | ✅ 50MB | ❌ | ❌ | 0 (not yet processed) |

July recordings in `/home/thakk100/Downloads/`. Need Livox Viewer or
`convert_lvx2.py` to extract them. **The 558MB July session is the highest-
priority data source — not yet converted.**

**Robot transform (CONFIRMED):**
- Livox Mid-360 is mounted **inverted** on the G1.
- Raw sensor Z points **down** (negative = above sensor, positive = below).
- Display/world convention: `Z_world = −Z_sensor`
- Sensor height above ground: ~1.33 m → after flip, ground is at Z ≈ −1.33 in
  sensor frame, or Z ≈ 0 in world frame.
- VoxelNeXt inference applies `pts[:,2] += 1.33` before the forward pass,
  then `boxes[:,2] -= 1.33` on output. So **NPZ boxes are already in raw
  sensor frame** (Z negative for objects below sensor).

### Mining Script (`mine_reid_crops.py`)
File: `/home/thakk100/Projects/thesis/g1_perception_ws/mine_reid_crops.py`

Produces deterministic (seed=42) person crops for ReID training:
- Crop extraction: bbox-center subtracted, yaw-undone rotation, 256 pts,
  replace=True if < 256 pts.
- Temporal pseudo-label: consecutive frames with person box < 0.8 m apart
  → same `pseudo_id`.

Output at `reid_data/`:
```
crops.npy          (447, 256, 3)  float32  — canonical crops
pseudo_ids.npy     (447,)         int      — -1 = isolated
session_ids.npy    (447,)         int
frame_ids.npy      (447,)         int
temporal_pairs.npy (330, 2)       int      — positive pairs
mining_stats.txt                           — summary
```

**Session 0 (16-38-40) contributed 0 crops** — legacy detector scores all < 0.25.
This session needs VoxelNeXt re-inference to get usable crops. See next steps.

### Person Namer Node
File: `src/livox_detection/livox_detection/person_namer_node.py`
Launch: `src/livox_detection/launch/person_namer.launch.py`

- Subscribes to `/g1/detections/livox` (Detection3DArray).
- Optional `/g1/detections/reid` (PersonReIDArray — see below).
- Keyboard: digit selects person index, Enter assigns name.
- Persists to `~/.ros/person_names.json`.
- Publishes `/g1/person_names` (std_msgs/String, JSON payload).

### LVX2 Converter (`convert_lvx2.py`)
File: `/home/thakk100/Projects/thesis/g1_perception_ws/convert_lvx2.py`

Parses LVX2 binary format without Livox Viewer:
- Z flip applied: `Z_world = −Z_sensor`.
- Outputs NPZ (`xyz`, `refl`, `ts_ns`, `frame_idx`) or CSV.
- Usage: `python3 convert_lvx2.py file.lvx2 --format npz`
- **Validated** on 50MB file: 2.5M points, 18s span, correct coordinates.
- Frame detection is broken (all assigned to frame 0) — timestamp binning
  fix not yet applied. Points are correct; just use `ts_ns` to bin manually.

### ReID Pair Annotator
- **Web artifact** (`reid_annotator.html`): embeds 100 random pairs as JSON,
  shows top/front canvas views. **Cannot see full spatial context** — not
  very useful for annotation decisions, as noted.
- **Local matplotlib tool** (`reid_annotate.py`): shows full LiDAR frame with
  all pedestrian boxes, highlighted person pair. **This is the right tool.**
  Fix required at import time (system mpl_toolkits conflict):
  ```python
  import matplotlib
  sys.path = [p for p in sys.path if 'dist-packages' not in p]
  for k in list(sys.modules.keys()):
      if 'mpl_toolkits' in k: del sys.modules[k]
  ```
  Already applied in the script.

---

## VoxelKP Embedding Head — Design (Not Yet Implemented)

### File to modify
`VoxelKP/pcdet/models/dense_heads/voxelnext_head_kp_merge.py`
Class: `VoxelNeXtHeadKPMerge` (inherits `VoxelNeXtHeadKP` → `VoxelNeXtHead`)

### Feature hook point
```python
# Inside VoxelNeXtHeadKPMerge.forward(data_dict):
x = data_dict['encoded_spconv_tensor']   # SparseConvTensor
# x.features shape: (N_voxels, 384)
# This is where all existing branches (hm, dim, rot, kp_*) read from.
# Add the embed branch here.
```

### Option A — SeparateHead branch (preferred)
Add to YAML config `HEAD_DICT`:
```yaml
'embed': {'out_channels': 128, 'num_conv': 2}
```
`SeparateHead.__init__` builds the branch automatically. Its output
`(N_vox, 128)` is then gathered at detection-centre voxels using the same
`inds` tensor already used by all other branches (see `centernet_utils.gather_feat_idx`).

### Option B — Standalone MLP (simpler for prototyping)
```python
# In VoxelNeXtHeadKP.__init__:
self.embed_head = nn.Sequential(
    nn.Linear(384, 256), nn.BatchNorm1d(256), nn.ReLU(),
    nn.Linear(256, 128)
)
# In forward(), after x = data_dict['encoded_spconv_tensor']:
embed_raw = self.embed_head(x.features)        # (N_vox, 128)
# gather at top-K voxel indices (same inds as cls/reg):
embed_at_centers = gather_feat_idx(embed_raw, inds)   # (B, K, 128)
embed_normed = F.normalize(embed_at_centers, dim=-1)
self.forward_ret_dict['embed'] = embed_normed
data_dict['pred_embed'] = embed_normed          # for inference
```

### Loss
Add to `get_loss()` in `VoxelNeXtHeadKPMerge`:
```python
def get_embed_loss(self):
    embed  = self.forward_ret_dict['embed']           # (B, K, 128)
    labels = self.forward_ret_dict['target_track_ids'] # (B, K) pseudo-ids
    B, K, D = embed.shape
    feats  = embed.view(B*K, D)
    lbl    = labels.view(B*K)
    valid  = lbl >= 0
    return supcon_loss(feats[valid], lbl[valid], temperature=0.07)

# In get_loss():
loss_embed = self.get_embed_loss()
loss += self.model_cfg.get('EMBED_LOSS_WEIGHT', 0.1) * loss_embed
tb_dict['embed_loss'] = loss_embed.item()
```

### Training strategy
- Freeze backbone + all existing heads.
- Train **only** `embed_head` params.
- Use `reid_data/crops.npy` + `temporal_pairs.npy` for SupCon/NT-Xent.
- Alternatively: self-supervised SimCLR on augmented crops (no pseudo-ids
  needed, just two augmented views of each crop).
- Config to copy/modify: `tools/cfgs/waymo_models/kp_effv2next4_voxelnext_iou_aug_bev_channel.yaml`

---

## ROS2 Message: PersonReIDArray (Not Yet Created)

Add to `src/g1_livox_pose_msgs/msg/PersonReIDArray.msg`:
```
std_msgs/Header header
# Parallel to Detection3DArray published at same stamp.
# Index i maps to detections[i] in /g1/detections/livox.
int32[]   detection_indices   # which detections are persons
int32[]   person_db_ids       # LUT key; -1 = new/unassigned
float32[] reid_confidences    # cosine similarity [0,1]
float32[] embed_vectors       # flattened D*N floats (optional, debug)
```

Add `PersonReIDArray` to `src/g1_livox_pose_msgs/CMakeLists.txt` and
`package.xml`, then `colcon build --packages-select g1_livox_pose_msgs`.

The `person_namer_node.py` already handles the `try/except ImportError` for
this message gracefully.

---

## LUT Implementation

```python
# reid_lut.py — runtime, independent of model weights
@dataclass
class PersonEntry:
    embedding: np.ndarray   # (128,) unit-normed
    last_seen: float        # Unix timestamp
    last_pos:  np.ndarray   # (3,) xyz in robot frame
    hit_count: int = 1

class ReIDLUT:
    def __init__(self, cos_thresh=0.82, expire_s=300.0, ema_alpha=0.1):
        self.lut: dict[int, PersonEntry] = {}
        self.next_id = 0
        ...

    def match_or_register(self, embed, pos) -> int:
        # cosine similarity → if max_sim >= cos_thresh, update + return existing id
        # else register new entry, return new id

    def prune(self):
        # remove entries older than expire_s

    def save(self, path):  # pickle or npz for persistence across restarts
    def load(self, path):
```

**Persist LUT** with `pickle` or `numpy .npz` between robot runs so previously
named persons survive a restart.

---

## ReID Literature References

| Method | Key idea | Applicable? |
|--------|----------|-------------|
| **PointContrast** (Xie et al., ECCV 2020) | Contrastive learning on 3D point features, no labels | ✅ Direct inspiration |
| **SupCon** (Khosla et al., NeurIPS 2020) | Supervised contrastive with pseudo-labels | ✅ Use temporal track IDs |
| **SimCLR** (Chen et al., ICML 2020) | NT-Xent on two augmented views, no labels | ✅ Works with augmented crops, zero labels |
| **LPT (Li et al., IROS 2021)** | LiDAR-specific pedestrian tracking + reID | ✅ Architecture reference |
| **BotSORT / StrongSORT** | IoU+Kalman in-frame + embedding for re-entry | ✅ Tracker to combine with ReID LUT |
| **LiDARHuman26M (Fan et al., CVPR 2023)** | LiDAR human pose + motion dataset | 📚 Reference dataset |
| **SLOPER4D (Dai et al., CVPR 2023)** | LiDAR-based human pose in large outdoor scenes | 📚 Reference dataset |

**Chosen approach:** SupCon on temporal pseudo-labels (track IDs from
consecutive-frame IoU matching) + SimCLR augmentation. No new data collection
needed — the 447 existing crops (330 positive pairs) are sufficient for a
prototype. Session-0 data (998 frames) needs VoxelNeXt re-inference to unlock
more crops.

---

## What Worked

- Copying pre-compiled `.so` files fixes VoxelKP instantly without recompile.
- `LD_LIBRARY_PATH` pointing to venv torch lib resolves `libc10.so` errors.
- Mining temporal pairs from consecutive-frame bbox proximity works well
  (86% of crops assigned a pseudo-id).
- `sys.path` stripping of `dist-packages` before importing `mpl_toolkits`
  fixes the system/venv matplotlib conflict.
- LVX2 binary parser: packets are 1371 bytes (27-byte header + 96×14 pts).
  First packet locatable by scanning for IP marker `c0:a8:7b:78`.
- Z flip (`Z_world = −Z_raw`) gives correct world-frame coordinates for
  inverted Mid-360 mount.

## What Didn't Work

- **Subagent for VoxelKP build** crashed the process 5× — CUDA compilation
  in a subagent hits memory/time limits. Never attempt `python setup.py develop`
  in a subagent. Always handle build operations in the main process via Bash.
- **Web artifact for point cloud annotation** — isolated crops without spatial
  context aren't enough to judge same/different person. Use the local
  matplotlib tool (`reid_annotate.py`) instead.
- **LVX2 frame detection** from timestamp jumps — Mid-360 timestamps are
  continuous across frame boundaries; the `frame_dur_ns` jump threshold does
  not trigger. Use `ts_ns` binning manually: `frame_idx = (ts_ns - ts_ns[0]) // frame_dur_ns`.

---

## Next Steps (Priority Order)

### 1. Convert July 558MB LVX2 and re-infer with VoxelNeXt [HIGH]
The user has Livox Viewer — convert:
- `/home/thakk100/Downloads/2026-07-29_17-21-48.lvx2` (558MB, ~400 frames est.)
- `/home/thakk100/Downloads/2026-07-29_17-20-14.lvx2` (50MB)

Then run VoxelNeXt inference on the resulting CSV/NPZ to get per-frame
`pred_boxes` + `pred_labels` (same format as the legacy `*_frames_*.npz` outputs).

Also re-run VoxelNeXt on session `2026-08-05_16-38-40` (998 frames, the legacy
detector gave 0 crops — VoxelNeXt should score better). Expected total crops: 3000–6000.

### 2. Add embed branch to VoxelNeXtHeadKPMerge [HIGH]
File: `VoxelKP/pcdet/models/dense_heads/voxelnext_head_kp_merge.py`

Use Option B (standalone MLP) for simplicity. Steps:
1. Add `self.embed_head` MLP in `__init__`
2. Gather embed at detection centers in `forward()`
3. Store in `data_dict['pred_embed']` for inference path
4. Add `get_embed_loss()` + wire into `get_loss()`

Config: copy `kp_effv2next4_voxelnext_iou_aug_bev_channel.yaml` → add
`EMBED_DIM: 128` and `EMBED_LOSS_WEIGHT: 0.1`.

### 3. Write training script [HIGH]
`VoxelKP/tools/train_embed.py` — minimal loop:
- Load mining outputs from `reid_data/`
- Forward pass with frozen backbone+heads
- Compute SupCon loss on embed branch only
- Save embed head weights separately from full checkpoint

### 4. Add PersonReIDArray.msg [MEDIUM]
File: `src/g1_livox_pose_msgs/msg/PersonReIDArray.msg`
Update `CMakeLists.txt` + `package.xml`.
Rebuild: `colcon build --packages-select g1_livox_pose_msgs`

### 5. Wire ReIDLUT into livox_detection_node [MEDIUM]
In `voxelnext_model.py`:
- `infer()` returns `(boxes, scores, labels, embeds)` — embeds shape `(M, 128)`

In `livox_detection_node.py`:
- Instantiate `ReIDLUT` on startup
- After each inference: `match_or_register()` per pedestrian detection
- Build + publish `PersonReIDArray` on `/g1/detections/reid`

### 6. Annotate pairs and validate embedding quality [MEDIUM]
Run `python3 reid_annotate.py` — annotate ~100 pairs from `reid_data/`.
Check: what fraction of tracker-assigned "same person" pairs are actually
the same person visually? If < 80%, the pseudo-labels are noisy and SimCLR
(label-free augmentation) is preferable to SupCon.

### 7. Fix LVX2 frame detection [LOW]
In `convert_lvx2.py`, replace timestamp-jump detection with binning:
```python
frame_idx_arr = ((ts_ns_arr - ts_ns_arr[0]) // frame_dur_ns).astype(np.int32)
```

---

## File Map (All New Files This Session)

```
g1_perception_ws/
├── mine_reid_crops.py                          ← data mining script
├── reid_annotate.py                            ← local matplotlib annotator
├── convert_lvx2.py                             ← LVX2 binary → NPZ/CSV
├── reid_data/
│   ├── crops.npy           (447, 256, 3)
│   ├── pseudo_ids.npy      (447,)
│   ├── session_ids.npy     (447,)
│   ├── frame_ids.npy       (447,)
│   ├── temporal_pairs.npy  (330, 2)
│   └── mining_stats.txt
└── src/
    ├── livox_detection/livox_detection/
    │   └── person_namer_node.py               ← new ROS2 naming node
    ├── livox_detection/launch/
    │   └── person_namer.launch.py             ← launch file
    └── g1_livox_pose_msgs/msg/
        └── PersonReIDArray.msg                ← NOT YET CREATED (design only)
```

**NOT YET IMPLEMENTED** (design + plan only):
- `VoxelKP/pcdet/models/dense_heads/voxelnext_head_kp_merge.py` embed branch
- `VoxelKP/tools/train_embed.py` training script
- `reid_lut.py` runtime LUT
- VoxelNeXt embed output wired into ROS2 pipeline

---

## Environment

- Python: 3.12, venv at `.venv/`
- PyTorch: 2.3.1+cu121, CUDA available ✅
- pcdet: 0.6.0 (from VoxelKP), importable ✅
- spconv: importable ✅
- CUDA ops: compiled + copied, `LD_LIBRARY_PATH` set ✅
- VoxelKP config: `VoxelKP/tools/cfgs/waymo_models/kp_effv2next4_voxelnext_iou_aug_bev_channel.yaml`
- VoxelKP checkpoint: **not found in repo** — check `~/` or ask user for path
  before training; inference-only path doesn't need it for the embed head
  (can load existing detection checkpoint + newly trained embed weights separately)
