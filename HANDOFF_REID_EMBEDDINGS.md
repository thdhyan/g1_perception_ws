# Handoff — from position-based tracking to TRUE ReID (learned embeddings)

Status snapshot, 2026-08-28. This doc is the plan for the next milestone:
replace (or augment) the geometric slot-assignment tracker with a real
ReID-style appearance/gait embedding so identity survives crossings, long
absences, and *across sessions*.

---

## 1. Where we are (what works today)

**Pipeline (all in `~/Projects/thesis/g1_perception_ws`):**

```
LVX2 (Mid-360 @10Hz)
  → convert_lvx2.py            # LVX2 → per-frame .npy  [x,y,z,intensity] float32, z-up
  → split_frames.py            # session → frames/<session>/frame_00000.npy …
  → reinfer_voxelnext.py       # VoxelNeXt (nuScenes ckpt, pt/voxelnext_nuscenes.pth)
  → <session>_frames_voxelnext.npz
       pred_boxes  (N, K, 7)  [x, y, z, dx, dy, dz, yaw]   z = floor (≈ 0 … −1.2 m bias)
       pred_labels (N, K)     1-indexed nuScenes class; PED = 2
       pred_scores (N, K)
       frame_files (N,)       "frame_00000.npy"
```

**Tools running:**

| Tool | Port | What it does |
|---|---|---|
| Pair annotator (`reid_web_server.py` + `reid_web/`) | 8765 | Side-by-side two-frame pedestrian pair, "same / different" verdicts → `reid_data/sameperson_<session>.json` (219 pairs, 15 verdicts so far) |
| Trajectory viewer (`reid_tracks_server.py` + `reid_tracks/`) | 8766 | 7 fixed identity slots over the whole session, live point-cloud playback, click-to-select, absence dropdown |

**Current July session result (2026-07-29_17-21-48, 2051 frames, 14,405 ped boxes,
params: `--people 7 --min-score 0.4 --max-speed 4.0 --gap-cost 0.05 --max-bridge 9`):**

- T1, T2: ≈50–70% presence — **plausible people** (user visually confirmed T1 ≈ good)
- T3: ≈13% presence — sparse, suspect
- T4, T5, T6: 1/8/11 obs — **junk slots** (greedy assignment gave transient boxes a slot)
- T1 has **21 absences totalling 867 frames (86.7 s of missing)** — those long spans are
  where position tracking is blind and ReID must take over
- 0 rejected "jumps", 0 clutter dropped

Restart commands (from the repo dir, after `. .venv/bin/activate`):

```bash
python3 -u reid_web_server.py --session 2026-07-29_17-21-48 \
  --lidar-dir ~/Projects/Thesis/Lidar\ Data \
  --min-score 0.5 --max-dist 0.8 --size-ratio 1.5 --gap-max 1 --port 8765

python3 -u reid_tracks_server.py --session 2026-07-29_17-21-48 \
  --people 7 --min-score 0.4 --max-speed 4.0 --port 8766
```

## 2. Why position-based tracking is at its ceiling

1. **Crossings/occlusions**: two people pass within ≤0.8 m → pure xy cost can't tell them
   apart; identity can swap silently (that is the T1 "sudden jump" the user still sees:
   the slot hops to a *different person* who happened to pass through).
2. **Long absences**: after an 87-frame gap, the nearest-box association is a coin flip —
   the current code just draws a faint bridge and hopes.
3. **Junk slots**: greedy "leftover boxes take a free slot" lets clutter steal identities
   (T4–T6).
4. **No cross-session identity**: "Alice in July" and "Alice in August" are unrelated IDs.
   True ReID = same embedding across sessions. Position tracking can't do this at all.
5. Scores from the detector are unhelpful (median ped score 0.187, p95 0.451) — we have no
   reliable per-box likelihood to rank candidate bindings.

**What real ReID adds**: an appearance/gait *embedding* per observation, so the binding
cost includes `1 − cos(emb_obs, emb_identity)`. Identity becomes content, not geometry.

## 3. What to research (in priority order)

1. **ReID3D** (arXiv **2312.03033**, CVPR 2024, code `GWxuan/ReID3D`) — the only
   pure-LiDAR person ReID paper we know of: fuses **3D shape + gait**, 94.0 rank-1 on LReID.
   Their preprocessing is the blueprint: points **box-centred**, sequence of 32 frames
   (~3.2 s at our 10 Hz), fixed point budget, shape branch + temporal/gait branch.
   *Steal the protocol, write our own smaller model.*
2. **LReID (+ LReID-Sync)** — the standard LiDAR person ReID benchmark. Read the eval
   protocol (gallery/candidate, rank-1, mAP), but note it's multi-vehicle urban traffic —
   **not** our indoor single-robot scene.
3. **FreeGait** (ECCV 2022) and **Gait3D** — gait-centric 3D ReID. Relevant *if gait is a
   cue here*; caveat: our people often **stand still** (≤7 unscripted people, one robot),
   so gait may carry less signal than shape + box size + height.
4. **SUSTech1K** — RGB person ReID; useful only as a reference for loss design
   (batch-hard triplet, ArcFace) with the class structure of "a handful of people".
5. **SORT / DeepSORT (3D variants)** — the canonical upgrade path for the *assignment*
   itself: replace our two-phase greedy with **cost matrix + Hungarian (linear_sum_assignment)**
   where cost = distance + gap + reid-term.
6. ⚠️ **arXiv 2506.04499 (FALO, Qualcomm)** is a 3D object **detection** paper — verified
   **not** ReID. Do not cite it as ReID.
7. **Benchmark reality check** (subagent-verified): **no public set matches "single robot,
   ≤7 unscripted people, indoor"** → *our own sessions are the benchmark*. Define our
   metrics and log them per run (see §6.5).

## 4. Implementation plan (stages; each stage shippable on its own)

### Stage 0 — Identity ground truth (human, do first, cheap)
- The 7-slot output is a **pseudo-label**. Verify it in the viewer (8766) and in the pair
  annotator (8765): which slot is which human? Mark each slot P1…P7 in a new file
  `reid_data/identity_map_<session>.json` (slot → human name).
- In the pair annotator, label at least the **hard pairs**: (a) different slots, same
  frame, (b) re-appearances after long absences (the absence dropdown on 8766 gives the
  exact frame numbers to seek to). Those become **manual hard negatives/positives**.
- Deliverable: per-observation `true_id` for every ped box in both sessions (≤ 2 × 14k
  boxes; slots T1–T3 carry most of it).

### Stage 1 — Crop extraction (`extract_crops.py`, new file in repo)
For every ped box (score ≥ 0.4):
1. Load the frame npy (`frames/<session>/frame_NNNNN.npy`).
2. **Filter (x==0 & y==0) points** — ~35% of Mid-360 returns are "no-return" placeholders;
   they poison crops.
3. Keep points inside the box **dilated by 0.15 m** (use dx, dy, dz; ignore yaw for
   simplicity — boxes are near-upright).
4. **Center**: `p − box_center` where `box_center = (x, y, dz/2)` (rest-on-ground
   convention; do NOT use raw box z, it has the −1.2 m floor bias).
5. **Normalize**: divide by `(dx, dy, dz)` → scale-invariant shape (ReID3D-style).
6. **Resample to 256 points** (farthest-point sampling or uniform stride + pad).
7. Save npz: per obs `{session, frame, box7, points256x3 float16, slot_id, min_score}` and a
   **temporal window**: ±15 frames of the *same slot* (31 × 256 pts) for the gait branch.

Sizes: 14,405 obs × 256 × 3 × 2 B ≈ **22 MB** (single) / ~700 MB (31-frame windows) —
disk is at **98%**, so store windows on `~` only if `df -h` shows ≥15 GB free; otherwise
store single-frame crops (shape branch) first and add windows later on `dl`.

### Stage 2 — Backbone (PointNet, ~100 lines of PyTorch, runs in `.venv`)
```
input: (B, 256, 3)
shared MLP: 3→32→64→128 (FC+BN+ReLU ×3)
max-pool over points → 128
FC: 128→256 → BN → ReLU → FC: 256→128 → L2-normalise → e_shape (128,)
```
- **Gait branch (ReID3D-style, second iteration)**: per-frame embedding over the 31-frame
  window → GRU or simple mean-pool → `e_gait`; final `e = concat/mean(e_shape, e_gait)`.
- PointNet has **no install step** — plain `torch`, no VoxelKP (`setup.py develop` is
  forbidden in subagents for that reason; a standalone script avoids it entirely).
- Training compute: batch 64–128, <2 M params → **minutes on CPU or 1 spark GPU**; `dl`
  if we want speed. (Laptop is client-only per workspace rules.)

### Stage 3 — Training data & loss
- **Positives**: same `slot_id` (stage-0 verified) across frames/sessions.
- **Hard negatives**: (a) different slots, **same frame** (guaranteed different people —
  the pair annotator's hard pairs are exactly these), (b) cross-session unknowns once
  labeled.
- **Loss**: `batch-hard triplet, margin 0.3` + `cross-entropy` over slot_id (we *do* know
  class structure, unlike street ReID). Add L2 norm on embeddings.
- Freeze backbone 3 epochs, last-layer only → then unfreeze. Weight decay 1e-4.
- Data volume: ~14k crops ≈ hundreds of (A,P,N) triples per epoch — plenty for 128-d.
- Train/val split **by session** (July train → August val) — that is the real cross-session
  ReID test.

### Stage 4 — Integration (`reid_tracks_server.py`)
1. Precompute `embeddings.npy`: shape `(n_obs, 128)` aligned to a new obs-index column in
   the npz (or a side-file `reid_data/emb_<session>.npy`).
2. Add **per-slot running anchor** `A_si = EMA of last K bound embeddings` (K ≈ 10, α 0.2).
3. New binding cost:
   ```
   cost(si, b) = 0.7 * (disp_xy / gate)
               + 0.1 * (gap_frames / max_bridge)      # current gap-cost 0.05 → rescaled
               + 0.2 * (1 − cos(emb_b, A_si))          # ReID term
   ```
   Tune the 0.7/0.1/0.2 weights against the T1 "jump" case (want the crossing person to LOSE).
4. Replace two-phase greedy with **Hungarian**: cost matrix (slots × boxes) + a free-slot
   pseudo-column (cost = 1 − cos against the *newest* person's anchor, penalised) →
   `scipy.optimize.linear_sum_assignment` (already in the venv). This kills the T4–T6 junk
   slots because junk observations have low similarity to every anchor.
5. **Re-ID after long absences**: when `gap > max_bridge`, currently the box is dropped.
   With embeddings, allow re-binding to a *recently-used* slot if `cos > 0.6`, else new slot.
   This is the feature that fixes the 867-frame T1 spans (if T1 reappears = T1 again).
6. Server API: expose `emb_sim` on each bound observation → show it in the table (a new
   `sim` column) and in the absence dropdown chips, so we can *see* which re-bindings the
   embedding accepted/rejected.

### Stage 5 — Evaluation (our own benchmark, §3.7)
Per session log to `reid_data/metrics_<session>.json`:
- **ID switch rate**: number of times a human (stage-0 map) changes slot
- **Track purity**: fraction of a slot's boxes belonging to its labeled human
- **Re-binding recall**: after gaps ≥ 10 frames, % correctly re-associated to same human
- **Junk-slot count** (should drop to ~0 with Hungarian + ReID term)
A/B: current position-only pipeline vs +embedding, same 219 pairs, same viewer.

## 5. Risks / gotchas (all learned the hard way in this repo)
- **Livox clock quirk**: 2026-07-29 session had two timestamp epochs → renumbering Stream B
  564–2050 → 2051. **Any new session: check for duplicate/non-monotonic timestamps first.**
- **~35% of Mid-360 points are (0,0,0) "no-return" placeholders** — keep for context, but
  EXCLUDE from crops/ReID (they are not geometry).
- **Detector z bias**: box z ≈ floor (−1.2 m) — use `z = h/2` for display; matching is
  **xy-only**, never 3D.
- **Disk is at 98%** (`/dev/nvme0n1p5`) — check `df -h` before writing >1 GB artifacts;
  prefer `dl` (storage rule: skip sparks with <500 GB free).
- **`python3 setup.py develop` in a subagent = crash** (workspace rule) — keep ReID work in
  standalone PyTorch scripts, not VoxelKP/VoxelNext repos.
- VoxelNeXt checkpoint is nuScenes-10-class → ped scores are low (median 0.187). Do not
  "fix" by lowering min-score further; the ReID term is the right answer to weak scores.

## 6. Immediate next actions (ordered)
1. [user] Label the slot→human identity map + ~30 hard pairs while looking at the viewer.
2. [agent] `extract_crops.py` (Stage 1) — single-frame crops first, run on July session,
   visualise a few crops (there is a `pointcloud-viz` skill).
3. [agent] PointNet + triplet/CE trainer (Stage 2–3) on `.venv` torch; train July→val Aug.
4. [agent] Hungarian + ReID-cost in `reid_tracks_server.py` (Stage 4) + `sim` column.
5. [both] A/B metrics (Stage 5), then decide if a gait branch (31-frame windows) is needed.
6. [research] Read ReID3D §data-prep + §backbone, LReID eval protocol (~2 h, only what maps
   to our stages; skip street-traffic ablations).

## 7. File map

| Path | Purpose |
|---|---|
| `reid_tracks_server.py` | 7-slot tracker server (:8766), two-phase greedy assignment — **Stage-4 target** |
| `reid_tracks/{index.html,app.js}` | viewer (click-select, gaps dropdown, live playback) |
| `reid_web_server.py`, `reid_web/` | pair annotator (:8765), verdicts → `reid_data/sameperson_*.json` |
| `reid_data/` | verdicts; **add**: `identity_map_<session>.json` (Stage 0), `emb_<session>.npy` (Stage 4), `metrics_<session>.json` (Stage 5) |
| `extract_crops.py` | **new** (Stage 1) |
| `reid_model.py`, `train_reid.py` | **new** (Stage 2–3) |
| `reinfer_voxelnext.py`, `split_frames.py`, `convert_lvx2.py` | data pipeline (stable) |
| `VoxelNeXt/` | detector repo (already patched `_topk_1d`) — do not modify for ReID |
| `pt/voxelnext_nuscenes.pth` | 10-class nuScenes checkpoint |
| `~/Projects/Thesis/Lidar Data/frames/<session>/frame_NNNNN.npy` | per-frame clouds |
| `~/Projects/Thesis/Lidar Data/<session>_frames_voxelnext.npz` | detections |
| `HANDOFF_REID.md` | previous pipeline handoff (data/inference) |
| `AGENTS.md` | workspace rules (username `thakk100`, `~/Projects/Thesis` capital T) |

---
*ReID research brief (subagent, complete): LReID, FreeGait, SUSTech1K, Gait3D, Spb3DTracker
compared; verdict — position+box is SOTA-correct for ≤7 people / short gaps, learned
embedding needed only for long-occlusion and cross-session identity. That is exactly this
plan.*
