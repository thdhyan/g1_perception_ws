# HANDOFF: ReID — model-vs-tracker finding + verified status (v2)

**Date:** 2026-09-01 (amends aug29_handoff.md ReID sections)
**Author:** thakk100 (diagnostics by opencode agent)
**Workspace:** ~/Projects/thesis/g1_perception_ws

This document supersedes the ReID status in `aug29_handoff.md` sections R4/R5.
It records a *corrected, verified* understanding of the ReID model and tracker,
plus the exact remaining blockers.

---

## TL;DR — three things changed since the last handoff

1. **`model_identity.pt` is GOOD, not "untested/weak".** Verified **96% rank-1**
   (724/755) on the 757 clean July crops (P_A 278 / P_B 479). The handoff's
   "val_rank1 ≈ 0.10 ≈ chance" number belonged to a *different* model (`model.pt`).
   See §1.
2. **`reid_embed_server.py` was pointed at the wrong model.** It defaulted to
   `model.pt` (243-class, trained on temporal-proximity pseudo-labels that encode
   NO identity). Now fixed to `model_identity.pt`. Track over-split dropped
   **9 → 5**, matching the "real ≈ 3–5 people" expectation. See §2.
3. **Tracker purity cannot be validated yet** — the identity map is *per-frame*,
   but tracking truth needs *per-detection* labels. This is the real remaining
   blocker (not the model). See §3.

---

## 1. The model is solved (for 2 people)

| Metric (on 757 clean crops) | Value |
|---|---|
| intra-person cosine (same person) | mean **0.953**, median 0.973 |
| inter-person cosine (diff people) | mean **0.610**, median 0.604 |
| best same/diff separation | **91.1%** @ cos-thresh 0.88 (chance 0.50) |
| **rank-1 (P_A vs P_B)** | **724/755 = 0.959** (chance 0.50) |

**Two different models exist and were conflated before:**

| Checkpoint | Classes | Trained on | Quality |
|---|---|---|---|
| `reid_data/model.pt` | **243** | `crops.npy` + `pseudo_ids` from *temporal proximity* (<0.8 m across frames) | **weak** (rank1≈0.10) — learns "near last frame", NOT identity |
| `reid_data/model_identity.pt` | **2** | `identity_crops.npy` (hand-labeled clean segments) | **good** (rank1≈0.96) |

**Why `model.pt` was always doomed:** its labels are a *tracking* signal, not an
*identity* signal. A model trained on "is this box near the previous frame's box"
cannot re-ID across a 300 s gap or across sessions — but that is exactly the task.

**Verified:** the ReID embedding is produced *entirely* by the PointNet encoder;
the classifier head contributes zero to `emb` (encoder-only vs full-model max diff
= 0.0). So `model_identity.pt`'s 2-class checkpoint yields perfectly usable
128-d embeddings regardless of its classifier size.

---

## 2. `reid_embed_server.py` — what was fixed

| Line / arg | Before | After |
|---|---|---|
| `--model` default | `reid_data/model.pt` (243-class) | **`reid_data/model_identity.pt`** |
| `--cos-thresh` default | 0.55 | **0.75** (0.88 = 91% sep; 0.75 = safer operating point) |
| `--min-len` default | 3 | **5** |
| embedding cache | keyed by session only → could silently reuse wrong-model embeds | **keyed by session + model basename** (`emb_<session>_<model>_*.npy`), plus a cache-write path added so first run persists and later runs skip recompute |

**Result (July, min-score 0.4):** track count **9 → 5**, stable across
`cos_thresh` 0.55–0.85. This confirms the over-split was *model-driven*, not
threshold-driven.

**Restart command (now correct):**
```bash
cd ~/Projects/thesis/g1_perception_ws && . .venv/bin/activate
setsid nohup python3 -u reid_embed_server.py --session 2026-07-29_17-21-48 \
  --max-range 5.0 --short-gap 5 --cos-thresh 0.75 --pos-gate 2.0 \
  --pos-weight 0.5 --min-len 5 --port 8767 >/tmp/reid_8767.log 2>&1 &
```
First run recomputes embeddings (~40 s CPU for 1,193 dets @ min-score 0.4) and
caches to `reid_data/emb_2026-07-29_17-21-48_model_identity_*.npy`. Later runs
hit the cache. To force recompute: add `--recompute`.

**Do NOT lower `min-score` to densify tracks.** At 0.15 the tracker produced
**19 tracks** with a B-collapse (worse, not better). Denser-but-noisier low-score
boxes feed the tracker confusion. Default 0.4 is fine.

---

## 3. The real remaining blocker — per-detection labels

The tracker cannot be *validated* (purity / ID-switch / re-entry recall) with the
current data:

- `identity_map_2026-07-29_17-21-48.json` marks **frame ranges** where a person is
  "cleanly tracked", but a single frame can contain ~4 simultaneous detections, so
  "P_A is in frames 93–1082" cannot tell us *which* detection at frame 400 is P_A.
- Tracking truth is **per-detection** (or at minimum per-(slot, frame) with a
  nearest-box tie-break), which the current map does not provide.
- Open segments in the map (still TODO): T1@112–631, T2@1083+, T4, T5.
- Pair-annotator verdicts: only ~15 in `sameperson_*.json` (219 pairs built).

**Implication:** do not tune the tracker further until per-detection labels exist.
Tuning against a coarse per-frame map is chasing a metric that is itself ill-posed.

---

## 4. LReID dataset — downloaded and decoded

**Location:** `/generalSSD/lreid/LReID.zip` (188 MB, complete: 197,591,587 bytes).
Root disk (`/`) is at 100% (3 GB free) — do NOT put new data/artifacts there.
`/generalSSD` (247 GB, ~88 GB free) is the data disk.

**Contents (verified):**
- `LReID/bbox_train/` — 102,060 `.bin` person-crops, **128 pts** each, float32 `[x,y,z,int]`.
- `LReID/bbox_test/` — 11,820 `.bin`, **256 pts** each.
- `LReID/info/train_name.txt` / `test_name.txt` — file lists.
- `LReID/info/tracks_train_info.txt` / `tracks_test_info.txt` — `<frame_start> <frame_end> <person_id> <tracklet_idx>` → 30-frame gait sequences.
- `LReID/info/query_IDX.txt` — query indices.
- Naming: `0000C1T0000F000.bin` = ID 0000, Camera 1, Tracklet 0, Frame 0.
- Captured with Livox Mid-100 @ 10 Hz (same family / rate as our Mid-360).

**Use:** pre-train our PointNet shape encoder on LReID-sync style tasks
(point-cloud completion / shape-param regression) before fine-tuning on our own
4 sessions. This is the ReID3D recipe, and it directly attacks the "single-view
sparsity → weak shape prior" weakness.

---

## 5. Biometric ("intrinsic parameter") features — measured

Computed per detection from raw points (de-yawed, box-centred), P_A vs P_B on the
clean July crops:

| Feature | P_A mean±std | P_B mean±std | best acc (chance 0.50) |
|---|---|---|---|
| height (raw, 95th-5th pct) | 1.28±0.12 | 1.27±0.07 | 0.69 |
| detector box_dz | 1.73±0.05 | 1.64±0.12 | 0.66 (t=14, contaminated) |
| depth (de-yawed dy) | 0.62±0.10 | 0.58±0.09 | 0.61 |
| width (de-yawed dx) | 0.54±0.12 | 0.55±0.11 | 0.53 |
| point count | 536±206 | 509±146 | 0.58 |

**Verdict:** P_A and P_B are similar height (~8 cm apart), so no single-frame
biometric cue separates them reliably (best 0.69). Biometrics are useful as a
*combined prior* fused with the embedding, NOT as the identity carrier. The
learned embedding is strictly stronger (0.96 vs 0.69).

---

## 6. Forward methodology (ordered)

1. **Per-detection labeling helper** — resolve the §3 blocker. A tool that, given
   the 7-slot tracker + identity map, flags exactly which (slot, frame) are
   ambiguous so the operator resolves only those. *(cost: 1 build + ~1h human)*
2. **N-person generalization** — label P_C/P_D (the 4 open segments), retrain the
   encoder on N classes. The model is currently 2-class-only; real scenes have 3–5.
3. **LReID pre-training** — shape-completion/param regression on the §4 data
   before N-person fine-tune. *(1–2 days, on `dl`/spark, not the laptop)*
4. **Fuse biometric + embedding + geometry** in the on-robot tracker
   (`reid_tracks_server.py` → Hungarian with
   `0.7·disp/gate + 0.1·gap/max_bridge + 0.2·(1−cos)`), then validate purity /
   ID-switch / re-entry recall against per-detection labels.

**Pose workstream (VoxelKP/DAPT) is unrelated to ReID** — do not spend sptr/DAPT
effort for identity; poses add little identity signal and the pose backend is
still the debug template. Leave it as a separate thesis contribution.

---

## 7. Files of note (new / changed this session)

| File | What |
|---|---|
| `reid_embed_server.py` | FIXED defaults (§2): model, cos-thresh, min-len, model-aware cache |
| `scratch/eval_reid_biometric.py` | runs the §1 + §5 diagnostics (in-venv) |
| `scratch/sweep_reid_cos_thresh.py` | headless cos_thresh sweep using `reid_track()` |
| `/generalSSD/lreid/LReID.zip` | LReID dataset (188 MB, §4) |
| `reid_data/model_identity.pt` | the GOOD model (§1) |
| `reid_data/model.pt` | the WEAK model (§1) — do not use for identity |

**Verified commands:**
```bash
# re-run the model diagnostic
.venv/bin/python scratch/eval_reid_biometric.py       # needs /tmp/tracks_8766.json for §5
# headless tracker sweep
.venv/bin/python scratch/sweep_reid_cos_thresh.py --min-score 0.4
```