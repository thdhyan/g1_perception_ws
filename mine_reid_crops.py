"""
mine_reid_crops.py — Mine person crops from LiDAR recordings for ReID training.

Outputs to reid_data/:
  crops.npy         (N_crops, 256, 3)  float32 — normalised+rotated xyz
  pseudo_ids.npy    (N_crops,)         int     — track id, -1 if isolated
  session_ids.npy   (N_crops,)         int     — session index 0/1/2
  frame_ids.npy     (N_crops,)         int     — frame index within session
  temporal_pairs.npy (M, 2)            int     — positive pairs (same person)
  mining_stats.txt                     text    — summary statistics
"""

import os
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
BASE_FRAMES = "/home/thakk100/Projects/Thesis/Lidar Data/frames/"
BASE_LABELS = "/home/thakk100/Projects/Thesis/Lidar Data/"
OUT_DIR     = "/home/thakk100/Projects/thesis/g1_perception_ws/reid_data/"

SESSIONS = [
    "2026-07-29_17-21-48",   # July — 14k ped boxes, primary training session
    "2026-08-05_16-59-33",
    "2026-08-05_17-00-24",
]

PED_LABEL   = 2
SCORE_THRESH = 0.25
N_PTS       = 256
MATCH_DIST  = 0.8   # metres — 2-D centre distance for consecutive-frame matching

os.makedirs(OUT_DIR, exist_ok=True)

# Deterministic RNG — seed once so re-runs give identical crops.
rng = np.random.default_rng(42)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_crop(pts_xyz: np.ndarray, box7: np.ndarray) -> np.ndarray:
    """Return (N_PTS, 3) float32 crop in canonical (centred + de-yawed) frame."""
    cx, cy, cz, dx, dy, dz, yaw = box7.astype(float)

    # 0. Remove Livox no-return placeholders (x==0 & y==0 — ~35% of points).
    valid = ~((pts_xyz[:, 0] == 0.0) & (pts_xyz[:, 1] == 0.0))
    pts_xyz = pts_xyz[valid]

    # 1. Translate so bbox centre is origin.
    pts = pts_xyz - np.array([cx, cy, cz], dtype=float)

    # 2. Rotate by -yaw around Z — undo LiDAR heading.
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0., 0., 1.0]])
    pts = (R @ pts.T).T                          # (N, 3)

    # 3. Filter points inside the rotated bounding box.
    mask = ((np.abs(pts[:, 0]) <= dx / 2.0) &
            (np.abs(pts[:, 1]) <= dy / 2.0) &
            (np.abs(pts[:, 2]) <= dz / 2.0))
    pts_in = pts[mask]

    # 4. Sample N_PTS points (replace=True handles sparse crops).
    n = len(pts_in)
    if n == 0:
        return np.zeros((N_PTS, 3), dtype=np.float32)
    idx = rng.choice(n, N_PTS, replace=True)
    return pts_in[idx].astype(np.float32)


def greedy_match(boxes_a: np.ndarray, boxes_b: np.ndarray) -> dict:
    """Greedy nearest-neighbour matching by 2-D (x,y) centre distance < MATCH_DIST.

    Returns dict {idx_in_a: idx_in_b} for matched pairs.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return {}
    ca = boxes_a[:, :2]   # (J, 2)
    cb = boxes_b[:, :2]   # (K, 2)
    # (J, K) distance matrix
    dists = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
    matched_a, matched_b = set(), set()
    result = {}
    # Sort all (dist, i, j) pairs ascending — greedy closest-first.
    pairs = sorted(
        ((dists[i, j], i, j) for i in range(len(boxes_a)) for j in range(len(boxes_b))),
        key=lambda t: t[0]
    )
    for dist, i, j in pairs:
        if dist >= MATCH_DIST:
            break
        if i not in matched_a and j not in matched_b:
            result[i] = j
            matched_a.add(i)
            matched_b.add(j)
    return result


# ── Per-session processing ────────────────────────────────────────────────────

def process_session(sess_idx: int, sess_name: str):
    """Return (crops, pseudo_ids, session_ids, frame_ids, temporal_pairs) arrays."""
    # per-frame detection labels: <sess>_frames_<backend>.npz (prefer _voxelnext)
    _matches = sorted(f for f in os.listdir(BASE_LABELS)
                      if f.startswith(f"{sess_name}_frames_") and f.endswith(".npz"))
    _pref = [f for f in _matches if f.endswith("_voxelnext.npz")]
    if not _matches:
        raise FileNotFoundError(f"no label file {sess_name}_frames_*.npz in {BASE_LABELS}")
    label_path = os.path.join(BASE_LABELS, (_pref or _matches)[0])
    frames_dir = os.path.join(BASE_FRAMES, sess_name)

    data = np.load(label_path, allow_pickle=True)
    frame_files  = data["frame_files"]     # (F,) str
    pred_boxes   = data["pred_boxes"]      # (F,) object → each (Ki, 7)
    pred_labels  = data["pred_labels"]     # (F,) object → each (Ki,)
    pred_scores  = data["pred_scores"]     # (F,) object → each (Ki,)

    n_frames = len(frame_files)

    # ── Step 1: collect per-frame ped detections ─────────────────────────────
    # ped_frames[i] = (frame_idx, boxes (K,7)) — only score-filtered peds
    ped_frames = []
    for fi in range(n_frames):
        boxes  = pred_boxes[fi]
        labels = pred_labels[fi]
        scores = pred_scores[fi]
        if len(labels) == 0:
            ped_frames.append((fi, np.empty((0, 7), dtype=np.float32)))
            continue
        mask = (labels == PED_LABEL) & (scores >= SCORE_THRESH)
        ped_frames.append((fi, boxes[mask].astype(np.float32)))

    # ── Step 2: temporal matching — assign provisional pseudo_ids ─────────────
    # Forward pass: each new unmatched detection starts a new track id.
    n_tracks = 0
    prev_boxes     = np.empty((0, 7), dtype=np.float32)
    prev_pids      = np.array([], dtype=int)
    frame_pids     = []    # provisional pseudo_ids per frame (list of arrays)

    for fi, boxes in ped_frames:
        K = len(boxes)
        cur_pids = np.full(K, -1, dtype=int)

        if K > 0 and len(prev_boxes) > 0:
            matches = greedy_match(prev_boxes, boxes)   # {prev_idx: cur_idx}
            for prev_i, cur_i in matches.items():
                cur_pids[cur_i] = prev_pids[prev_i]

        # Assign fresh ids to unmatched current detections.
        for k in range(K):
            if cur_pids[k] == -1:
                cur_pids[k] = n_tracks
                n_tracks += 1

        frame_pids.append(cur_pids)
        prev_boxes = boxes
        prev_pids  = cur_pids

    # Mark singleton tracks (appeared in only one frame) as -1.
    all_pids_flat = np.concatenate(frame_pids) if frame_pids else np.array([], dtype=int)
    valid = all_pids_flat[all_pids_flat >= 0]
    if len(valid) > 0:
        unique_ids, counts = np.unique(valid, return_counts=True)
        singleton_set = set(unique_ids[counts == 1].tolist())
    else:
        singleton_set = set()

    for arr in frame_pids:
        for k in range(len(arr)):
            if arr[k] in singleton_set:
                arr[k] = -1

    # ── Step 3: build temporal_pairs index map ────────────────────────────────
    # We need crop-level indices, which we don't know until after extraction.
    # Record (frame_fi, box_k) → pseudo_id, then resolve indices post-extraction.
    # Instead, record which (fi, k) pairs share a pseudo_id; we'll convert below.

    # ── Step 4: extract crops ─────────────────────────────────────────────────
    crops_list    = []
    pids_list     = []
    sess_ids_list = []
    fids_list     = []

    # Also record (frame_idx, box_k) → crop_index for pair resolution.
    crop_map = {}   # (fi, k) → crop_index (within this session's list)

    for (fi, boxes), cur_pids in zip(ped_frames, frame_pids):
        K = len(boxes)
        if K == 0:
            continue

        frame_name = f"frame_{fi:05d}.npy"
        frame_path = os.path.join(frames_dir, frame_name)
        if not os.path.exists(frame_path):
            continue

        pts = np.load(frame_path)      # (N, 4) — x,y,z,intensity
        pts_xyz = pts[:, :3].astype(float)

        for k in range(K):
            crop = extract_crop(pts_xyz, boxes[k])
            crop_idx = len(crops_list)
            crop_map[(fi, k)] = crop_idx
            crops_list.append(crop)
            pids_list.append(cur_pids[k])
            sess_ids_list.append(sess_idx)
            fids_list.append(fi)

    # ── Step 5: build temporal_pairs ─────────────────────────────────────────
    # Positive pair: two crops from consecutive frames sharing a pseudo_id.
    # Reconstruct which (fi,k) corresponded to each pseudo_id per frame.

    # Build per-frame (fi, k) lists alongside pids.
    fi_k_per_frame = []
    for (fi, boxes), cur_pids in zip(ped_frames, frame_pids):
        K = len(boxes)
        fi_k_per_frame.append([(fi, k) for k in range(K)])

    pairs_list = []
    for t in range(len(fi_k_per_frame) - 1):
        pids_t  = frame_pids[t]
        pids_t1 = frame_pids[t + 1]
        fk_t    = fi_k_per_frame[t]
        fk_t1   = fi_k_per_frame[t + 1]

        # Build pid → crop_index for frame t and t+1.
        pid_to_crop_t  = {pids_t[k]:  crop_map[fk_t[k]]  for k in range(len(pids_t))
                          if fk_t[k]  in crop_map and pids_t[k]  >= 0}
        pid_to_crop_t1 = {pids_t1[k]: crop_map[fk_t1[k]] for k in range(len(pids_t1))
                          if fk_t1[k] in crop_map and pids_t1[k] >= 0}

        for pid, ci in pid_to_crop_t.items():
            if pid in pid_to_crop_t1:
                pairs_list.append((ci, pid_to_crop_t1[pid]))

    return (
        np.array(crops_list,    dtype=np.float32) if crops_list else np.empty((0, N_PTS, 3), dtype=np.float32),
        np.array(pids_list,     dtype=int),
        np.array(sess_ids_list, dtype=int),
        np.array(fids_list,     dtype=int),
        np.array(pairs_list,    dtype=int).reshape(-1, 2) if pairs_list else np.empty((0, 2), dtype=int),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

all_crops     = []
all_pids      = []
all_sess_ids  = []
all_frame_ids = []
all_pairs     = []   # will need global crop-index offsets

crop_offset = 0
per_session_crop_counts = []

for si, sess in enumerate(SESSIONS):
    print(f"Processing session {si}: {sess} ...")
    crops, pids, sess_ids, frame_ids, pairs = process_session(si, sess)

    # Offset the pair indices into the global crop array.
    if len(pairs) > 0:
        global_pairs = pairs + crop_offset
        all_pairs.append(global_pairs)

    all_crops.append(crops)
    all_pids.append(pids)
    all_sess_ids.append(sess_ids)
    all_frame_ids.append(frame_ids)
    per_session_crop_counts.append(len(crops))
    crop_offset += len(crops)
    print(f"  → {len(crops)} crops, {len(pairs)} positive pairs")

# Concatenate across sessions.
crops_out    = np.concatenate(all_crops,    axis=0) if all_crops    else np.empty((0, N_PTS, 3), dtype=np.float32)
pids_out     = np.concatenate(all_pids,     axis=0) if all_pids     else np.array([], dtype=int)
sess_out     = np.concatenate(all_sess_ids, axis=0) if all_sess_ids else np.array([], dtype=int)
fids_out     = np.concatenate(all_frame_ids,axis=0) if all_frame_ids else np.array([], dtype=int)
pairs_out    = np.concatenate(all_pairs,    axis=0) if all_pairs    else np.empty((0, 2), dtype=int)

# ── Save ──────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "crops.npy"),         crops_out)
np.save(os.path.join(OUT_DIR, "pseudo_ids.npy"),    pids_out)
np.save(os.path.join(OUT_DIR, "session_ids.npy"),   sess_out)
np.save(os.path.join(OUT_DIR, "frame_ids.npy"),     fids_out)
np.save(os.path.join(OUT_DIR, "temporal_pairs.npy"),pairs_out)

# ── Stats ─────────────────────────────────────────────────────────────────────
total_crops      = len(crops_out)
crops_with_pid   = int((pids_out >= 0).sum())
positive_pairs   = len(pairs_out)

# Negative pairs: unmatched detections in same frame.
# Count all pairs of crops in the same (session, frame) that have different pseudo_ids
# (or at least one is -1).  Only count unordered pairs.
neg_pairs = 0
for si in range(len(SESSIONS)):
    sess_mask = sess_out == si
    frame_ids_sess = fids_out[sess_mask]
    pids_sess      = pids_out[sess_mask]
    sess_crop_idx  = np.where(sess_mask)[0]
    for fi in np.unique(frame_ids_sess):
        fi_mask = frame_ids_sess == fi
        fi_pids = pids_sess[fi_mask]
        n = fi_pids.sum() if False else fi_mask.sum()   # number of crops in this frame
        # All pairs where pseudo_ids differ — unordered
        for a in range(n):
            for b in range(a + 1, n):
                if fi_pids[a] != fi_pids[b]:
                    neg_pairs += 1

stats_lines = [
    f"total_crops:       {total_crops}",
    f"crops_with_pid:    {crops_with_pid}",
    f"positive_pairs:    {positive_pairs}",
    f"negative_pairs:    {neg_pairs}",
]
for si, (sess, cnt) in enumerate(zip(SESSIONS, per_session_crop_counts)):
    stats_lines.append(f"session_{si}_crops:  {cnt}  ({sess})")

stats_text = "\n".join(stats_lines)
print("\n" + stats_text)

with open(os.path.join(OUT_DIR, "mining_stats.txt"), "w") as f:
    f.write(stats_text + "\n")

print(f"\nAll outputs written to {OUT_DIR}")
