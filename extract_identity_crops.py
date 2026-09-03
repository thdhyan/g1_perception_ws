"""
extract_identity_crops.py — Extract person crops using ground-truth identity annotations.

Reads identity_map_<session>.json (human-annotated clean segments),
matches each slot observation's (frame, x, y) to the nearest ped box in the
detection NPZ, extracts the LiDAR crop, and saves with a true person_id label.

Output (reid_data/):
  identity_crops.npy         (N, 256, 3)  float32
  identity_person_ids.npy    (N,)         int     — 0-based person index
  identity_frame_ids.npy     (N,)         int
  identity_session_ids.npy   (N,)         int     — 0 for July, etc.
  identity_stats.txt

Usage:
    python3 extract_identity_crops.py
"""
import json
import math
import os
import numpy as np

SESSION    = "2026-07-29_17-21-48"
LIDAR_DIR  = os.path.expanduser("~/Projects/Thesis/Lidar Data")
NPZ_PATH   = os.path.join(LIDAR_DIR, f"{SESSION}_frames_voxelnext.npz")
FRAMES_DIR = os.path.join(LIDAR_DIR, "frames", SESSION)
IDMAP_PATH = f"reid_data/identity_map_{SESSION}.json"
OUT_DIR    = "reid_data"
PED_LABEL  = 2
MIN_SCORE  = 0.15
N_PTS      = 256
MATCH_DIST = 0.5   # max xy distance (m) to match slot obs → npz box

rng = np.random.default_rng(42)


# ── crop extraction ───────────────────────────────────────────────────────────

def extract_crop(pts_xyz, box7):
    cx, cy, cz, dx, dy, dz, yaw = box7.astype(float)
    valid = ~((pts_xyz[:, 0] == 0.0) & (pts_xyz[:, 1] == 0.0))
    pts = pts_xyz[valid] - np.array([cx, cy, cz])
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = (R @ pts.T).T
    mask = ((np.abs(pts[:, 0]) <= dx / 2) &
            (np.abs(pts[:, 1]) <= dy / 2) &
            (np.abs(pts[:, 2]) <= dz / 2))
    p = pts[mask]
    if len(p) == 0:
        return np.zeros((N_PTS, 3), dtype=np.float32)
    idx = rng.choice(len(p), N_PTS, replace=True)
    return p[idx].astype(np.float32)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Load identity map.
    with open(IDMAP_PATH) as f:
        idmap = json.load(f)

    # Load tracker output (7-slot, port 8766 API dump).
    # We need per-slot, per-frame positions to match to npz boxes.
    # Re-fetch from the saved file if present, else load npz + re-track.
    tracks_path = "/tmp/tracks_8766.json"
    if not os.path.exists(tracks_path):
        raise FileNotFoundError(
            f"Need {tracks_path}. Run:\n"
            "  curl -s http://localhost:8766/api/tracks > /tmp/tracks_8766.json")
    with open(tracks_path) as f:
        tracks_data = json.load(f)["tracks"]   # list of track dicts

    # Build slot → {frame → (x, y)} lookup from measured points only.
    slot_frame_xy = {}   # slot_id → {frame → np.array([x,y])}
    for t in tracks_data:
        sid = t["tid"]
        slot_frame_xy[sid] = {}
        for p in t["pts"]:
            if p[4] == "m":
                slot_frame_xy[sid][int(p[0])] = np.array([p[1], p[2]], dtype=np.float32)

    # Load detection NPZ.
    npz = np.load(NPZ_PATH, allow_pickle=True)
    n_frames = len(npz["frame_files"])

    # Build per-frame ped-box lookup: frame → boxes (K,7), filtered by score.
    print("Building per-frame ped-box index …")
    frame_boxes = {}
    for fi in range(n_frames):
        boxes  = np.asarray(npz["pred_boxes"][fi])
        labels = np.asarray(npz["pred_labels"][fi])
        scores = np.asarray(npz["pred_scores"][fi])
        mask   = (labels == PED_LABEL) & (scores >= MIN_SCORE)
        if mask.any():
            frame_boxes[fi] = boxes[mask].astype(np.float32)

    # Assign person labels from clean_segments.
    person_names = sorted(idmap["people"].keys())
    person_id_map = {name: i for i, name in enumerate(person_names)}
    print(f"People: {person_names}")

    crops_out   = []
    pids_out    = []
    fids_out    = []
    sids_out    = []   # session index (0 = July)
    matched     = 0
    unmatched   = 0

    for person_name, person_info in idmap["people"].items():
        pid = person_id_map[person_name]
        for seg in person_info["clean_segments"]:
            slot       = seg["slot"]
            fi_start   = seg["frame_start"]
            fi_end     = seg["frame_end"]

            if slot not in slot_frame_xy:
                print(f"  WARNING: slot {slot} not in track data")
                continue

            slot_obs = slot_frame_xy[slot]
            for fi in range(fi_start, fi_end + 1):
                if fi not in slot_obs:
                    continue   # slot not observed this frame
                if fi not in frame_boxes:
                    continue   # no ped detections in npz this frame

                # Match slot position to nearest npz ped box.
                slot_xy = slot_obs[fi]         # (2,)
                boxes   = frame_boxes[fi]      # (K, 7)
                dists   = np.hypot(boxes[:, 0] - slot_xy[0],
                                   boxes[:, 1] - slot_xy[1])
                best    = int(dists.argmin())
                if dists[best] > MATCH_DIST:
                    unmatched += 1
                    continue   # no close box — skip

                # Load frame and extract crop.
                frame_path = os.path.join(FRAMES_DIR, str(npz["frame_files"][fi]))
                if not os.path.exists(frame_path):
                    continue
                pts = np.load(frame_path)[:, :3].astype(float)
                crop = extract_crop(pts, boxes[best])

                crops_out.append(crop)
                pids_out.append(pid)
                fids_out.append(fi)
                sids_out.append(0)   # July = session 0
                matched += 1

            print(f"  {person_name} slot {slot} f{fi_start}-{fi_end}: "
                  f"{matched - len(pids_out) + sum(1 for p in pids_out if p==pid)} crops so far")

    print(f"\nMatched: {matched}  Unmatched (box too far): {unmatched}")

    if not crops_out:
        print("No crops extracted. Check identity_map paths and slot numbers.")
        return

    crops_np = np.array(crops_out,  dtype=np.float32)
    pids_np  = np.array(pids_out,   dtype=np.int32)
    fids_np  = np.array(fids_out,   dtype=np.int32)
    sids_np  = np.array(sids_out,   dtype=np.int32)

    np.save(os.path.join(OUT_DIR, "identity_crops.npy"),       crops_np)
    np.save(os.path.join(OUT_DIR, "identity_person_ids.npy"),  pids_np)
    np.save(os.path.join(OUT_DIR, "identity_frame_ids.npy"),   fids_np)
    np.save(os.path.join(OUT_DIR, "identity_session_ids.npy"), sids_np)

    from collections import Counter
    cnt = Counter(pids_np.tolist())
    lines = [
        f"session: {SESSION}",
        f"total_crops: {len(crops_np)}",
        f"unmatched_slots: {unmatched}",
    ]
    for pid, name in enumerate(person_names):
        lines.append(f"{name} (id={pid}): {cnt.get(pid, 0)} crops")
    stats_text = "\n".join(lines)
    print("\n" + stats_text)
    with open(os.path.join(OUT_DIR, "identity_stats.txt"), "w") as f:
        f.write(stats_text + "\n")

    print(f"\nSaved to {OUT_DIR}/identity_*.npy")


if __name__ == "__main__":
    main()
