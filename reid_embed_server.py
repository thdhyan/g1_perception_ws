#!/usr/bin/env python3
"""
reid_embed_server.py — trajectory viewer using LEARNED ReID embeddings.

Differences from reid_tracks_server.py:
  * No fixed slot cap (unlimited tracks)
  * No linear interpolation (only real measured frames shown)
  * Hungarian assignment with cost = pos_term + embedding_term
  * Long-gap re-ID: dormant tracks matched by embedding cosine similarity
    when position gate is too wide to be useful

Run:
    python3 reid_embed_server.py --session 2026-07-29_17-21-48 --port 8767

Open:  http://localhost:8767
"""
import argparse
import base64
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE     = os.path.dirname(os.path.abspath(__file__))
WEB_DIR  = os.path.join(HERE, "reid_tracks")   # reuse existing frontend
VEND_DIR = os.path.join(HERE, "reid_web", "vendor")
PED_LABEL = 2

STATE = {"lock": threading.RLock(), "npz": None, "frames_dir": None,
         "session": None, "tracks": [], "stats": {}, "params": {},
         "frame_cache": {}}


# ── crop extraction (same as mine_reid_crops.py) ──────────────────────────────

def _extract_crop(pts_xyz: np.ndarray, box7: np.ndarray, n_pts: int = 256,
                  rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(0)
    cx, cy, cz, dx, dy, dz, yaw = box7.astype(float)
    # Remove Livox no-return placeholders.
    valid = ~((pts_xyz[:, 0] == 0.0) & (pts_xyz[:, 1] == 0.0))
    pts = pts_xyz[valid]
    # Translate to box centre.
    pts = pts - np.array([cx, cy, cz])
    # De-yaw.
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = (R @ pts.T).T
    # Filter inside box.
    mask = ((np.abs(pts[:, 0]) <= dx / 2.0) &
            (np.abs(pts[:, 1]) <= dy / 2.0) &
            (np.abs(pts[:, 2]) <= dz / 2.0))
    pts_in = pts[mask]
    if len(pts_in) == 0:
        return np.zeros((n_pts, 3), dtype=np.float32)
    idx = rng.choice(len(pts_in), n_pts, replace=True)
    return pts_in[idx].astype(np.float32)


# ── pre-extract all embeddings for the session ────────────────────────────────

def embed_session(npz, frames_dir, model_path, min_score=0.4, device_str="cpu"):
    """
    Returns:
      frame_embs  list[np.ndarray | None] — (K_fi, 128) per frame, None if no peds
      frame_boxes list[np.ndarray | None] — (K_fi, 7)  box7 per frame
      frame_z     list[np.ndarray | None] — (K_fi,)    display-z = h/2
    """
    import torch
    from reid_model import ReIDModel

    # Try to detect n_classes from model checkpoint.
    state = torch.load(model_path, map_location="cpu")
    # classifier.weight shape → (n_classes, emb_dim)
    n_classes = state["classifier.weight"].shape[0]
    model = ReIDModel(n_classes=n_classes, emb_dim=128)
    model.load_state_dict(state)
    model.eval()
    device = torch.device(device_str)
    model.to(device)

    n_frames = len(npz["frame_files"])
    rng = np.random.default_rng(42)

    all_crops = []   # list of (fi, box_k, crop_256x3)
    for fi in range(n_frames):
        boxes  = np.asarray(npz["pred_boxes"][fi])
        labels = np.asarray(npz["pred_labels"][fi])
        scores = np.asarray(npz["pred_scores"][fi])
        mask   = (labels == PED_LABEL)
        if min_score > 0:
            mask &= scores >= min_score
        if not mask.any():
            continue
        b = boxes[mask]
        frame_name = str(npz["frame_files"][fi])
        frame_path = os.path.join(frames_dir, frame_name)
        if not os.path.exists(frame_path):
            continue
        pts = np.load(frame_path)[:, :3].astype(float)
        for k, box7 in enumerate(b):
            crop = _extract_crop(pts, box7, rng=rng)
            all_crops.append((fi, k, box7, crop))

    if not all_crops:
        return [], [], []

    # Batch embed.
    batch_size = 128
    crops_np = np.stack([c[3] for c in all_crops])   # (N, 256, 3)
    import torch
    embs_list = []
    with torch.no_grad():
        for start in range(0, len(crops_np), batch_size):
            cb = torch.from_numpy(crops_np[start:start+batch_size]).to(device)
            emb, _ = model(cb)
            embs_list.append(emb.cpu().numpy())
    embs_all = np.concatenate(embs_list, axis=0)   # (N, 128)

    # Re-index: per frame → list of (box7, emb, z_display).
    frame_boxes_map = {}
    frame_embs_map  = {}
    frame_z_map     = {}
    for idx, (fi, k, box7, _) in enumerate(all_crops):
        frame_boxes_map.setdefault(fi, []).append(box7)
        frame_embs_map.setdefault(fi, []).append(embs_all[idx])
        frame_z_map.setdefault(fi, []).append(float(box7[5] / 2.0))  # h/2

    frame_boxes, frame_embs, frame_z = [], [], []
    for fi in range(n_frames):
        if fi in frame_boxes_map:
            frame_boxes.append(np.array(frame_boxes_map[fi], dtype=np.float32))
            frame_embs.append(np.array(frame_embs_map[fi],  dtype=np.float32))
            frame_z.append(np.array(frame_z_map[fi], dtype=np.float32))
        else:
            frame_boxes.append(None)
            frame_embs.append(None)
            frame_z.append(None)

    return frame_boxes, frame_embs, frame_z


# ── ReID-aware Hungarian tracker (no interpolation, no slot cap) ──────────────

def reid_track(npz, frame_boxes, frame_embs, frame_z,
               short_gap=5, cos_thresh=0.55,
               pos_gate=2.0, pos_weight=0.5, min_track_len=2):
    """
    Hungarian tracker with learned embedding cost.

    - short_gap:  frames gap ≤ this → use pos + embed cost
    - cos_thresh: for dormant tracks (gap > short_gap), re-ID if cos ≥ this
    - pos_gate:   max xy distance to consider a position match at all
    - pos_weight: weight of position term; embed weight = 1 - pos_weight
    - min_track_len: tracks with fewer measured frames suppressed in output
    No interpolation: pts always have state "m".
    """
    from scipy.optimize import linear_sum_assignment

    n_frames = len(npz["frame_files"])
    tracks   = {}          # tid → track dict
    next_tid = 1
    EMA_ALPHA = 0.3        # embedding anchor update rate

    def new_track(fi, box7, emb, z):
        nonlocal next_tid
        tid = next_tid; next_tid += 1
        tracks[tid] = {
            "tid": tid,
            "last_fi": fi,
            "last_xy": box7[:2].copy(),
            "anchor": emb.copy(),
            "pts": [[fi, float(box7[0]), float(box7[1]), float(z), "m"]],
            "ni": 0, "nm": 0,
        }

    for fi in range(n_frames):
        if frame_boxes[fi] is None:
            continue
        B = frame_boxes[fi]   # (K, 7)
        E = frame_embs[fi]    # (K, 128)
        Z = frame_z[fi]       # (K,)
        K = len(B)

        active_tids  = [t for t, tr in tracks.items() if fi - tr["last_fi"] <= short_gap]
        dormant_tids = [t for t, tr in tracks.items() if fi - tr["last_fi"] >  short_gap]

        # ── Phase 1: active tracks — Hungarian (pos + embed cost) ────────────
        used_boxes = set()
        if active_tids and K:
            A_xy  = np.array([tracks[t]["last_xy"]  for t in active_tids])  # (A, 2)
            A_emb = np.array([tracks[t]["anchor"]   for t in active_tids])  # (A, 128)
            det_xy = B[:, :2]   # (K, 2)

            pos_dist = np.linalg.norm(A_xy[:, None, :] - det_xy[None, :, :], axis=-1)  # (A, K)
            cos_sim  = A_emb @ E.T          # (A, K) — both L2-normed
            embed_dist = 1.0 - cos_sim      # ∈ [0, 2]

            # Gate: inf if position too far (unphysical).
            gate_ok  = pos_dist < pos_gate
            pos_norm = pos_dist / pos_gate   # normalise to [0, 1]
            cost = pos_weight * pos_norm + (1 - pos_weight) * embed_dist * 0.5
            cost[~gate_ok] = 1e9

            row_ind, col_ind = linear_sum_assignment(cost)
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] >= 1e6:
                    continue
                tid = active_tids[ri]
                tr  = tracks[tid]
                gap = fi - tr["last_fi"]
                tr["nm"] += gap - 1                 # account for missing frames
                tr["last_fi"] = fi
                tr["last_xy"] = B[ci, :2].copy()
                tr["anchor"]  = (1 - EMA_ALPHA) * tr["anchor"] + EMA_ALPHA * E[ci]
                tr["anchor"] /= np.linalg.norm(tr["anchor"]) + 1e-8  # keep unit
                tr["pts"].append([fi, float(B[ci, 0]), float(B[ci, 1]), float(Z[ci]), "m"])
                used_boxes.add(ci)

        # ── Phase 2: dormant tracks — re-ID by embedding only ────────────────
        remaining = [j for j in range(K) if j not in used_boxes]
        if dormant_tids and remaining:
            D_emb = np.array([tracks[t]["anchor"] for t in dormant_tids])  # (D, 128)
            R_emb = E[remaining]                                            # (R, 128)
            cos   = D_emb @ R_emb.T                                        # (D, R)
            cost  = 1.0 - cos
            cost[cost > (1.0 - cos_thresh)] = 1e9   # reject below threshold

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_r = set()
            for ri, ci in zip(row_ind, col_ind):
                if cost[ri, ci] >= 1e6:
                    continue
                tid = dormant_tids[ri]
                tr  = tracks[tid]
                j   = remaining[ci]
                gap = fi - tr["last_fi"]
                tr["nm"] += gap - 1
                tr["last_fi"] = fi
                tr["last_xy"] = B[j, :2].copy()
                tr["anchor"]  = (1 - EMA_ALPHA) * tr["anchor"] + EMA_ALPHA * E[j]
                tr["anchor"] /= np.linalg.norm(tr["anchor"]) + 1e-8
                tr["pts"].append([fi, float(B[j, 0]), float(B[j, 1]), float(Z[j]), "m"])
                matched_r.add(ci)
            remaining = [remaining[ci] for ci in range(len(remaining)) if ci not in matched_r]

        # ── Phase 3: new tracks for unmatched detections ──────────────────────
        for j in remaining:
            new_track(fi, B[j], E[j], Z[j])

    # ── Build output ──────────────────────────────────────────────────────────
    out = []
    for tid, tr in sorted(tracks.items()):
        pts = tr["pts"]
        measured = [p for p in pts if p[4] == "m"]
        if len(measured) < min_track_len:
            continue
        frames   = [p[0] for p in measured]
        out.append({
            "tid": tid,
            "start": frames[0],
            "end":   frames[-1],
            "last":  frames[-1],
            "len":   len(measured),
            "ni":    0,          # no interpolation
            "nm":    tr["nm"],
            "nj":    0,
            "st":    "M" if tr["nm"] > 10 else "OK",
            "pts":   measured,   # only measured points
        })

    out.sort(key=lambda t: -t["len"])
    # Renumber by observation count.
    for rank, t in enumerate(out, 1):
        t["tid"] = rank

    stats = {
        "n_tracks":          len(out),
        "n_slots":           len(out),      # compat alias
        "ok":                sum(1 for t in out if t["st"] == "OK"),
        "interp":            0,
        "missing":           sum(1 for t in out if t["st"] == "M"),
        "n_interp_frames":   0,
        "n_missing_frames":  sum(t["nm"] for t in out),
        "n_matched_frames":  sum(t["len"] for t in out),
        "n_jumps":           0,             # no jump-rejection in ReID tracker
        "n_unassigned":      0,             # no hard clutter cap in ReID tracker
        "longest":           max((t["len"] for t in out), default=0),
    }
    return out, stats


# ── HTTP server (same API as reid_tracks_server.py) ───────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "ReIDEmbed/1.0"

    def log_message(self, fmt, *a):
        print("  ", fmt % a, flush=True)

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path, base):
        p = os.path.join(base, os.path.basename(path))
        if os.path.isfile(p):
            ext = p.rsplit(".", 1)[-1]
            ct  = {"js": "text/javascript; charset=utf-8",
                   "html": "text/html; charset=utf-8"}.get(ext, "application/octet-stream")
            self._send(200, open(p, "rb").read(), ct)
        else:
            self._send(404, b"not found", "text/plain")

    def do_GET(self):
        u    = urlparse(self.path)
        path = u.path
        if path == "/":
            self._static("index.html", WEB_DIR)
        elif path in ("/app.js", "/index.html"):
            self._static(path.lstrip("/"), WEB_DIR)
        elif path.startswith("/vendor/"):
            self._static(path[len("/vendor/"):], VEND_DIR)
        elif path == "/api/tracks":
            S = STATE
            self._send(200, {"session": S["session"],
                             "n_frames": len(S["npz"]["frame_files"]),
                             "params": S["params"],
                             "stats":  S["stats"],
                             "tracks": S["tracks"]})
        elif path == "/api/framecloud":
            S  = STATE
            fi = int(parse_qs(u.query).get("fi", ["0"])[0])
            nf = len(S["npz"]["frame_files"])
            if not (0 <= fi < nf):
                self._send(400, {"error": "bad fi"})
                return
            with S["lock"]:
                if fi not in S["frame_cache"]:
                    if len(S["frame_cache"]) > 24:
                        S["frame_cache"].pop(next(iter(S["frame_cache"])))
                    pts = np.load(os.path.join(S["frames_dir"],
                                               str(S["npz"]["frame_files"][fi])))
                    S["frame_cache"][fi] = np.ascontiguousarray(pts[:, :3].astype("<f4"))
                a = S["frame_cache"][fi]
            self._send(200, {"fi": fi, "pts_b64": base64.b64encode(a.tobytes()).decode()})
        else:
            self._send(404, b"not found", "text/plain")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session",    default="2026-07-29_17-21-48")
    ap.add_argument("--lidar-dir",  default="~/Projects/Thesis/Lidar Data")
    ap.add_argument("--model",      default="reid_data/model.pt")
    ap.add_argument("--min-score",  type=float, default=0.4)
    ap.add_argument("--short-gap",  type=int,   default=5,
                    help="Frames gap ≤ this uses pos+embed cost; beyond → embed-only re-ID")
    ap.add_argument("--cos-thresh", type=float, default=0.55,
                    help="Cosine similarity threshold for long-gap re-ID (higher = stricter)")
    ap.add_argument("--pos-gate",   type=float, default=2.0,
                    help="Max xy distance (m) to consider a position match at all")
    ap.add_argument("--pos-weight", type=float, default=0.5,
                    help="Position term weight in cost (embed weight = 1 - this)")
    ap.add_argument("--min-len",    type=int,   default=3,
                    help="Suppress tracks with fewer than this many observed frames")
    ap.add_argument("--device",     default="cuda" if _cuda_ok() else "cpu")
    ap.add_argument("--max-range",  type=float, default=5.0,
                    help="Discard detections farther than this many metres from sensor (0=off)")
    ap.add_argument("--recompute",  action="store_true",
                    help="Ignore cached embeddings and recompute from scratch")
    ap.add_argument("--port",       type=int,   default=8767)
    args = ap.parse_args()

    LIDAR     = os.path.expanduser(args.lidar_dir)
    labels_p  = os.path.join(LIDAR, f"{args.session}_frames_voxelnext.npz")
    frames_d  = os.path.join(LIDAR, "frames", args.session)
    model_p   = os.path.join(HERE, args.model)
    if not os.path.exists(labels_p):
        import sys; sys.exit(f"labels not found: {labels_p}")
    if not os.path.isdir(frames_d):
        import sys; sys.exit(f"frames dir not found: {frames_d}")
    if not os.path.exists(model_p):
        import sys; sys.exit(f"model not found: {model_p}")

    S = STATE
    S["session"]    = args.session
    S["frames_dir"] = frames_d
    S["npz"]        = np.load(labels_p, allow_pickle=True)

    # Check for pre-computed embeddings (from precompute script).
    emb_prefix = os.path.join(HERE, "reid_data", f"emb_{args.session}")
    emb_fi_p   = emb_prefix + "_fi.npy"
    if os.path.exists(emb_fi_p) and not args.recompute:
        t0 = time.time()
        print(f"[embed] Loading pre-computed embeddings from {emb_prefix}_*.npy")
        fi_arr    = np.load(emb_prefix + "_fi.npy")
        box_arr   = np.load(emb_prefix + "_box.npy")
        z_arr     = np.load(emb_prefix + "_z.npy")
        emb_arr   = np.load(emb_prefix + "_emb.npy")
        # Range filter: keep only detections within max_range metres of sensor.
        if args.max_range > 0:
            r = np.hypot(box_arr[:, 0], box_arr[:, 1])
            keep = r <= args.max_range
            fi_arr  = fi_arr[keep]
            box_arr = box_arr[keep]
            z_arr   = z_arr[keep]
            emb_arr = emb_arr[keep]
            print(f"[embed] Range filter ≤{args.max_range}m: kept {keep.sum()} / {len(keep)}")
        # Reindex into per-frame lists.
        n_frames  = len(S["npz"]["frame_files"])
        frame_boxes = [None] * n_frames
        frame_embs  = [None] * n_frames
        frame_z     = [None] * n_frames
        for fi in np.unique(fi_arr):
            sel = fi_arr == fi
            frame_boxes[fi] = box_arr[sel]
            frame_embs[fi]  = emb_arr[sel]
            frame_z[fi]     = z_arr[sel]
        n_dets = int(len(fi_arr))
        print(f"[embed] {n_dets} detections loaded  ({time.time()-t0:.1f}s)")
    else:
        print(f"[embed] Loading model {model_p} → device={args.device}")
        t0 = time.time()
        frame_boxes, frame_embs, frame_z = embed_session(
            S["npz"], frames_d, model_p, args.min_score, args.device)
        n_dets = sum(len(b) for b in frame_boxes if b is not None)
        print(f"[embed] {n_dets} ped detections embedded  ({time.time()-t0:.1f}s)")

    print("[track] Running ReID Hungarian tracker…")
    t0 = time.time()
    S["tracks"], S["stats"] = reid_track(
        S["npz"], frame_boxes, frame_embs, frame_z,
        short_gap  = args.short_gap,
        cos_thresh = args.cos_thresh,
        pos_gate   = args.pos_gate,
        pos_weight = args.pos_weight,
        min_track_len = args.min_len)
    S["params"] = {
        **vars(args),
        "people":    0,          # no fixed cap — compat key for frontend buildStats
        "max_speed": 0,          # not used in ReID tracker
        "gap_cost":  0,
        "gate":      args.pos_gate,
        "max_bridge": args.short_gap,
    }
    st = S["stats"]
    print(f"[track] {st['n_tracks']} tracks  ({time.time()-t0:.1f}s)")
    print(f"[track] matched={st['n_matched_frames']:,}  missing={st['n_missing_frames']:,}")
    for t in S["tracks"][:12]:
        print(f"   T{t['tid']:<4} frames {t['start']:>4}–{t['last']:>4}  "
              f"len={t['len']:<5} MISS={t['nm']:<5} {t['st']}")

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n[web] http://localhost:{args.port}   (Ctrl-C to stop)\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopping")


def _cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
