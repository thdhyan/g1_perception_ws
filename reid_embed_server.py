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
         "frame_cache": {}, "smpl_mode": False,
         "frame_boxes": None, "frame_beta": None, "frame_theta": None,
         "mesh_decoder": None}


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
            "anchor": tr["anchor"].tolist(),  # EMA-smoothed embedding/beta for this track
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
        elif path == "/api/meshfaces":
            S = STATE
            if not S["smpl_mode"]:
                self._send(400, {"error": "not in --smpl-mode"})
                return
            with S["lock"]:
                if S["mesh_decoder"] is None:
                    from mesh_utils import get_decoder
                    S["mesh_decoder"] = get_decoder()
                faces = S["mesh_decoder"].faces  # (F, 3) int32
            self._send(200, {"faces_b64": base64.b64encode(
                np.ascontiguousarray(faces, dtype="<i4").tobytes()).decode(),
                             "n_faces": int(faces.shape[0])})
        elif path == "/api/mesh":
            S = STATE
            if not S["smpl_mode"]:
                self._send(400, {"error": "not in --smpl-mode"})
                return
            q = parse_qs(u.query)
            try:
                tid = int(q.get("tid", ["0"])[0])
                fi  = int(q.get("fi", ["0"])[0])
            except ValueError:
                self._send(400, {"error": "bad tid/fi"})
                return
            track = next((t for t in S["tracks"] if t["tid"] == tid), None)
            if track is None or not track["pts"]:
                self._send(404, {"error": "unknown tid"})
                return
            # Gather a small temporal window of this track's nearest measured
            # points and average box pose + theta over them, to smooth out
            # per-frame box/pose regression jitter (a single frame's box can
            # jitter in yaw/position and theta can have a noisy regression).
            pts = sorted(track["pts"], key=lambda p: abs(p[0] - fi))[:5]
            pts = [p for p in pts if abs(p[0] - fi) <= 15]
            if not pts:
                self._send(404, {"error": "no nearby measured frame"})
                return
            used_fi = min(pts, key=lambda p: abs(p[0] - fi))[0]

            boxes_l, thetas_l = [], []
            for p in pts:
                pfi, tx, ty = p[0], p[1], p[2]
                boxes_at = S["frame_boxes"][pfi]
                theta_at = S["frame_theta"][pfi]
                if boxes_at is None or theta_at is None:
                    continue
                d = np.hypot(boxes_at[:, 0] - tx, boxes_at[:, 1] - ty)
                j = int(np.argmin(d))
                if d[j] > 0.5:
                    continue
                boxes_l.append(boxes_at[j])
                thetas_l.append(theta_at[j])
            if not boxes_l:
                self._send(404, {"error": "no matching detection near track points"})
                return
            boxes_arr  = np.array(boxes_l, dtype=np.float64)   # (K, 7)
            thetas_arr = np.array(thetas_l, dtype=np.float64)  # (K, 72)

            cx, cy, cz = boxes_arr[:, 0].mean(), boxes_arr[:, 1].mean(), boxes_arr[:, 2].mean()
            yaw_all = boxes_arr[:, 6]
            yaw = math.atan2(np.sin(yaw_all).mean(), np.cos(yaw_all).mean())  # circular mean
            theta = thetas_arr.mean(axis=0)                    # elementwise avg (small-motion approx)
            beta  = np.array(track["anchor"], dtype=np.float64)  # tracker's EMA-smoothed identity beta

            with S["lock"]:
                if S["mesh_decoder"] is None:
                    from mesh_utils import get_decoder
                    S["mesh_decoder"] = get_decoder()
                dec = S["mesh_decoder"]
            verts_local, _ = dec.vertices(beta.astype(np.float32), theta.astype(np.float32))  # (1, 6890, 3), root-centred
            v = verts_local[0]
            c, s = math.cos(yaw), math.sin(yaw)   # inverse of the -yaw de-yaw used at crop time
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            v_world = (R @ v.T).T + np.array([cx, cy, cz], dtype=np.float32)
            self._send(200, {"tid": tid, "fi": used_fi,
                             "verts_b64": base64.b64encode(
                                 np.ascontiguousarray(v_world, dtype="<f4").tobytes()).decode()})
        else:
            self._send(404, b"not found", "text/plain")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session",    default="2026-07-29_17-21-48")
    ap.add_argument("--lidar-dir",  default="~/Projects/Thesis/Lidar Data")
    ap.add_argument("--model",      default="reid_data/model_identity.pt")
    ap.add_argument("--min-score",  type=float, default=0.4)
    ap.add_argument("--short-gap",  type=int,   default=5,
                    help="Frames gap ≤ this uses pos+embed cost; beyond → embed-only re-ID")
    ap.add_argument("--cos-thresh", type=float, default=0.75,
                    help="Cosine similarity threshold for long-gap re-ID (higher = stricter)")
    ap.add_argument("--pos-gate",   type=float, default=2.0,
                    help="Max xy distance (m) to consider a position match at all")
    ap.add_argument("--pos-weight", type=float, default=0.5,
                    help="Position term weight in cost (embed weight = 1 - this)")
    ap.add_argument("--min-len",    type=int,   default=5,
                    help="Suppress tracks with fewer than this many observed frames")
    ap.add_argument("--device",     default="cuda" if _cuda_ok() else "cpu")
    ap.add_argument("--max-range",  type=float, default=5.0,
                    help="Discard detections farther than this many metres from sensor (0=off)")
    ap.add_argument("--recompute",  action="store_true",
                    help="Ignore cached embeddings and recompute from scratch")
    ap.add_argument("--smpl-mode",  action="store_true",
                    help="Use SMPL beta (10-d shape) for ReID instead of learned embeddings")
    ap.add_argument("--smpl-checkpoint", default="humanm3",
                    help="LiDAR-HMR checkpoint tag used for pre-computed smpl_<tag>_<session>_*.npy files")
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
    if not args.smpl_mode and not os.path.exists(model_p):
        import sys; sys.exit(f"model not found: {model_p}")

    S = STATE
    S["session"]    = args.session
    S["frames_dir"] = frames_d
    S["npz"]        = np.load(labels_p, allow_pickle=True)

    if args.smpl_mode:
        # Load pre-computed SMPL beta vectors (produced by extract_smpl.py).
        # beta slots directly into the embed-cost tracker (dimension-agnostic).
        smpl_prefix = os.path.join(HERE, "reid_data",
                                   f"smpl_{args.smpl_checkpoint}_{args.session}")
        if not os.path.exists(smpl_prefix + "_beta.npy"):
            # fall back to un-tagged filename (single-checkpoint extraction)
            smpl_prefix = os.path.join(HERE, "reid_data", f"smpl_{args.session}")
        if not os.path.exists(smpl_prefix + "_beta.npy"):
            import sys; sys.exit(f"smpl data not found: {smpl_prefix}_beta.npy "
                                 f"(run extract_smpl.py --session {args.session} first)")
        t0 = time.time()
        print(f"[smpl] Loading pre-computed beta from {smpl_prefix}_*.npy")
        fi_arr   = np.load(smpl_prefix + "_fi.npy")
        box_arr  = np.load(smpl_prefix + "_box.npy")
        beta_arr = np.load(smpl_prefix + "_beta.npy")
        theta_p  = smpl_prefix + "_theta.npy"
        theta_arr = np.load(theta_p) if os.path.exists(theta_p) else None
        z_arr   = box_arr[:, 5] / 2.0   # h/2, same display convention as embed path
        if args.max_range > 0:
            r = np.hypot(box_arr[:, 0], box_arr[:, 1])
            keep = r <= args.max_range
            fi_arr, box_arr, beta_arr, z_arr = fi_arr[keep], box_arr[keep], beta_arr[keep], z_arr[keep]
            if theta_arr is not None:
                theta_arr = theta_arr[keep]
            print(f"[smpl] Range filter ≤{args.max_range}m: kept {keep.sum()} / {len(keep)}")
        n_frames = len(S["npz"]["frame_files"])
        frame_boxes = [None] * n_frames
        frame_embs  = [None] * n_frames
        frame_z     = [None] * n_frames
        frame_theta = [None] * n_frames
        for fi in np.unique(fi_arr):
            sel = fi_arr == fi
            frame_boxes[fi] = box_arr[sel]
            frame_embs[fi]  = beta_arr[sel]
            frame_z[fi]     = z_arr[sel]
            if theta_arr is not None:
                frame_theta[fi] = theta_arr[sel]
        n_dets = int(len(fi_arr))
        print(f"[smpl] {n_dets} detections loaded  ({time.time()-t0:.1f}s)")
        S["smpl_mode"]   = True
        S["frame_boxes"] = frame_boxes
        S["frame_beta"]  = frame_embs
        S["frame_theta"] = frame_theta
        _finish_and_serve(args, S, frame_boxes, frame_embs, frame_z)
        return

    # Check for pre-computed embeddings (from precompute script).
    # Cache is keyed by session AND model so that switching models never
    # silently reuses embeddings from a stale checkpoint.
    model_tag = os.path.splitext(os.path.basename(args.model))[0]
    emb_prefix = os.path.join(HERE, "reid_data", f"emb_{args.session}_{model_tag}")
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
        # Persist these embeddings under the model-aware key so subsequent
        # runs skip recomputation (and cannot mix models).
        fi_l, box_l, z_l, emb_l = [], [], [], []
        for fi in range(len(frame_boxes)):
            if frame_boxes[fi] is None:
                continue
            for k in range(len(frame_boxes[fi])):
                fi_l.append(fi)
                box_l.append(frame_boxes[fi][k])
                z_l.append(frame_z[fi][k])
                emb_l.append(frame_embs[fi][k])
        if fi_l:
            np.save(emb_prefix + "_fi.npy",  np.array(fi_l,  dtype=np.int32))
            np.save(emb_prefix + "_box.npy", np.array(box_l, dtype=np.float32))
            np.save(emb_prefix + "_z.npy",   np.array(z_l,   dtype=np.float32))
            np.save(emb_prefix + "_emb.npy", np.array(emb_l, dtype=np.float32))
            print(f"[embed] Saved cache → {emb_prefix}_*.npy")

    _finish_and_serve(args, S, frame_boxes, frame_embs, frame_z)


def _finish_and_serve(args, S, frame_boxes, frame_embs, frame_z):
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
