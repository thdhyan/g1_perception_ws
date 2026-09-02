#!/usr/bin/env python3
"""
reid_web_server.py — same-person ReID annotator as a LOCAL WEB APP (three.js).

Run this, then open  http://localhost:8765  in your browser.

    Two side-by-side interactive 3D views  (FRAME A | FRAME B)
    left-drag = rotate · right-drag = pan · wheel = zoom · dbl-click = fit
    Y same person · N different · Space skip · ← previous · Q finish

Python stdlib only (http.server). Three.js is vendored in ./reid_web/vendor
(no internet needed). Annotations are saved after EVERY verdict to
reid_data/sameperson_<session>.json  (resumable; overwrites if re-answered).

Usage:
    python3 reid_web_server.py \
        [--session 2026-08-05_16-38-40] [--port 8765] \
        [--labels <session>_frames_voxelnext.npz] [--max-dist 2.5]
"""
import argparse
import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE      = os.path.dirname(os.path.abspath(__file__))
WS        = HERE
WEB_DIR   = os.path.join(HERE, "reid_web")
PED_LABEL = 2
FLOOR_LO, FLOOR_HI = 0.0, 0.7
PERSON_TINT_R = 1.3

STATE = {
    "lock": threading.RLock(),   # reentrant: do_POST holds it, then save_annotations also locks
    "npz": None, "frames_dir": None, "session": None,
    "pairs": [], "answered": {}, "counts": {"yes": 0, "no": 0, "skip": 0},
    "ann_path": None, "frame_cache": {},
    "emb": None, "emb_fi": None, "emb_box": None, "emb_score": None,
    "emb_lookup": {},   # (fi, ki) → emb vector (128,)
}


# ── pair construction (identical to the matplotlib tool) ─────────────────────
def greedy_match(boxes_a, boxes_b, max_dist):
    pairs, used_a, used_b = [], set(), set()
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return pairs
    c_a, c_b = boxes_a[:, :3], boxes_b[:, :3]
    d = np.linalg.norm(c_a[:, None, :] - c_b[None, :, :], axis=-1)
    for flat in np.argsort(d, axis=None):
        i, j = int(flat // len(boxes_b)), int(flat % len(boxes_b))
        if d[i, j] >= max_dist:
            break
        if i not in used_a and j not in used_b:
            pairs.append((i, j, float(d[i, j])))
            used_a.add(i); used_b.add(j)
    return pairs


def build_pairs(npz, max_dist, min_score=0.0, size_ratio=None, gap_max=2,
                min_gap=0, appearance_ratio=1.3, emb_lookup=None, cos_thresh=0.7):
    """Build candidate pairs. min_gap>1 switches to REID mode: far-apart
    frames, appearance-similar boxes. Uses embeddings if available (cosine
    similarity), falls back to box-dim log-L2 distance."""
    n = len(npz["frame_files"])
    per_frame = []
    for fi in range(n):
        boxes, labels, scores = npz["pred_boxes"][fi], npz["pred_labels"][fi], npz["pred_scores"][fi]
        m = labels == PED_LABEL
        boxes, scores = np.asarray(boxes[m], dtype=np.float32), np.asarray(scores[m], dtype=np.float32)
        if min_score > 0:
            keep = scores >= min_score
            boxes, scores = boxes[keep], scores[keep]
        per_frame.append((boxes, scores))

    # ── REID mode: far-apart, appearance-similar ──────────────────────────────
    if min_gap > 1:
        return _build_reid_pairs(per_frame, n, min_gap, gap_max, appearance_ratio,
                                 emb_lookup=emb_lookup or {}, cos_thresh=cos_thresh)

    # ── LOCAL mode: original adjacent-frame behaviour ─────────────────────────
    pairs = []
    for fi in range(n):
        ba = per_frame[fi][0]
        if len(ba) == 0:
            continue
        taken = set()
        for gap in range(1, gap_max + 1):
            if fi + gap >= n:
                continue
            b1 = per_frame[fi + gap][0]
            if len(b1) == 0:
                continue
            limit = max_dist * (1 + 0.5 * (gap - 1))
            for i, j, dist in greedy_match(ba, b1, limit):
                if i in taken:
                    continue
                if size_ratio is not None:
                    r = np.maximum(ba[i, :3], b1[j, :3]) / np.maximum(np.minimum(ba[i, :3], b1[j, :3]), 1e-6)
                    if r.max() > size_ratio:
                        continue
                taken.add(i)
                pairs.append({"f0": fi, "k0": i, "f1": fi + gap, "k1": j,
                              "dist": dist, "gap": gap})
    pairs.sort(key=lambda p: (p["dist"], -p["gap"]))
    return pairs


def _build_reid_pairs(per_frame, n, min_gap, gap_max, appearance_ratio,
                       emb_lookup=None, cos_thresh=0.7):
    """Pair detections from distant frames with similar appearance.

    If emb_lookup is populated: rank by cosine similarity of 128-d embeddings
    (L2-normalised, so dot product = cosine).  Otherwise fall back to
    log-dim L2 distance.

    Always includes h (height/dz), w (hip width/dx) for both boxes."""
    emb_lookup = emb_lookup or {}
    use_emb = len(emb_lookup) > 0

    # ── gather detections and precompute frame groups ──────────────────────────
    # frame_groups[fi] = list of local indices into the flat arrays
    frame_groups = {}
    all_fi, all_ki, all_dims, all_emb = [], [], [], []
    for fi in range(n):
        boxes = per_frame[fi][0]
        for ki in range(len(boxes)):
            b = boxes[ki]
            idx = len(all_fi)
            all_fi.append(fi)
            all_ki.append(ki)
            all_dims.append((float(b[3]), float(b[4]), float(b[5])))
            all_emb.append(emb_lookup.get((fi, ki)))
            frame_groups.setdefault(fi, []).append(idx)

    if not all_fi:
        return []

    frames = np.array(all_fi, dtype=np.int32)
    kis    = np.array(all_ki, dtype=np.int32)
    dims   = np.array(all_dims, dtype=np.float32)
    log_d  = np.log(np.maximum(dims, 1e-6))
    embs   = np.array([e if e is not None else np.zeros(128, dtype=np.float32)
                        for e in all_emb], dtype=np.float32) if use_emb else None
    has_emb = np.array([e is not None for e in all_emb]) if use_emb else None

    dim_thresh = np.sqrt(3) * np.log(appearance_ratio)

    pairs = []
    MAX_PER_FRAME = 20   # cap pairs originating from each source frame
    log_interval = max(1, n // 10)
    for fi in range(n):
        if fi % log_interval == 0:
            print(f"  [reid] frame {fi}/{n} … ({len(pairs)} pairs so far)")
        ai = frame_groups.get(fi)
        if not ai:
            continue

        # find detection indices in distant frames [fi+min_gap, fi+gap_max]
        bi = []
        for fj in range(fi + min_gap, min(fi + gap_max + 1, n)):
            bi.extend(frame_groups.get(fj, []))
        if not bi:
            continue

        ai_arr = np.array(ai, dtype=np.int32)
        bi_arr = np.array(bi, dtype=np.int32)

        if use_emb:
            # cosine similarity via dot product (embeddings are L2-normed)
            cos_sim = embs[ai_arr] @ embs[bi_arr].T  # (Na, Nb)
            fj_vals = frames[bi_arr]
            for a_local in range(len(ai)):
                if not has_emb[ai[a_local]]:
                    continue
                good = np.where(cos_sim[a_local] >= cos_thresh)[0]
                # cap per-source-frame to avoid memory explosion
                if len(good) > MAX_PER_FRAME:
                    order = np.argsort(-cos_sim[a_local][good])
                    good = good[order[:MAX_PER_FRAME]]
                for b_local in good:
                    if not has_emb[bi[b_local]]:
                        continue
                    pairs.append({
                        "f0": int(fi),  "k0": int(kis[ai[a_local]]),
                        "f1": int(fj_vals[b_local]), "k1": int(kis[bi[b_local]]),
                        "dist": round(float(1.0 - cos_sim[a_local, b_local]), 4),
                        "cos_sim": round(float(cos_sim[a_local, b_local]), 4),
                        "gap":  int(fj_vals[b_local]) - fi,
                        "h0": round(float(dims[ai[a_local], 2]), 3),
                        "w0": round(float(dims[ai[a_local], 0]), 3),
                        "h1": round(float(dims[bi[b_local], 2]), 3),
                        "w1": round(float(dims[bi[b_local], 0]), 3),
                    })
        else:
            # fallback: log-dim L2 distance
            d = np.linalg.norm(log_d[ai_arr][:, None, :] - log_d[bi_arr][None, :, :], axis=-1)
            fj_vals = frames[bi_arr]
            for a_local in range(len(ai)):
                good = np.where(d[a_local] < dim_thresh)[0]
                if len(good) > MAX_PER_FRAME:
                    order = np.argsort(d[a_local][good])
                    good = good[order[:MAX_PER_FRAME]]
                for b_local in good:
                    pairs.append({
                        "f0": int(fi),  "k0": int(kis[ai[a_local]]),
                        "f1": int(fj_vals[b_local]), "k1": int(kis[bi[b_local]]),
                        "dist": round(float(d[a_local, b_local]), 4),
                        "cos_sim": None,
                        "gap":  int(fj_vals[b_local]) - fi,
                        "h0": round(float(dims[ai[a_local], 2]), 3),
                        "w0": round(float(dims[ai[a_local], 0]), 3),
                        "h1": round(float(dims[bi[b_local], 2]), 3),
                        "w1": round(float(dims[bi[b_local], 0]), 3),
                    })

    # sort: most similar first (lowest dist = highest cos_sim)
    pairs.sort(key=lambda p: p["dist"])
    if len(pairs) > 1000:
        pairs = pairs[:1000]
    return pairs


# ── data helpers ──────────────────────────────────────────────────────────────
def local_floor_z(pts, x, y, r=1.0):
    z = pts[(pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2 <= r * r, 2]
    if len(z) < 20:
        pass
    else:
        z = z[z < np.percentile(z, 80)]
    return float(np.median(z)) if len(z) else 0.0


def frame_payload(fi, hi_box_idx, pts_full, peds, peds_sc):
    c = peds[hi_box_idx, :3]
    floor_z = local_floor_z(pts_full, c[0], c[1])
    boxes = []
    for k, b in enumerate(peds):
        b = b.astype(float).tolist()
        b[2] = floor_z + b[5] / 2.0          # rest on local floor (display only)
        boxes.append({"box": b, "score": float(peds_sc[k]), "hi": int(k == hi_box_idx)})
    b64 = base64.b64encode(np.ascontiguousarray(pts_full[:, :3].astype("<f4")).tobytes()).decode()
    return {"pts_b64": b64, "peds": boxes}


def pair_payload(j):
    S = STATE
    p = S["pairs"][j]
    npz = S["npz"]
    # add detection scores to meta for the frontend
    meta = dict(p)
    for side, fi, hi in (("a", p["f0"], p["k0"]), ("b", p["f1"], p["k1"])):
        if fi not in S["frame_cache"]:
            fname = str(npz["frame_files"][fi])
            pts = np.load(os.path.join(S["frames_dir"], fname))
            boxes, labels, scores = npz["pred_boxes"][fi], npz["pred_labels"][fi], npz["pred_scores"][fi]
            S["frame_cache"][fi] = (
                np.ascontiguousarray(pts[:, :3].astype(np.float32)),
                np.asarray(boxes[labels == PED_LABEL], dtype=np.float32),
                np.asarray(scores[labels == PED_LABEL], dtype=np.float32),
            )
        pts, peds, peds_sc = S["frame_cache"][fi]
        if hi < len(peds_sc):
            meta[f"score_{side}"] = round(float(peds_sc[hi]), 3)
    out = {"meta": meta}
    for side, fi, hi in (("a", p["f0"], p["k0"]), ("b", p["f1"], p["k1"])):
        pts, peds, peds_sc = S["frame_cache"][fi]
        out[side] = frame_payload(fi, hi, pts, peds, peds_sc)
    return out


def load_session(args):
    S = STATE
    LIDAR = os.path.expanduser(args.lidar_dir)
    labels_path = args.labels or os.path.join(LIDAR, f"{args.session}_frames_voxelnext.npz")
    frames_dir = args.frames_dir or os.path.join(LIDAR, "frames", args.session)
    if not os.path.exists(labels_path):
        sys.exit(f"labels npz not found: {labels_path}")
    if not os.path.isdir(frames_dir):
        sys.exit(f"frames dir not found: {frames_dir}")

    S["session"], S["frames_dir"] = args.session, frames_dir
    npz = np.load(labels_path, allow_pickle=True)
    S["npz"] = npz

    # ── load precomputed embeddings (128-d, L2-normalised) ────────────────────
    # embeddings live in WS/reid_data/ alongside the annotation files
    emb_stem = os.path.join(WS, "reid_data", f"emb_{args.session}")
    emb_path = emb_stem + "_emb.npy"
    if os.path.exists(emb_path):
        S["emb"]      = np.load(emb_path)                 # (N, 128) float32
        S["emb_fi"]   = np.load(emb_stem + "_fi.npy")     # (N,) int32
        S["emb_box"]  = np.load(emb_stem + "_box.npy")    # (N, 7) float32
        S["emb_score"]= np.load(emb_stem + "_score.npy")  # (N,) float32
        # build (fi, box_idx) → emb vector  (entries are ordered by frame then box_idx)
        entry_count = {}
        lookup = {}
        for i in range(len(S["emb_fi"])):
            fi = int(S["emb_fi"][i])
            ki = entry_count.get(fi, 0)
            lookup[(fi, ki)] = S["emb"][i]
            entry_count[fi] = ki + 1
        S["emb_lookup"] = lookup
        print(f"[emb] loaded {len(S['emb'])} embeddings ({len(lookup)} frame-box keys) from {emb_path}")
    else:
        S["emb_lookup"] = {}
        print(f"[emb] no embeddings found at {emb_path} — using box-dim pairing only")

    S["pairs"] = build_pairs(npz, args.max_dist, min_score=args.min_score,
                             size_ratio=args.size_ratio or None, gap_max=args.gap_max,
                             min_gap=args.min_gap, appearance_ratio=args.appearance_ratio,
                             emb_lookup=S["emb_lookup"], cos_thresh=args.cos_thresh)
    mode = f"REID (min_gap={args.min_gap}, cos_thresh={args.cos_thresh})" if args.min_gap > 1 \
           else f"LOCAL (max_dist {args.max_dist} m)"
    print(f"[pairs] {len(S['pairs'])} candidate pairs  [{mode}] "
          f"(min_score {args.min_score}, size_ratio {args.size_ratio or 'off'}, gap<= {args.gap_max})")
    if not S["pairs"]:
        sys.exit("No pairs found — check --labels / --max-dist.")

    S["ann_path"] = args.out or os.path.join(WS, "reid_data", f"sameperson_{args.session}.json")
    if os.path.exists(S["ann_path"]):
        try:
            prev = json.loads(open(S["ann_path"]).read())
            for a in prev.get("annotations", []):
                S["answered"][(a["f0"], a["k0"], a["f1"], a["k1"])] = a
            print(f"[resume] {len(S['answered'])} already answered")
        except Exception:
            pass
    refresh_counts()


def refresh_counts():
    S = STATE
    c = {"yes": 0, "no": 0, "skip": 0}
    for a in S["answered"].values():
        c[a["verdict"]] = c.get(a["verdict"], 0) + 1
    S["counts"] = c


def save_annotations():
    S = STATE
    os.makedirs(os.path.dirname(S["ann_path"]), exist_ok=True)
    with S["lock"]:
        json.dump({
            "session": S["session"], "pairs_total": len(S["pairs"]),
            "counts": S["counts"],
            "annotations": sorted(S["answered"].values(),
                                  key=lambda a: (a["f0"], a["k0"], a["f1"], a["k1"])),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, open(S["ann_path"], "w"), indent=2)


def pair_key(p):
    return (p["f0"], p["k0"], p["f1"], p["k1"])


# ── sequence endpoint: track a person from f0 → f1 ─────────────────────────
def _get_ped_boxes(fi, npz, frames_dir):
    """Return (pts_f32, peds_list, scores_list) for a single frame."""
    fname = str(npz["frame_files"][fi])
    pts = np.load(os.path.join(frames_dir, fname))
    boxes, labels, scores = npz["pred_boxes"][fi], npz["pred_labels"][fi], npz["pred_scores"][fi]
    m = labels == PED_LABEL
    peds = np.asarray(boxes[m], dtype=np.float32)
    sc   = np.asarray(scores[m], dtype=np.float32)
    return (np.ascontiguousarray(pts[:, :3].astype(np.float32)), peds, sc)


def build_sequence(f0, k0, f1, npz, frames_dir, min_score=0.0, max_gap_step=200,
                   emb_lookup=None):
    """Build a list of frames from f0 → f1, tracking the person.

    Uses embedding cosine similarity when available, falls back to
    nearest-position tracking.  Returns list of dicts:
    {fi, pts_b64, peds[{box,score,hi}], hi_idx, track_lost}."""
    emb_lookup = emb_lookup or {}
    n = len(npz["frame_files"])
    gap = f1 - f0
    if gap <= 0:
        return []

    # decide which frame indices to include
    if gap <= max_gap_step:
        frame_indices = list(range(f0, f1 + 1))
    else:
        step = max(1, gap // max_gap_step)
        frame_indices = list(range(f0, f1 + 1, step))
        if frame_indices[-1] != f1:
            frame_indices.append(f1)

    # get the seed embedding for k0 at f0
    seed_emb = emb_lookup.get((f0, k0))
    use_emb = seed_emb is not None and len(emb_lookup) > 0

    last_pos = None
    last_emb = seed_emb
    frames_data = []

    for fi in frame_indices:
        pts_f32, peds, sc = _get_ped_boxes(fi, npz, frames_dir)

        # filter by min_score
        if min_score > 0 and len(sc):
            keep = sc >= min_score
            peds, sc = peds[keep], sc[keep]

        hi_idx = -1

        if use_emb and last_emb is not None and len(peds):
            # track by embedding cosine similarity
            raw_embs = [emb_lookup.get((fi, ki)) for ki in range(len(peds))]
            valid = np.array([e is not None for e in raw_embs])
            emb_this = np.array([e if e is not None else np.zeros(128, dtype=np.float32)
                                  for e in raw_embs], dtype=np.float32)
            if valid.any():
                sims = np.full(len(peds), -1.0)
                sims[valid] = emb_this[valid] @ last_emb  # dot = cos (L2-normed)
                best = int(np.argmax(sims))
                if sims[best] > 0.3:  # minimum cosine threshold
                    hi_idx = best
                    last_emb = emb_this[best] if valid[best] else last_emb
                    last_pos = peds[best, :3].copy()
            # if no embedding match, try position fallback
            if hi_idx < 0 and last_pos is not None and len(peds):
                centres = peds[:, :3]
                dists = np.linalg.norm(centres - last_pos, axis=1)
                best = int(np.argmin(dists))
                if dists[best] < 3.0:
                    hi_idx = best
                    last_pos = centres[best].copy()
        elif last_pos is not None and len(peds):
            # position-only tracking (no embeddings)
            centres = peds[:, :3]
            dists = np.linalg.norm(centres - last_pos, axis=1)
            best = int(np.argmin(dists))
            if dists[best] < 3.0:
                hi_idx = best
                last_pos = centres[best].copy()
        elif fi == f0 and k0 < len(peds):
            hi_idx = k0
            last_pos = peds[k0, :3].copy()
            if use_emb:
                last_emb = emb_lookup.get((f0, k0))

        # build ped list for frontend
        ped_list = []
        for ki in range(len(peds)):
            b = peds[ki].astype(float).tolist()
            floor_z = local_floor_z(pts_f32, b[0], b[1])
            b[2] = floor_z + b[5] / 2.0
            ped_list.append({"box": b, "score": float(sc[ki]), "hi": int(ki == hi_idx)})

        b64 = base64.b64encode(
            np.ascontiguousarray(pts_f32[:, :3].astype("<f4")).tobytes()
        ).decode()

        frames_data.append({"fi": fi, "pts_b64": b64, "peds": ped_list,
                            "hi_idx": hi_idx, "track_lost": hi_idx < 0})

    return frames_data


# ── HTTP ──────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "ReIDWeb/1.0"

    def log_message(self, fmt, *a):
        print("  ", fmt % a)

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        parts = u.path.lstrip("/").split("/")
        if u.path == "/" or not u.path.strip("/"):
            p = os.path.join(WEB_DIR, "index.html")
            self._send(200, open(p, "rb").read(), "text/html; charset=utf-8")
        elif parts and parts[0] == "vendor" and len(parts) == 2:
            p = os.path.join(WEB_DIR, "vendor", os.path.basename(parts[1]))
            if os.path.isfile(p):
                self._send(200, open(p, "rb").read(),
                           "text/javascript; charset=utf-8" if p.endswith(".js") else "application/octet-stream")
            else:
                self._send(404, b"not found", "text/plain")
        elif len(parts) == 1 and parts[0] in ("app.js", "index.html"):
            p = os.path.join(WEB_DIR, parts[0])
            self._send(200, open(p, "rb").read(),
                       "text/javascript; charset=utf-8" if p.endswith(".js") else "text/html; charset=utf-8")
        elif u.path == "/api/init":
            S = STATE
            self._send(200, {
                "session": S["session"], "n_pairs": len(S["pairs"]),
                "pairs": S["pairs"], "answered": {f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v["verdict"]
                                                  for k, v in S["answered"].items()},
                "counts": S["counts"],
            })
        elif u.path == "/api/pairdata":
            j = int(parse_qs(u.query).get("j", ["0"])[0])
            S = STATE
            if 0 <= j < len(S["pairs"]):
                self._send(200, pair_payload(j))
            else:
                self._send(400, {"error": "bad pair index"}, )
        elif u.path == "/api/status":
            S = STATE
            self._send(200, {"counts": S["counts"],
                             "answered": len(S["answered"]), "total": len(S["pairs"])})
        elif u.path == "/api/sequence":
            q = parse_qs(u.query)
            f0 = int(q.get("f0", ["0"])[0])
            k0 = int(q.get("k0", ["0"])[0])
            f1 = int(q.get("f1", ["0"])[0])
            S = STATE
            frames_data = build_sequence(f0, k0, f1, S["npz"], S["frames_dir"],
                                         min_score=0.0, emb_lookup=S["emb_lookup"])
            self._send(200, {"sequence": frames_data, "n_frames": len(frames_data),
                             "f0": f0, "f1": f1})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/api/verdict":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n))
            j, v = int(body["j"]), body["verdict"]
            assert v in ("yes", "no", "skip")
            S = STATE
            p = S["pairs"][j]
            rec = {**p, "verdict": v, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "web"}
            with S["lock"]:
                S["answered"][pair_key(p)] = rec
                refresh_counts()
                save_annotations()
                self._send(200, {"ok": True, "counts": S["counts"],
                                 "answered": len(S["answered"]), "total": len(S["pairs"])})
        except Exception as e:
            self._send(400, {"error": str(e)}, "application/json")


def main():
    import sys
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="2026-08-05_16-38-40")
    ap.add_argument("--lidar-dir", default="~/Projects/Thesis/Lidar Data")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--max-dist", type=float, default=2.5)
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="Only pair ped boxes with score >= this")
    ap.add_argument("--size-ratio", type=float, default=0.0,
                    help="Max worst-dim ratio between paired boxes (0 = off)")
    ap.add_argument("--gap-max", type=int, default=2, help="Max frame gap (1 = adjacent only)")
    ap.add_argument("--min-gap", type=int, default=0,
                    help="Min frame gap — >1 activates REID mode (far-apart appearance pairs)")
    ap.add_argument("--appearance-ratio", type=float, default=1.3,
                    help="Max (dx,dy,dz) dimension ratio for REID-mode pairing")
    ap.add_argument("--cos-thresh", type=float, default=0.7,
                    help="Min cosine similarity for embedding-based REID pairing")
    ap.add_argument("--out", default=None)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    load_session(args)

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n[web] serving on  http://localhost:{args.port}   (Ctrl-C to stop)")
    print(f"[web] annotations → {STATE['ann_path']}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopping")
        save_annotations()


if __name__ == "__main__":
    main()
