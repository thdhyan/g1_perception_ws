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


def build_pairs(npz, max_dist, min_score=0.0, size_ratio=None, gap_max=2):
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
    out = {"meta": p}
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
    S["pairs"] = build_pairs(npz, args.max_dist, min_score=args.min_score,
                             size_ratio=args.size_ratio or None, gap_max=args.gap_max)
    print(f"[pairs] {len(S['pairs'])} candidate pairs "
          f"(max_dist {args.max_dist} m, min_score {args.min_score}, "
          f"size_ratio {args.size_ratio or 'off'}, gap<= {args.gap_max})")
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
