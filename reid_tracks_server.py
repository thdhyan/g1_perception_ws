#!/usr/bin/env python3
"""
reid_tracks_server.py — trajectory viewer: track every "person" (ped box) across
a whole session and draw all trajectories in a local web app (three.js).

Rules (fixed-identity mode, --people N):
  * at most N people at any moment (boxes beyond N are dropped as clutter)
  * a box binds to a person only if its xy displacement since that person's
    last TRUSTED position is plausible: d <= max(0.4 m, max_speed * dt)
    (similarity threshold — kills the suspicious "jumps to the edge")
  * between two trusted positions:  <10 frames missing -> LINEARLY
    INTERPOLATED (column I);  >=10 frames missing -> MISSING span (MIS);
    implausible observation -> REJECTED, trusted neighbours bridged (JMP)
  * slots never die (a person may leave and re-enter the scene)

Run this, then open  http://localhost:8766

    Left  : 3D trajectories (click one to select), labelled moving markers,
            point-cloud context, live playback of the whole session
    Right : people table  (# / last / frames / I / MIS / JMP / status),
            click a row to focus & highlight that trajectory
    Bottom: play/pause + speed, frame slider, cloud + follow toggles

Usage:
    python3 reid_tracks_server.py \
        [--session 2026-07-29_17-21-48] [--port 8766] \
        [--people 7] [--min-score 0.4] [--max-speed 4.0]

Python stdlib + numpy only.
"""
import argparse
import base64
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "reid_tracks")
VEND_DIR = os.path.join(HERE, "reid_web", "vendor")
PED_LABEL = 2

STATE = {
    "lock": threading.RLock(),
    "npz": None, "frames_dir": None, "session": None,
    "tracks": [], "stats": {}, "params": {},
    "frame_cache": {},
}


# ── tracking ──────────────────────────────────────────────────────────────────
def track_session(npz, min_score, gate_m, max_bridge):
    """
    Greedy nearest-neighbour person tracking across consecutive frames.

    Returns (tracks, stats):
      track = {tid, start, end, last, len, ni(interpolated), nm(missing),
               st: 'OK' | 'I' | 'M', pts: [[frame,x,y,z,state]...]}
    """
    n = len(npz["frame_files"])
    per = []  # (centers (k,3), scores (k,))
    for fi in range(n):
        boxes = npz["pred_boxes"][fi]
        labels = npz["pred_labels"][fi]
        scores = npz["pred_scores"][fi]
        m = labels == PED_LABEL
        b, s = np.asarray(boxes[m], dtype=np.float32), np.asarray(scores[m], dtype=np.float32)
        if min_score > 0:
            k = s >= min_score
            b, s = b[k], s[k]
        per.append(b)

    tracks = {}
    next_tid = 1
    for fi in range(n):
        C = per[fi]
        k = len(C)
        atids = [t for t in tracks if tracks[t]["st"] != "closed"]

        matches = []
        if k and atids:
            A = np.stack([tracks[t]["lc"] for t in atids])
            D = np.linalg.norm(A[:, None, :] - C[None, :, :], axis=-1)
            gaps = np.array([fi - tracks[t]["f"] for t in atids], dtype=np.float64)
            Dg = np.where(D > gaps[:, None] * gate_m, np.inf, D)
            used_a, used_c = set(), set()
            for flat in np.argsort(Dg, axis=None):
                dd = Dg.flat[int(flat)]
                if np.isinf(dd):
                    break
                i, j = int(flat) // k, int(flat) % k
                if i in used_a or j in used_c:
                    continue
                used_a.add(i); used_c.add(j)
                matches.append((atids[i], j, float(dd)))

        for tid, j, _ in matches:
            tr = tracks[tid]
            c = C[j]
            gap = fi - tr["f"]
            if gap > 1:
                c0 = tr["lc"]
                for g in range(1, gap):
                    tt = g / gap
                    cp = c0 * (1.0 - tt) + c * tt
                    tr["pts"].append([tr["f"] + g, float(cp[0]), float(cp[1]), float(cp[2]), "i"])
                    tr["ni"] += 1
            tr["pts"].append([fi, float(c[0]), float(c[1]), float(c[2]), "m"])
            tr["f"] = fi
            tr["lc"] = c

        # close tracks missing for more than max_bridge frames
        for tid in list(tracks):
            tr = tracks[tid]
            if tr["st"] != "closed" and fi - tr["f"] > max_bridge:
                tr["st"] = "closed"
                tr["nm"] = fi - tr["f"] - 1  # frames they were gone

        # new tracks
        used_c = {j for _, j, _ in matches}
        for j in range(k):
            if j in used_c:
                continue
            tid = next_tid
            next_tid += 1
            c = C[j]
            tracks[tid] = {"tid": tid, "f": fi, "lc": c, "st": "open",
                           "ni": 0, "nm": 0, "pts": [[fi, float(c[0]), float(c[1]), float(c[2]), "m"]]}

    # finalise
    out = []
    for tid in sorted(tracks):
        tr = tracks[tid]
        if tr["st"] == "closed":
            st = "M" if tr["nm"] > 0 else "I"
        elif tr["ni"] > 0:
            st = "I"
        else:
            st = "OK"
        frames = [p[0] for p in tr["pts"]]
        out.append({
            "tid": tid, "start": frames[0], "end": frames[-1],
            "last": max(p[0] for p in tr["pts"] if p[4] == "m"),
            "len": sum(1 for p in tr["pts"] if p[4] == "m"),
            "ni": tr["ni"], "nm": tr["nm"], "st": st,
            "pts": tr["pts"],
        })
    out.sort(key=lambda t: -(t["len"] + t["ni"]))

    stats = {
        "n_tracks": len(out),
        "ok": sum(1 for t in out if t["st"] == "OK"),
        "interp": sum(1 for t in out if t["st"] == "I"),
        "missing": sum(1 for t in out if t["st"] == "M"),
        "n_interp_frames": sum(t["ni"] for t in out),
        "n_missing_frames": sum(t["nm"] for t in out),
        "n_matched_frames": sum(t["len"] for t in out),
        "longest": max((t["len"] + t["ni"] for t in out), default=0),
    }
    del tracks
    return out, stats


# ── fixed-identity assignment: at most N people for the whole session ─────────
# A box may bind to a slot only if its displacement since the slot's last
# trusted position is physically plausible.  allowed(gap) = max(JITTER_M,
# max_speed * gap_seconds).  This is the "similarity threshold": it is small
# for near-consecutive frames (0.4 m kills 1-frame 4-13 m sensor spikes, the
# suspicious "jumps to the edge") and grows with time, so a person who walks
# across the room while undetected is still re-associated.
JITTER_M = 0.4


def _allowed_disp(gap_frames, max_speed):
    return max(JITTER_M, max_speed * gap_frames * 0.1)


def assign_people(npz, n_people=7, min_score=0.4, gap_cost=0.05,
                  max_interp=9, max_speed=4.0):
    """
    Whole-session identity assignment with a FIXED pool of n_people slots.

    - every frame: 1-to-1 boxes -> slots,
        * occupied slot: binds only if xy-distance <= allowed(gap) (similarity
          gate); a box that fails every gate may take an UNUSED slot (a new
          person), else it is dropped as clutter
        * the slot's anchor stays at its last TRUSTED position on rejection,
          so a later genuine detection re-binds against it
    - trajectory build: between two trusted observations, gap <= max_interp
          -> positions LINEARLY INTERPOLATED (column I)
                      gap >  max_interp -> counted as MISSING frames (MIS)
                      displacement > allowed(gap) -> observation REJECTED,
          the two trusted neighbours are interpolated between instead (JMP)
    - slots never die (a person may leave and re-enter the scene)
    - matching/display use xy + box-height-rested z (constant model z-bias
      removed: z_disp = h/2, ground ~ 0)
    """
    n = len(npz["frame_files"])
    K = max(len(b) for b in npz["pred_boxes"])
    BOX = np.zeros((n, K, 7), dtype=np.float32)
    LAB = np.full((n, K), -1, dtype=np.int64)
    SCL = np.zeros((n, K), dtype=np.float32)
    for i in range(n):
        b, l, s = np.asarray(npz["pred_boxes"][i]), np.asarray(npz["pred_labels"][i]), np.asarray(npz["pred_scores"][i])
        k = len(b)
        if k:
            BOX[i, :k] = b
            LAB[i, :k] = l
            SCL[i, :k] = s

    def ped_boxes(fi):
        keep = (LAB[fi] == PED_LABEL)
        if min_score > 0:
            keep &= SCL[fi] >= min_score
        k = int(keep.sum())
        if k == 0:
            return None
        b, s = BOX[fi][keep], SCL[fi][keep]
        if k > n_people:  # at most n_people people at any moment; rest = clutter
            o = np.argsort(-s)[:n_people]
            b, s = b[o], s[o]
        return b, s

    slots = [{"f": None, "xy": None} for _ in range(n_people)]
    obs = [[] for _ in range(n_people)]
    unassigned = 0

    for fi in range(n):
        got = ped_boxes(fi)
        if got is None:
            continue
        C, S = got
        k = len(C)

        # ── phase 1: bind to occupied slots, similarity-gated (xy) ──
        pairs = []
        for si in range(n_people):
            sl = slots[si]
            if sl["xy"] is None:
                continue
            gap = max(1, fi - sl["f"])
            allow = _allowed_disp(gap, max_speed)
            dx = C[:, 0] - sl["xy"][0]
            dy = C[:, 1] - sl["xy"][1]
            d = np.hypot(dx, dy)
            for j in range(k):
                if d[j] <= allow:
                    pairs.append((float(d[j]) + 0.01 * (1.0 - float(S[j])), si, j))
        pairs.sort(key=lambda p: p[0])
        used_s, used_c = set(), set()
        for _, si, j in pairs:
            if si in used_s or j in used_c:
                continue
            used_s.add(si); used_c.add(j)
            p = C[j]
            obs[si].append((fi, np.array([p[0], p[1], p[5] / 2.0], dtype=np.float32)))
            slots[si] = {"f": fi, "xy": np.array([p[0], p[1]], dtype=np.float32)}

        # ── phase 2: left-over boxes take UNUSED slots (new persons) ──
        free = [si for si in range(n_people) if slots[si]["xy"] is None]
        left = [j for j in range(k) if j not in used_c]
        left.sort(key=lambda j: -float(S[j]))
        for si, j in zip(free, left):
            p = C[j]
            obs[si].append((fi, np.array([p[0], p[1], p[5] / 2.0], dtype=np.float32)))
            slots[si] = {"f": fi, "xy": np.array([p[0], p[1]], dtype=np.float32)}
            used_c.add(j)

        unassigned += k - len(used_c)

    def finish(o):
        pts, ni, nmis, njumps = [], 0, 0, 0
        anchor = None  # (frame, xyz) of last TRUSTED observation
        for fi, p in o:
            if anchor is not None:
                af, a = anchor
                gap = fi - af      # frame span of the hop (>= 1)
                miss = gap - 1     # frames missing in between
                d = float(np.hypot(p[0] - a[0], p[1] - a[1]))
                if d > _allowed_disp(gap, max_speed):
                    # implausible jump (likely a spurious edge detection):
                    # drop it; the next trusted detection interpolates against
                    # the unchanged anchor (trusted-to-trusted only)
                    njumps += 1
                    continue
                if 1 <= miss <= max_interp:
                    for g in range(1, gap):  # fill af+1 .. fi-1
                        t = g / gap
                        cp = a * (1.0 - t) + p * t
                        pts.append([af + g, float(cp[0]), float(cp[1]), float(cp[2]), "i"])
                    ni += miss
                elif miss > max_interp:
                    nmis += miss
            pts.append([fi, float(p[0]), float(p[1]), float(p[2]), "m"])
            anchor = (fi, p)
        if not pts:
            return None
        frames = [p[0] for p in pts]
        return {
            "tid": 0, "start": frames[0], "end": frames[-1],
            "last": max(p[0] for p in pts if p[4] == "m"),
            "len": sum(1 for p in pts if p[4] == "m"),
            "ni": ni, "nm": nmis, "nj": njumps,
            "st": "M" if nmis >= 10 else ("I" if ni > 0 else "OK"),
            "pts": pts,
        }

    used = [(si, sorted(obs[si])) for si in range(n_people) if obs[si]]
    used.sort(key=lambda so: -(len(so[1])))
    tracks = []
    for rank, (si, o) in enumerate(used, start=1):
        t = finish(o)
        if t is None:
            continue
        t["tid"] = rank
        tracks.append(t)

    stats = {
        "n_tracks": len(tracks),
        "n_slots": n_people,
        "ok": sum(1 for t in tracks if t["st"] == "OK"),
        "interp": sum(1 for t in tracks if t["st"] == "I"),
        "missing": sum(1 for t in tracks if t["st"] == "M"),
        "n_interp_frames": sum(t["ni"] for t in tracks),
        "n_missing_frames": sum(t["nm"] for t in tracks),
        "n_jumps": sum(t["nj"] for t in tracks),
        "n_matched_frames": sum(t["len"] for t in tracks),
        "n_unassigned": unassigned,
        "longest": max((t["len"] + t["ni"] for t in tracks), default=0),
    }
    return tracks, stats


# ── http ──────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    server_version = "ReIDTracks/1.0"

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
            t = {"js": "text/javascript; charset=utf-8",
                 "html": "text/html; charset=utf-8"}.get(p.rsplit(".", 1)[-1], "application/octet-stream")
            self._send(200, open(p, "rb").read(), t)
        else:
            self._send(404, b"not found", "text/plain")

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        if path == "/":
            self._static("index.html", WEB_DIR)
        elif path in ("/app.js", "/index.html"):
            self._static(path.lstrip("/"), WEB_DIR)
        elif path.startswith("/vendor/"):
            self._static(path[len("/vendor/"):], VEND_DIR)
        elif path == "/api/tracks":
            S = STATE
            self._send(200, {"session": S["session"], "n_frames": len(S["npz"]["frame_files"]),
                             "params": S["params"], "stats": S["stats"], "tracks": S["tracks"]})
        elif path == "/api/framecloud":
            S = STATE
            fi = int(parse_qs(u.query).get("fi", ["0"])[0])
            if not (0 <= fi < len(S["npz"]["frame_files"])):
                self._send(400, {"error": "bad frame index"})
                return
            if fi not in S["frame_cache"]:
                if len(S["frame_cache"]) > 24:
                    S["frame_cache"].pop(next(iter(S["frame_cache"])))
                pts = np.load(os.path.join(S["frames_dir"], str(S["npz"]["frame_files"][fi])))
                S["frame_cache"][fi] = np.ascontiguousarray(pts[:, :3].astype("<f4"))
            a = S["frame_cache"][fi]
            self._send(200, {"fi": fi,
                             "pts_b64": base64.b64encode(a.tobytes()).decode()})
        else:
            self._send(404, b"not found", "text/plain")


def main():
    import sys
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="2026-07-29_17-21-48")
    ap.add_argument("--lidar-dir", default="~/Projects/Thesis/Lidar Data")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--min-score", type=float, default=0.4,
                    help="pedestrian detection confidence cut-off (higher = cleaner, fewer)")
    ap.add_argument("--gate", type=float, default=0.8,
                    help="Max centre displacement per 100 ms frame (m); widens x gap when bridging")
    ap.add_argument("--max-bridge", type=int, default=9,
                    help="Interpolate gaps up to this many frames; >= max_bridge+1 => MISSING")
    ap.add_argument("--people", type=int, default=0,
                    help=">0: FIXED identity pool of this many people for the whole "
                         "session (max 7 = 'at most 7 people in the room'); 0 = legacy "
                         "greedy per-frame tracking")
    ap.add_argument("--gap-cost", type=float, default=0.05,
                    help="cost per missed frame (m) used by --people assignment")
    ap.add_argument("--max-speed", type=float, default=4.0,
                    help="similarity threshold: max plausible person speed (m/s); "
                         "a box farther than max(0.4, max_speed*gap_s) from its slot's "
                         "last trusted position is rejected (kills jumps-to-the-edge)")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()

    LIDAR = os.path.expanduser(args.lidar_dir)
    labels_path = args.labels or os.path.join(LIDAR, f"{args.session}_frames_voxelnext.npz")
    frames_dir = args.frames_dir or os.path.join(LIDAR, "frames", args.session)
    if not os.path.exists(labels_path):
        sys.exit(f"labels npz not found: {labels_path}")
    if not os.path.isdir(frames_dir):
        sys.exit(f"frames dir not found: {frames_dir}")

    S = STATE
    S["session"], S["frames_dir"] = args.session, frames_dir
    S["npz"] = np.load(labels_path, allow_pickle=True)
    t0 = time.time()
    if args.people > 0:
        S["tracks"], S["stats"] = assign_people(
            S["npz"], args.people, args.min_score, args.gap_cost,
            args.max_bridge, args.max_speed)
    else:
        S["tracks"], S["stats"] = track_session(
            S["npz"], args.min_score, args.gate, args.max_bridge)
    S["params"] = {"min_score": args.min_score, "gate": args.gate,
                   "max_bridge": args.max_bridge, "people": args.people,
                   "gap_cost": args.gap_cost, "max_speed": args.max_speed}
    st = S["stats"]
    print(f"\n[tracks] {S['session']}  ({time.time()-t0:.1f}s)  "
          f"mode={'fixed-people x' + str(args.people) if args.people else 'greedy'}")
    print(f"[tracks] {st['n_tracks']} tracks: {st['ok']} OK, {st['interp']} w/interp, {st['missing']} MISSING")
    print(f"[tracks] frames: matched={st['n_matched_frames']:,}  interpolated={st['n_interp_frames']:,}  "
          f"missing={st['n_missing_frames']:,}"
          + (f"  rejected-jumps={st.get('n_jumps', 0):,}  dropped-as-clutter={st.get('n_unassigned', 0):,}"
             if args.people else ""))
    for t in S["tracks"][:10]:
        print(f"   T{t['tid']:<4} frames {t['start']:>4}-{t['last']:>4}  "
              f"len={t['len']:<4} I={t['ni']:<4} MISS={t['nm']:<4} JMP={t.get('nj', 0):<3} {t['st']}")

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n[web] serving on  http://localhost:{args.port}   (Ctrl-C to stop)\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopping")


if __name__ == "__main__":
    main()
