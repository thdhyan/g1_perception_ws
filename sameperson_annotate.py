#!/usr/bin/env python3
"""
sameperson_annotate.py — ReID annotator for CONSECUTIVE-FRAME person pairs.

For a LiDAR session + its per-frame detection label file (VoxelNeXt), this tool
builds candidate pairs of the same detected person across two consecutive
frames (greedy nearest-centre matching) and shows both 3D point clouds side
by side, so you can judge: SAME person? or DIFFERENT?

Data expected (1-indexed label convention, 2 = Pedestrian):
  <session>_frames_*.npz  → frame_files, pred_boxes (M,7), pred_scores, pred_labels
  frames/<session>/frame_%05d.npy   (N,4) float32 [x, y, z, intensity]
Point clouds are in the z-up world frame (ground ≈ 0). VoxelNeXt box z carries
a constant model offset — boxes are auto-rested on the local floor for display
only; x/y and matching are unaffected.

Controls:
  Y / y    — same person
  N / n    — different person
  Space    — skip pair
  Z / z    — toggle close-up (±3 m) / overview (±8 m)
  Q / q    — quit & save   (progress is saved after every verdict)

Output: reid_data/sameperson_<session>.json  (resumable)
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ── fix: Debian nspkg.pth pre-registers the SYSTEM mpl_toolkits (3.6.3) into
# sys.modules at interpreter STARTUP. It must be cleared BEFORE matplotlib is
# imported, otherwise Axes3D loads from the wrong matplotlib and 3D panels fail.
import sys
sys.path = [p for p in sys.path if 'dist-packages' not in p]
for _k in list(sys.modules.keys()):
    if _k == 'mpl_toolkits' or _k.startswith('mpl_toolkits.'):
        del sys.modules[_k]
# ──────────────────────────────────────────────────────────────────────────────

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

WS       = Path(__file__).resolve().parent
ANN_DIR  = WS / "reid_data"
SUBSAMPLE = 4000          # points plotted per panel
OVERVIEW_R = 8.0          # m
CLOSEUP_R  = 3.0          # m
FLOOR_LO, FLOOR_HI = 0.0, 0.6   # z-bucket used to estimate the local floor


# ── 3D box wireframe helpers ──────────────────────────────────────────────────
def box_corners(cx, cy, cz, dx, dy, dz, yaw):
    hw, hh, ht = dx / 2, dy / 2, dz / 2
    c, s = np.cos(yaw), np.sin(yaw)
    lc = np.array([
        [ hw,  hh,  ht], [ hw, -hh,  ht], [-hw, -hh,  ht], [-hw,  hh,  ht],
        [ hw,  hh, -ht], [ hw, -hh, -ht], [-hw, -hh, -ht], [-hw,  hh, -ht],
    ])
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (lc @ rot.T) + np.array([cx, cy, cz])


def box_edges(corners):
    idx = [(0,1),(1,2),(2,3),(3,0),
           (4,5),(5,6),(6,7),(7,4),
           (0,4),(1,5),(2,6),(3,7)]
    return [(corners[a], corners[b]) for a, b in idx]


def rest_box_on_floor(z_center: float, h: float, floor_z: float) -> float:
    """VoxelNeXt box z carries a constant model bias; drop/raise the box so its
    bottom sits on `floor_z` (display only — never written back)."""
    return floor_z + h / 2.0


# ── pair construction ─────────────────────────────────────────────────────────
def greedy_match(boxes_a, boxes_b, max_dist):
    """Nearest-centre greedy matching → list of (i, j) disjoint pairs."""
    pairs, used_a, used_b = [], set(), set()
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return pairs
    c_a, c_b = boxes_a[:, :3], boxes_b[:, :3]
    d = np.linalg.norm(c_a[:, None, :] - c_b[None, :, :], axis=-1)
    order = np.argsort(d, axis=None)
    for flat in order:
        i, j = int(flat // len(boxes_b)), int(flat % len(boxes_b))
        if d[i, j] >= max_dist:
            break
        if i not in used_a and j not in used_b:
            pairs.append((i, j, float(d[i, j])))
            used_a.add(i); used_b.add(j)
    return pairs


def build_pairs(npz, ped_label, max_dist):
    """Candidate pairs: person in frame f vs its nearest match in f+1 (within
    `max_dist`) and f+2 (within 1.6x — detections flicker at 10 Hz)."""
    np_ = npz
    per_frame_ped = []      # (M, 7) peds per frame, or empty (0,7)
    for fi in range(len(np_["frame_files"])):
        boxes  = np_["pred_boxes"][fi]
        labels = np_["pred_labels"][fi]
        mask   = labels == ped_label
        per_frame_ped.append(np.asarray(boxes[mask], dtype=np.float32))

    pairs = []
    for fi in range(len(per_frame_ped)):
        ba = per_frame_ped[fi]
        if len(ba) == 0:
            continue
        taken = set()
        for gap, limit in ((1, max_dist), (2, max_dist * 1.6)):
            b1 = per_frame_ped[fi + gap] if fi + gap < len(per_frame_ped) else None
            if b1 is None or len(b1) == 0:
                continue
            for i, j, dist in greedy_match(ba, b1, limit):
                if i in taken:
                    continue
                taken.add(i)
                pairs.append({"f0": fi, "k0": i, "f1": fi + gap, "k1": j,
                              "dist": dist, "gap": gap})
    return pairs


def frame_data(npz, frames_dir, fi):
    fname = str(npz["frame_files"][fi])
    pts   = np.load(os.path.join(frames_dir, fname))
    boxes   = npz["pred_boxes"][fi]
    labels  = npz["pred_labels"][fi]
    scores  = npz["pred_scores"][fi]
    peds    = np.asarray(boxes[labels == PED_LABEL], dtype=np.float32)
    peds_sc = scores[labels == PED_LABEL]
    return pts, peds, peds_sc


def local_floor_z(pts, x, y, r=1.0):
    """Median z of the lower 20 % of points near (x, y)."""
    m = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2 <= r * r
    if m.sum() < 20:
        m = np.ones(len(pts), bool)
    z = pts[m, 2]
    z = z[z < np.percentile(z, 80)]
    return float(np.median(z)) if len(z) else 0.0


# ── plotting ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#0D1117",
    "axes.edgecolor":   "#30363D",
    "text.color":       "#C8D4EE",
    "xtick.color":      "#5A6985",
    "ytick.color":      "#5A6985",
    "grid.color":       "#1E2940",
    "grid.linewidth":   0.4,
})

def draw_panel(ax, pts, peds, peds_sc, hi_idx, radius, label, color_hi):
    ax.cla()
    ax.set_facecolor("#0D1117")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), min(SUBSAMPLE, len(pts)), replace=False)
    p = pts[idx]
    z = p[:, 2]
    zn = (z - z.min()) / (z.ptp() + 1e-6)
    colors = plt.cm.viridis(zn)
    colors[:, 3] = 0.35
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, s=1.2, depthshade=False)

    floor_z = local_floor_z(pts, peds[hi_idx, 0], peds[hi_idx, 1]) if len(peds) else 0.0
    for bi, box in enumerate(peds):
        cx, cy, cz, dx, dy, dz, yaw = box
        czv = rest_box_on_floor(cz, dz, floor_z)
        corners = box_corners(cx, cy, czv, dx, dy, dz, yaw)
        segs = [np.array([a, b]) for a, b in box_edges(corners)]
        hot = bi == hi_idx
        ax.add_collection3d(Line3DCollection(
            segs,
            colors=color_hi if hot else "#2A6FBF",
            linewidths=2.6 if hot else 0.9,
            alpha=1.0 if hot else 0.55,
        ))
        if hot:
            sc = peds_sc[bi] if len(peds_sc) else 0.0
            ax.text(cx, cy, czv + dz / 2 + 0.12, f"#{bi}  s={sc:.2f}",
                    color=color_hi, fontsize=8, ha="center")

    cx = pts[:, 0].mean(); cy = pts[:, 1].mean()
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_zlim(-0.4, 2.4)
    ax.set_xlabel("X", fontsize=7, labelpad=1)
    ax.set_ylabel("Y", fontsize=7, labelpad=1)
    ax.set_zlabel("Z ↑", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6)
    ax.set_title(label, color=color_hi, fontsize=9, pad=4)
    ax.grid(True, linewidth=0.3, alpha=0.4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="2026-08-05_16-38-40")
    ap.add_argument("--labels", default=None,
                    help="detection npz (default: <LIDAR>/<session>_frames_voxelnext.npz)")
    ap.add_argument("--frames-dir", default=None,
                    help="per-frame npy dir (default: <LIDAR>/frames/<session>)")
    ap.add_argument("--lidar-dir", default= "~/Projects/Thesis/Lidar Data")
    ap.add_argument("--ped-label", type=int, default=2)
    ap.add_argument("--max-dist", type=float, default=2.5,
                    help="max centre distance (m) for a consecutive-frame pair "
                         "(gap-1; gap-2 uses 1.6x this default 2.5)")
    ap.add_argument("--closeup", action="store_true",
                    help="start in ±3 m close-up mode")
    ap.add_argument("--out", default=None, help="annotation json path")
    args = ap.parse_args()

    global PED_LABEL
    PED_LABEL = args.ped_label

    LIDAR = Path(os.path.expanduser(args.lidar_dir))
    labels_path = Path(os.path.expanduser(args.labels)) if args.labels else \
        LIDAR / f"{args.session}_frames_voxelnext.npz"
    frames_dir = Path(os.path.expanduser(args.frames_dir)) if args.frames_dir else \
        LIDAR / "frames" / args.session
    ann_path = Path(os.path.expanduser(args.out)) if args.out else \
        ANN_DIR / f"sameperson_{args.session}.json"

    for req, name in [(labels_path, "labels npz"), (frames_dir, "frames dir")]:
        if not req.exists():
            sys.exit(f"{name} not found: {req}")

    npz = np.load(labels_path, allow_pickle=True)
    pairs = build_pairs(npz, PED_LABEL, args.max_dist)
    print(f"[pairs] {len(pairs)} consecutive-frame candidate pairs "
          f"({args.max_dist} m window)  →  {ann_path}")
    if not pairs:
        sys.exit("No ped pairs found — check --max-dist / score coverage.")

    answered = {}
    if ann_path.exists():
        prev = json.loads(ann_path.read_text())
        for a in prev.get("annotations", []):
            answered[(a["f0"], a["k0"], a["f1"], a["k1"])] = a
        print(f"[resume] {len(answered)} already answered")

    counts = {"yes": 0, "no": 0, "skip": 0}
    for a in answered.values():
        counts[a["verdict"]] = counts.get(a["verdict"], 0) + 1

    def save():
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        ann_path.write_text(json.dumps({
            "session": args.session, "labels_file": str(labels_path),
            "max_dist_m": args.max_dist, "pairs_total": len(pairs),
            "counts": counts, "annotations": list(answered.values()),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))

    todo = [p for p in pairs
            if (p["f0"], p["k0"], p["f1"], p["k1"]) not in answered]

    if not todo:
        print("[done] all pairs already answered.")
        return

    # ── GUI ────────────────────────────────────────────────────────────────────
    verdict_holder = [None]
    lock = threading.Event()
    radius_mode = [CLOSEUP_R if args.closeup else OVERVIEW_R]
    cur = {"j": None}
    redraw = [lambda: None]

    fig = plt.figure(figsize=(15, 6.4))
    fig.patch.set_facecolor("#0D1117")
    ax_a = fig.add_subplot(121, projection="3d")
    ax_b = fig.add_subplot(122, projection="3d")
    st_ax = fig.add_axes([0, 0, 1, 0.055])
    st_ax.set_facecolor("#161B22"); st_ax.axis("off")
    st_txt = st_ax.text(0.5, 0.5, "Loading…", ha="center", va="center",
                        color="#8B949E", fontsize=9, fontfamily="monospace",
                        transform=st_ax.transAxes)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.075)

    def render(j):
        p = todo[j]
        pts0, peds0, sc0 = frame_data(npz, frames_dir, p["f0"])
        pts1, peds1, sc1 = frame_data(npz, frames_dir, p["f1"])
        draw_panel(ax_a, pts0, peds0, sc0, p["k0"], radius_mode[0],
                   f"FRAME {p['f0']}   person #{p['k0']}   "
                   f"(score {sc0[p['k0']]:.2f})", "#F9C74F")
        draw_panel(ax_b, pts1, peds1, sc1, p["k1"], radius_mode[0],
                   f"FRAME {p['f1']}  (+{p['gap']*100:.0f} ms)   "
                   f"person #{p['k1']}   (score {sc1[p['k1']]:.2f})", "#43D9AD")
        st_txt.set_text(
            f"Pair {j+1}/{len(todo)}  (frame {p['f0']}→{p['f1']}, "
            f"+{p['gap']*100:.0f} ms, centre dist {p['dist']:.2f} m)   "
            f"✓ same={counts['yes']}   ✗ diff={counts['no']}   → skipped={counts['skip']}   "
            f"[Y] same person   [N] different   [Space] skip   [Z] zoom   [Q] quit"
        )
        fig.canvas.draw()
        fig.canvas.flush_events()
        redraw[0] = (lambda: render(cur["j"])) if cur["j"] is not None else None

    def on_key(event):
        k = event.key
        if k in ("y", "Y"):   verdict_holder[0] = "yes";  lock.set()
        elif k in ("n", "N"): verdict_holder[0] = "no";   lock.set()
        elif k == " ":        verdict_holder[0] = "skip"; lock.set()
        elif k in ("z", "Z"):
            radius_mode[0] = CLOSEUP_R if radius_mode[0] > 1 else OVERVIEW_R
            redraw[0]()
        elif k in ("q", "Q"): verdict_holder[0] = "quit"; lock.set()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.ion()
    plt.show()

    for j, p in enumerate(todo):
        cur["j"] = j
        render(j)
        verdict_holder[0] = None
        lock.clear()
        lock.wait()

        v = verdict_holder[0]
        if v == "quit":
            print("\n[save & quit]")
            save()
            break
        if v is None:
            continue
        rec = {**p, "verdict": v, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        answered[(p["f0"], p["k0"], p["f1"], p["k1"])] = rec
        counts[v] = counts.get(v, 0) + 1
        save()

    plt.ioff()
    print(f"\nDone. counts={counts}")
    if plt.get_fignums():
        plt.close("all")
    save()


if __name__ == "__main__":
    matplotlib.use("TkAgg")   # switch to Qt5Agg if Tk fails
    main()
