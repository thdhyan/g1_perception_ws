#!/usr/bin/env python3
"""
ReID Pair Annotator — full-frame context with Open3D / matplotlib.

Shows the complete LiDAR frame for each pair with:
  - Full point cloud (grey, coloured by height)
  - All pedestrian boxes (blue wireframe)
  - The two compared persons highlighted (A=yellow, B=cyan)

Controls:
  Y / y  — same person  ✓
  N / n  — different    ✗
  Space  — skip
  Q / q  — quit & save

Annotations saved to reid_data/annotations.json  (resumable).
Correct transform applied: Z_display = -Z_raw  (inverted mount).
"""

import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

# ── fix: system dist-packages mpl_toolkits conflicts with venv matplotlib ─────
import matplotlib                      # load venv matplotlib first
sys.path = [p for p in sys.path if 'dist-packages' not in p]
for _k in list(sys.modules.keys()):
    if 'mpl_toolkits' in _k:
        del sys.modules[_k]
# ──────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D

# ── paths ─────────────────────────────────────────────────────────────────────
LIDAR_DATA = Path("/home/thakk100/Projects/Thesis/Lidar Data")
REID_DATA   = Path("/home/thakk100/Projects/thesis/g1_perception_ws/reid_data")
ANN_FILE    = REID_DATA / "annotations.json"
SESSIONS    = [
    "2026-08-05_16-38-40",
    "2026-08-05_16-59-33",
    "2026-08-05_17-00-24",
]
PED_LABEL = 2

# ── robot transform ────────────────────────────────────────────────────────────
# Livox Mid-360 is mounted inverted: sensor +Z points DOWN.
# Flip Z so +Z_display points UP (world convention).
def transform_pts(pts):
    """(N,4) raw → (N,3) display coords with +Z = up."""
    out = pts[:, :3].copy()
    out[:, 2] = -out[:, 2]
    return out

def transform_box(box7):
    """box7 = [cx,cy,cz, dx,dy,dz, yaw] in raw sensor frame → display frame."""
    b = box7.copy()
    b[2] = -b[2]   # flip z center
    # yaw convention unchanged (rotation around Z stays the same when Z flips
    # and X,Y unchanged — but heading sign flips; visualise both and check)
    return b

# ── box wireframe ──────────────────────────────────────────────────────────────
def box_corners(b):
    """Return (8,3) corner coords for a 7-DOF box in display frame."""
    cx, cy, cz, dx, dy, dz, yaw = b
    hw, hh, ht = dx/2, dy/2, dz/2
    c, s = np.cos(yaw), np.sin(yaw)
    # local corners
    lc = np.array([
        [ hw,  hh,  ht], [ hw, -hh,  ht], [-hw, -hh,  ht], [-hw,  hh,  ht],
        [ hw,  hh, -ht], [ hw, -hh, -ht], [-hw, -hh, -ht], [-hw,  hh, -ht],
    ])
    rot = np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]])
    return (lc @ rot.T) + np.array([cx, cy, cz])

def box_edges(corners):
    """Return list of (2,3) line segments for 3D wireframe."""
    idx = [(0,1),(1,2),(2,3),(3,0),  # top face
           (4,5),(5,6),(6,7),(7,4),  # bottom face
           (0,4),(1,5),(2,6),(3,7)]  # verticals
    return [(corners[a], corners[b]) for a, b in idx]

# ── data loading ───────────────────────────────────────────────────────────────
pairs    = np.load(REID_DATA / "temporal_pairs.npy")   # (M, 2)
pids     = np.load(REID_DATA / "pseudo_ids.npy")
sess_ids = np.load(REID_DATA / "session_ids.npy")
frame_ids= np.load(REID_DATA / "frame_ids.npy")

npz_cache = {}
def get_npz(sess_idx):
    if sess_idx not in npz_cache:
        s = SESSIONS[sess_idx]
        matches = sorted(LIDAR_DATA.glob(f"{s}_frames_*.npz"))
        pref = [m for m in matches if m.name.endswith("_voxelnext.npz")]
        path = (pref or matches)[0] if matches else None
        if path is None:
            raise FileNotFoundError(f"no label npz matching {LIDAR_DATA}/{s}_frames_*.npz")
        npz_cache[sess_idx] = np.load(str(path), allow_pickle=True)
    return npz_cache[sess_idx]

def get_frame(crop_idx):
    """Return (pts_display, ped_boxes_display, ped_scores) for a crop index."""
    si = int(sess_ids[crop_idx])
    fi = int(frame_ids[crop_idx])
    npz = get_npz(si)
    fname = str(npz['frame_files'][fi])
    pts_raw = np.load(LIDAR_DATA / "frames" / SESSIONS[si] / fname)
    pts_disp = transform_pts(pts_raw)

    boxes_all  = npz['pred_boxes'][fi]   # (K, 7)
    labels_all = npz['pred_labels'][fi]  # (K,)
    scores_all = npz['pred_scores'][fi]  # (K,)

    ped_mask   = labels_all == PED_LABEL
    ped_boxes  = np.array([transform_box(b) for b in boxes_all[ped_mask]])
    ped_scores = scores_all[ped_mask]

    # index of this specific crop within pedestrian detections
    crop_box_idx = np.sum(ped_mask[:np.where(ped_mask)[0][  # find which ped this is
        # We stored crops in order of iteration; need to find match by proximity
        # Use crop center from crops.npy to find nearest box
        0]])  # placeholder — see below

    return pts_disp, ped_boxes, ped_scores, si, fi

def find_crop_box_idx(crop_idx, ped_boxes):
    """Find which pedestrian box this crop corresponds to using stored crop data."""
    crops = np.load(REID_DATA / "crops.npy", mmap_mode='r')
    crop_centre = crops[crop_idx].mean(axis=0)   # (3,) in crop's canonical frame
    # crop is centered at bbox center (after yaw undo), but still in sensor Z convention
    # so crop centre ≈ (0,0,0); instead match by session+frame order
    # The crops are stored in order per frame, so crop_idx within a frame = ped box order
    return None   # not critical for visualisation; highlight all peds + mark the pair

# ── annotation state ───────────────────────────────────────────────────────────
ann = []
counts = {"yes": 0, "no": 0, "skip": 0}
start_idx = 0

if ANN_FILE.exists():
    try:
        saved = json.loads(ANN_FILE.read_text())
        ann    = saved.get("annotations", [])
        counts = saved.get("counts", counts)
        start_idx = len(ann)
        print(f"Resumed from annotation {start_idx}/{len(pairs)}")
    except Exception:
        pass

def save():
    ANN_FILE.parent.mkdir(parents=True, exist_ok=True)
    ANN_FILE.write_text(json.dumps({"counts": counts, "annotations": ann}, indent=2))

# ── plotting ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0D1117",
    "axes.facecolor":    "#0D1117",
    "axes.edgecolor":    "#30363D",
    "text.color":        "#C8D4EE",
    "xtick.color":       "#5A6985",
    "ytick.color":       "#5A6985",
    "grid.color":        "#1E2940",
    "grid.linewidth":    0.4,
})

SUBSAMPLE = 4000   # max points to plot per frame (speed)

def draw_frame(ax, pts, ped_boxes, ped_scores, highlight_box_idx, label, color_hi):
    ax.cla()
    ax.set_facecolor("#0D1117")

    # subsample for speed
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), min(SUBSAMPLE, len(pts)), replace=False)
    p = pts[idx]

    # colour by Z (height)
    z = p[:, 2]
    zn = (z - z.min()) / (z.ptp() + 1e-6)
    colors = plt.cm.viridis(zn)
    colors[:, 3] = 0.35   # semi-transparent

    ax.scatter(p[:,0], p[:,1], p[:,2], c=colors, s=1.2, depthshade=False)

    # all ped boxes — dim blue wireframe
    for bi, box in enumerate(ped_boxes):
        corners = box_corners(box)
        edges   = box_edges(corners)
        segs    = [np.array([a, b]) for a, b in edges]   # each seg: (2,3)
        lc = Line3DCollection(segs,
                              colors=color_hi if bi == highlight_box_idx else "#2A6FBF",
                              linewidths=2.5 if bi == highlight_box_idx else 0.8,
                              alpha=1.0 if bi == highlight_box_idx else 0.6)
        ax.add_collection3d(lc)

    ax.set_xlabel("X", fontsize=7, labelpad=1)
    ax.set_ylabel("Y", fontsize=7, labelpad=1)
    ax.set_zlabel("Z ↑", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6)
    ax.set_title(label, color=color_hi, fontsize=9, pad=4)

    # equal-ish aspect — fix Z to reasonable range
    ax.set_zlim(-0.5, 2.5)
    xy_range = 8
    cx = pts[:,0].mean(); cy = pts[:,1].mean()
    ax.set_xlim(cx - xy_range, cx + xy_range)
    ax.set_ylim(cy - xy_range, cy + xy_range)
    ax.grid(True, linewidth=0.3, alpha=0.4)

verdict_holder = [None]
lock = threading.Event()

def on_key(event):
    k = event.key
    if k in ('y', 'Y'):   verdict_holder[0] = 'yes';  lock.set()
    elif k in ('n', 'N'): verdict_holder[0] = 'no';   lock.set()
    elif k == ' ':         verdict_holder[0] = 'skip'; lock.set()
    elif k in ('q', 'Q'): verdict_holder[0] = 'quit'; lock.set()

def run():
    global start_idx, ann, counts

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")
    fig.canvas.mpl_connect('key_press_event', on_key)

    ax_a = fig.add_subplot(121, projection='3d')
    ax_b = fig.add_subplot(122, projection='3d')

    # status text
    status_ax = fig.add_axes([0, 0, 1, 0.06])
    status_ax.set_facecolor("#161B22")
    status_ax.axis('off')
    status_txt = status_ax.text(
        0.5, 0.5,
        "Loading...",
        ha='center', va='center',
        color='#8B949E', fontsize=9,
        fontfamily='monospace',
        transform=status_ax.transAxes
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08)
    plt.ion()
    plt.show()

    for pair_num in range(start_idx, len(pairs)):
        ci, cj = int(pairs[pair_num, 0]), int(pairs[pair_num, 1])

        # load frames
        si_a, fi_a = int(sess_ids[ci]), int(frame_ids[ci])
        si_b, fi_b = int(sess_ids[cj]), int(frame_ids[cj])
        npz_a = get_npz(si_a);  npz_b = get_npz(si_b)

        fname_a = str(npz_a['frame_files'][fi_a])
        fname_b = str(npz_b['frame_files'][fi_b])

        pts_a = transform_pts(np.load(LIDAR_DATA / "frames" / SESSIONS[si_a] / fname_a))
        pts_b = transform_pts(np.load(LIDAR_DATA / "frames" / SESSIONS[si_b] / fname_b))

        boxes_a_raw = npz_a['pred_boxes'][fi_a];  labels_a = npz_a['pred_labels'][fi_a]
        boxes_b_raw = npz_b['pred_boxes'][fi_b];  labels_b = npz_b['pred_labels'][fi_b]

        ped_boxes_a = np.array([transform_box(b) for b in boxes_a_raw[labels_a == PED_LABEL]])
        ped_boxes_b = np.array([transform_box(b) for b in boxes_b_raw[labels_b == PED_LABEL]])

        # which box in each frame corresponds to this crop? — heuristic: closest to
        # the stored crop center (we use frame order within pedestrians)
        crops = np.load(REID_DATA / "crops.npy", mmap_mode='r')
        # crop is centered; estimate its sensor-frame center by finding count of
        # crops before this one in the same frame
        def box_idx_for_crop(crop_idx, ped_boxes):
            """Find which ped box this crop came from using farm ordering in mining script."""
            same_frame = np.where((sess_ids == sess_ids[crop_idx]) &
                                  (frame_ids == frame_ids[crop_idx]) &
                                  (np.arange(len(frame_ids)) <= crop_idx))[0]
            order = len(same_frame) - 1  # 0-indexed within frame
            return min(order, len(ped_boxes) - 1) if len(ped_boxes) > 0 else 0

        hi_a = box_idx_for_crop(ci, ped_boxes_a)
        hi_b = box_idx_for_crop(cj, ped_boxes_b)

        pid_a, pid_b = int(pids[ci]), int(pids[cj])
        same_pid = pid_a == pid_b

        # draw
        draw_frame(ax_a, pts_a, ped_boxes_a, None, hi_a,
                   f"A  sess={si_a} frame={fi_a}  pid={pid_a}", "#F9C74F")
        draw_frame(ax_b, pts_b, ped_boxes_b, None, hi_b,
                   f"B  sess={si_b} frame={fi_b}  pid={pid_b}", "#43D9AD")

        pct = pair_num / len(pairs) * 100
        status_txt.set_text(
            f"Pair {pair_num+1}/{len(pairs)}  ({pct:.0f}%)   "
            f"tracker: {'same pid' if same_pid else 'diff pids'} ({pid_a} vs {pid_b})   "
            f"✓{counts['yes']}  ✗{counts['no']}  →{counts['skip']}   "
            f"[Y] same   [N] different   [Space] skip   [Q] quit"
        )
        fig.canvas.draw()
        fig.canvas.flush_events()

        # wait for keypress
        verdict_holder[0] = None
        lock.clear()
        lock.wait()

        v = verdict_holder[0]
        if v == 'quit':
            print("Saving and quitting...")
            save()
            break

        ann.append({"pair_num": pair_num, "i": ci, "j": cj,
                     "pid_i": pid_a, "pid_j": pid_b,
                     "same_tracker": same_pid, "verdict": v})
        counts[v] += 1
        save()

    plt.ioff()
    print(f"\nDone. {counts}")
    if plt.get_fignums():
        plt.close('all')

if __name__ == "__main__":
    matplotlib.use("TkAgg")  # change to Qt5Agg if TkAgg not available
    run()
