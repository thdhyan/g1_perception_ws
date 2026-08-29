#!/usr/bin/env python3
"""
reinfer_voxelnext.py — VoxelNeXt re-inference over a LiDAR session's per-frame .npy files.

Re-runs VoxelNeXt detection for a session and saves the boxes in the SAME
schema as the existing `<session>_frames_*.npz` label files so mine_reid_crops.py /
reid_annotate.py / the annotator can consume it unchanged:

    frame_files  (F,) str
    pred_boxes   (F,) object → each (M_i, 7) float32 [x,y,z,dx,dy,dz,yaw]
    pred_scores  (F,) object → each (M_i,) float32
    pred_labels  (F,) object → each (M_i,) int64  (1-indexed: 1=Car,2=Ped,3=Cyclist)
    class_names  ['Car','Pedestrian','Cyclist']

Usage:
    python3 reinfer_voxelnext.py \
        --session 2026-08-05_16-38-40 \
        [--frames-dir ~/Projects/Thesis/Lidar Data/frames/2026-08-05_16-38-40] \
        [--out ~/Projects/Thesis/Lidar Data/2026-08-05_16-38-40_frames_voxelnext.npz] \
        [--min-score 0.0] [--max-frames N]
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
warnings.filterwarnings("ignore")

ws = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# 1-indexed G1 convention (matches existing *_frames_*.npz label files):
#    0=car→1, 1=ped→2, 2=cyclist→3
G1_TO_1INDEXED = {0: 1, 1: 2, 2: 3}
CLASS_NAMES = np.array(["Car", "Pedestrian", "Cyclist"], dtype=object)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="Session name (dir under frames/)")
    ap.add_argument("--frames-dir", default=None,
                    help="Per-frame .npy dir (default: ~/Projects/Thesis/Lidar Data/frames/<session>)")
    ap.add_argument("--out", default=None,
                    help="Output npz (default: ~/Projects/Thesis/Lidar Data/<session>_frames_voxelnext.npz)")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="Drop boxes below this score before saving (default 0.0 = keep all)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    frames_dir = args.frames_dir or os.path.expanduser(
        f"~/Projects/Thesis/Lidar Data/frames/{args.session}")
    out_path = args.out or os.path.expanduser(
        f"~/Projects/Thesis/Lidar Data/{args.session}_frames_voxelnext.npz")
    if not os.path.isdir(frames_dir):
        sys.exit(f"frames dir not found: {frames_dir}")

    # ── model (defer ROS import; use the ROS package's standalone backend) ──
    vn = next(os.path.join(ws, d) for d in os.listdir(ws)
              if d.lower().startswith("voxel") and d.upper() != "VOXELKP"
              and os.path.isdir(os.path.join(ws, d, "pcdet")))
    import glob
    cfg_file = glob.glob(os.path.join(vn, "tools", "cfgs", "nuscenes_models",
                                       "cbgs_voxel0075_voxelnext.yaml"))[0]
    ckpt = os.path.join(ws, "pt", "voxelnext_nuscenes.pth")
    if not os.path.exists(ckpt):
        sys.exit(f"checkpoint not found: {ckpt}")

    sys.path.insert(0, vn)
    sys.path.insert(0, os.path.join(ws, "src", "livox_detection"))
    import logging
    logging.disable(logging.INFO)
    from livox_detection.voxelnext_model import VoxelNeXtBackend

    backend = VoxelNeXtBackend(
        checkpoint=ckpt, device="cuda", score_threshold=0.0,
        cfg_file=cfg_file, voxelnext_dir=vn)
    t0 = time.time()
    backend.load()
    print(f"[load] {time.time()-t0:.1f}s  ckpt={os.path.basename(ckpt)}")

    # ── iterate frames ───────────────────────────────────────────────────────
    frames = sorted(f for f in os.listdir(frames_dir)
                    if f.startswith("frame_") and f.endswith(".npy"))
    if args.max_frames:
        frames = frames[: args.max_frames]
    print(f"[run] {len(frames)} frames → {out_path}")

    all_files, all_boxes, all_scores, all_labels = [], [], [], []
    ped_frames, ped_boxes_total, t_start = 0, 0, time.time()

    for i, fname in enumerate(frames):
        pts = np.load(os.path.join(frames_dir, fname))
        boxes, scores, labels = backend.infer(pts)
        if len(boxes):
            keep = scores >= args.min_score
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            labels = np.array([G1_TO_1INDEXED.get(int(l), -1) for l in labels], dtype=np.int64)
            good = labels >= 0
            boxes, scores, labels = boxes[good], scores[good], labels[good]
        all_files.append(fname)
        all_boxes.append(np.ascontiguousarray(boxes.astype(np.float32)))
        all_scores.append(np.ascontiguousarray(scores.astype(np.float32)))
        all_labels.append(np.ascontiguousarray(labels))
        n_ped = int((labels == 2).sum())
        ped_frames += n_ped > 0
        ped_boxes_total += n_ped
        if (i + 1) % 100 == 0 or i + 1 == len(frames):
            el = time.time() - t_start
            print(f"  {i+1}/{len(frames)}  elapsed={el:.0f}s  "
                  f"ped_boxes_sofar={ped_boxes_total}  "
                  f"eta≈{el/(i+1)*(len(frames)-i-1):.0f}s", flush=True)

    np.savez(
        out_path,
        frame_files=np.array(all_files, dtype=object),
        pred_boxes=np.array(all_boxes, dtype=object),
        pred_scores=np.array(all_scores, dtype=object),
        pred_labels=np.array(all_labels, dtype=object),
        class_names=CLASS_NAMES,
    )
    el = time.time() - t_start
    print(f"\n[done] {len(frames)} frames in {el:.0f}s ({el/len(frames)*1000:.0f} ms/frame)")
    print(f"[done] frames_with_ped={ped_frames}/{len(frames)}  ped_boxes_total={ped_boxes_total}")
    print(f"[out]  {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
