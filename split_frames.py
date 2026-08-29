#!/usr/bin/env python3
"""
split_frames.py — points.csv (x,y,z,reflectivity,timestamp_ns,frame_idx)
              → per-frame <out>/frame_%05d.npy   (N×4 float32, [x,y,z,intensity])

Streaming: one frame buffer at a time (assumes frame_idx is non-decreasing,
as in convert_lvx2.py's time-ordered output).
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np


def flush(buf, out_dir, fidx):
    if not buf:
        return 0
    a = np.asarray(buf, dtype=np.float32).reshape(-1, 4)
    np.save(os.path.join(out_dir, f"frame_{fidx:05d}.npy"), a)
    return len(buf)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    n_frames = 0
    total = 0
    cur_fidx = None
    buf = []
    order_ok = True

    with open(src, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        cols = {c: i for i, c in enumerate(header)}
        xi, yi, zi = cols["x"], cols["y"], cols["z"]
        ri = cols["reflectivity"]
        fi = cols["frame_idx"]
        for row in r:
            fv = int(row[fi])
            if cur_fidx is None:
                cur_fidx = fv
            elif fv != cur_fidx:
                if fv < cur_fidx:
                    order_ok = False
                n_frames += flush(buf, out_dir, cur_fidx)
                total += len(buf)
                buf = []
                cur_fidx = fv
            buf.append((float(row[xi]), float(row[yi]), float(row[zi]), int(row[ri])))
    if buf:
        n_frames += flush(buf, out_dir, cur_fidx)
        total += len(buf)

    print(f"{'!' if not order_ok else '✓'} frames={n_frames} points={total:,}"
          + ("" if order_ok else "  [WARNING: frame_idx not sorted — verify!]"))


if __name__ == "__main__":
    main()
