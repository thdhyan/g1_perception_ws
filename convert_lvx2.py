#!/usr/bin/env python3
"""
convert_lvx2.py  —  LVX2 binary → CSV / NPZ

Tested against Livox Mid-360 recordings (SDK v2.x, data_type=2, int32 xyz mm).

Usage:
    python3 convert_lvx2.py <file.lvx2> [--out-dir /path/to/out] [--format csv|npz|both]

Output (csv):
    <stem>.csv  —  x,y,z (metres, robot frame),reflectivity,timestamp_ns,frame_idx

Output (npz):
    <stem>.npz  —  arrays: xyz (N,3) float32 m, refl (N,) uint8,
                            ts_ns (N,) int64, frame_idx (N,) int32

Robot transform applied:   Z_robot = −Z_sensor   (Livox Mid-360 inverted mount)
"""

import argparse, csv, struct, sys, time
from pathlib import Path
import numpy as np

# ── LVX2 constants ────────────────────────────────────────────────────────────
SIGNATURE      = b"livox_tech\x00\x00\x00\x00\x00\x00"
MAGIC          = 0xAC0EA767
PUBLIC_HDR_SZ  = 24   # sig(16)+ver(4)+magic(4)
PRIV_HDR_SZ    = 5    # frame_duration_ms(4)+device_count(1)
DEVICE_INFO_SZ = 43   # lidar_sn(16)+lidar_type(1)+dev_idx(1)+extrinsic_en(1)+float[6]
PKT_HDR_SZ     = 27   # empirically determined from binary pattern analysis
PTS_PER_PKT    = 96
PT_SZ          = 14   # int32 x,y,z + uint8 refl + uint8 tag
PKT_DATA_SZ    = PTS_PER_PKT * PT_SZ   # 1344 bytes
PKT_TOTAL_SZ   = PKT_HDR_SZ + PKT_DATA_SZ   # 1371 bytes

# In the 27-byte packet header, the LiDAR IP (c0 a8 7b 78 = 192.168.123.120)
# appears at byte offset 1 as a reliable packet-start marker.
# Timestamp (uint64 ns) is at byte offset 7 in the packet header.
PKT_IP_OFFSET  = 1    # offset of IP bytes within 27-byte header
PKT_TS_OFFSET  = 7    # offset of uint64 timestamp within 27-byte header

# The data segment starts after all headers; first packet header at some fixed offset.
# We locate it by scanning for the IP marker.
LIDAR_IP_BYTES = bytes([0xC0, 0xA8, 0x7B, 0x78])   # 192.168.123.120 default Mid-360 IP

def find_first_packet(data: bytes, search_from: int = 0) -> int:
    """Return byte offset of first packet header containing the LiDAR IP."""
    # Packet header layout: [??][IP:4][...26 more bytes]
    # Then 1344 bytes of point data follow.
    # We verify a candidate by checking that PKT_TOTAL_SZ later there's another IP marker.
    pos = search_from
    while True:
        idx = data.find(LIDAR_IP_BYTES, pos)
        if idx == -1 or idx < PKT_IP_OFFSET:
            return -1
        pkt_start = idx - PKT_IP_OFFSET
        # Verify: next packet header IP should be exactly PKT_TOTAL_SZ bytes later
        next_ip = pkt_start + PKT_TOTAL_SZ + PKT_IP_OFFSET
        if next_ip + 4 <= len(data) and data[next_ip:next_ip+4] == LIDAR_IP_BYTES:
            return pkt_start
        pos = idx + 1

def read_header(f) -> dict:
    """Read and validate LVX2 file header.  Returns metadata dict."""
    sig = f.read(16)
    if sig != SIGNATURE:
        raise ValueError(f"Not an LVX2 file (bad signature: {sig!r})")
    ver = f.read(4)
    magic = struct.unpack('<I', f.read(4))[0]
    if magic != MAGIC:
        raise ValueError(f"Bad magic 0x{magic:08X} (expected 0x{MAGIC:08X})")
    frame_dur_ms = struct.unpack('<I', f.read(4))[0]
    n_dev = f.read(1)[0]
    devices = []
    for _ in range(n_dev):
        sn_bytes   = f.read(16)
        lidar_type = f.read(1)[0]
        dev_idx    = f.read(1)[0]
        ext_en     = f.read(1)[0]
        extrinsics = struct.unpack('<ffffff', f.read(24))
        devices.append({
            'sn': sn_bytes.rstrip(b'\x00').decode('ascii', errors='replace'),
            'lidar_type': lidar_type,
            'dev_idx': dev_idx,
            'extrinsic_enable': ext_en,
            'extrinsics': extrinsics,  # roll,pitch,yaw,x,y,z
        })
    return {
        'version': list(ver),
        'frame_duration_ms': frame_dur_ms,
        'devices': devices,
        'data_start': f.tell(),
    }

def parse_lvx2(path: Path, verbose: bool = True, frame_ms: int = 100) -> dict:
    """
    Parse entire LVX2 file.  Returns:
      xyz_m     (N,3) float32 — x,y,z in metres, robot frame (Z inverted)
      refl      (N,)  uint8
      ts_ns     (N,)  int64  — nanosecond timestamp per point
      frame_idx (N,)  int32  — frame index per point
      meta      dict         — header metadata
    frame_ms: frame period in ms (Mid-360 = 100). The LVX2 header field
      'frame_duration_ms' is unreliable (reports 50 for these files).
    """
    path = Path(path)
    file_size = path.stat().st_size
    t0 = time.time()

    with open(path, 'rb') as f:
        meta = read_header(f)
        header_end = f.tell()

    # Memory-map for speed
    import mmap
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    # Locate first packet (scan from data_start with some tolerance)
    scan_buf = mm[header_end:header_end + 8192]
    first_pkt_rel = find_first_packet(scan_buf)
    if first_pkt_rel == -1:
        mm.close()
        raise RuntimeError("Could not locate first data packet — unknown LVX2 sub-format")
    first_pkt_abs = header_end + first_pkt_rel
    if verbose:
        print(f"  First packet at byte {first_pkt_abs}")

    # ── sequential packet extraction ─────────────────────────────────────────
    all_xyz  = []
    all_refl = []
    all_ts   = []

    pkt_count  = 0
    off        = first_pkt_abs
    file_bytes = len(mm)
    # Mid-360 true frame period is 100 ms (10 Hz). The LVX2 header field
    # 'frame_duration_ms' reports 50 — verified wrong against the Livox-
    # Viewer ground-truth frame counts for all three August sessions:
    #   16-38-40: 99.74 s → 998 frames (span/100ms ≈ 997 ✓; span/50ms = 1995 ✗)
    #   16-59-33: 19.12 s → 193 frames (span/100ms ≈ 191 ✓; span/50ms = 384 ✗)
    #   17-00-24:         → 257 frames (span/100ms ≈ 255 ✓; span/50ms = 513 ✗)
    frame_dur_ns = frame_ms * 1_000_000
    prev_ts_ns = None
    start_ts_ns = None
    frame_idx  = 0
    all_fidx   = []

    while off + PKT_TOTAL_SZ <= file_bytes:
        # Verify IP marker
        if mm[off + PKT_IP_OFFSET: off + PKT_IP_OFFSET + 4] != LIDAR_IP_BYTES:
            # re-sync: search forward
            rel = mm.find(LIDAR_IP_BYTES, off)
            if rel == -1:
                break
            off = rel - PKT_IP_OFFSET
            if off < first_pkt_abs or off + PKT_TOTAL_SZ > file_bytes:
                break
            continue

        # Timestamp from packet header
        ts_ns = struct.unpack_from('<Q', mm, off + PKT_TS_OFFSET)[0]

        # Frame binning by timestamp. Mid-360 timestamps run continuously
        # across frame boundaries, so a timestamp *jump* test never fires and
        # every point collapsed into frame 0. Instead, divide elapsed time by
        # the frame duration to get the frame index directly:
        #     frame_idx = (ts_ns - ts_ns[0]) // frame_dur_ns
        if start_ts_ns is None:
            start_ts_ns = ts_ns
        frame_idx = int((ts_ns - start_ts_ns) // frame_dur_ns)
        prev_ts_ns = ts_ns

        # Read 96 points from packet data area
        pt_off = off + PKT_HDR_SZ
        pt_end = pt_off + PKT_DATA_SZ
        raw = mm[pt_off:pt_end]

        # Unpack as (96, 14) byte records: int32 x,y,z, uint8 refl, uint8 tag
        pts = np.frombuffer(raw, dtype=np.dtype([
            ('x', '<i4'), ('y', '<i4'), ('z', '<i4'),
            ('refl', 'u1'), ('tag', 'u1')
        ]))

        # NOTE: all-zero points (Livox "no-return" placeholders) are KEPT,
        # matching Livox Viewer's official CSV exports (~35% of Mid-360 points
        # in these sessions). Downstream voxelizers (VoxelNeXt) drop them.
        # Filtering them here breaks point-count consistency with the
        # already-converted Livox CSVs.
        n_valid = len(pts)
        if n_valid > 0:
            # Convert mm → m, apply robot transform: Z_robot = -Z_sensor
            xyz = np.column_stack([
                pts['x'].astype(np.float32) / 1000.0,
                pts['y'].astype(np.float32) / 1000.0,
                -pts['z'].astype(np.float32) / 1000.0,  # ← Z flip
            ])
            all_xyz.append(xyz)
            all_refl.append(pts['refl'])
            all_ts.append(np.full(n_valid, ts_ns, dtype=np.int64))
            all_fidx.append(np.full(n_valid, frame_idx, dtype=np.int32))

        pkt_count += 1
        off += PKT_TOTAL_SZ

        if verbose and pkt_count % 5000 == 0:
            pct = off / file_bytes * 100
            elapsed = time.time() - t0
            print(f"  {pct:.1f}%  packets={pkt_count}  frames={frame_idx+1}  "
                  f"pts={sum(len(a) for a in all_xyz):,}  t={elapsed:.1f}s", end='\r')

    mm.close()

    if not all_xyz:
        raise RuntimeError("No valid points found in LVX2 file")

    xyz_m     = np.concatenate(all_xyz,  axis=0)
    refl      = np.concatenate(all_refl, axis=0)
    ts_ns_arr = np.concatenate(all_ts,   axis=0)
    fidx_arr  = np.concatenate(all_fidx, axis=0)

    n_frames = int(fidx_arr.max()) + 1
    elapsed  = time.time() - t0
    if verbose:
        print(f"\n  Done: {len(xyz_m):,} pts  {n_frames} frames  "
              f"{pkt_count} packets  {elapsed:.1f}s")

    return {
        'xyz_m':      xyz_m,
        'refl':       refl,
        'ts_ns':      ts_ns_arr,
        'frame_idx':  fidx_arr,
        'meta':       meta,
    }

def save_csv(data: dict, out_path: Path):
    xyz    = data['xyz_m']
    refl   = data['refl']
    ts_ns  = data['ts_ns']
    fidx   = data['frame_idx']
    print(f"Writing CSV → {out_path}  ({len(xyz):,} rows)...")
    t0 = time.time()
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'z', 'reflectivity', 'timestamp_ns', 'frame_idx'])
        # write in chunks for speed
        chunk = 50000
        for i in range(0, len(xyz), chunk):
            sl = slice(i, i+chunk)
            rows = zip(
                xyz[sl, 0].round(4), xyz[sl, 1].round(4), xyz[sl, 2].round(4),
                refl[sl], ts_ns[sl], fidx[sl]
            )
            w.writerows(rows)
    print(f"  CSV written in {time.time()-t0:.1f}s")

def save_npz(data: dict, out_path: Path):
    print(f"Writing NPZ → {out_path}...")
    np.savez_compressed(
        out_path,
        xyz=data['xyz_m'],
        refl=data['refl'],
        ts_ns=data['ts_ns'],
        frame_idx=data['frame_idx'],
    )
    print(f"  NPZ size: {out_path.with_suffix('.npz').stat().st_size/1e6:.1f} MB")

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('lvx2',    help='Input .lvx2 file')
    ap.add_argument('--out-dir', default=None, help='Output directory (default: same as input)')
    ap.add_argument('--format', choices=['csv','npz','both'], default='npz',
                    help='Output format (default: npz)')
    ap.add_argument('--frame-ms', type=int, default=100,
                    help='Frame period in ms (Mid-360: 100 = 10 Hz). Override if '
                         'the sensor ran at a different sweep rate.')
    args = ap.parse_args()

    lvx2_path = Path(args.lvx2)
    if not lvx2_path.exists():
        print(f"Error: {lvx2_path} not found"); sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else lvx2_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / lvx2_path.stem

    print(f"Parsing {lvx2_path.name}  ({lvx2_path.stat().st_size/1e6:.1f} MB)...")
    data = parse_lvx2(lvx2_path, frame_ms=args.frame_ms)

    meta = data['meta']
    print(f"  Device: {meta['devices'][0]['sn'] if meta['devices'] else '?'}")
    print(f"  Frame period (binning): {args.frame_ms} ms  "
          f"[LVX2 header 'frame_duration_ms' = {meta['frame_duration_ms']} — header unreliable]")

    if args.format in ('npz', 'both'):
        save_npz(data, Path(str(stem) + '_points.npz'))
    if args.format in ('csv', 'both'):
        save_csv(data, Path(str(stem) + '_points.csv'))

if __name__ == '__main__':
    main()
