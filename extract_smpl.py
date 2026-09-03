#!/usr/bin/env python3
"""
extract_smpl.py — extract SMPL body-shape parameters (beta: 10-d) from LiDAR
person detections using the LiDAR-HMR model.

Run:
    python3 extract_smpl.py --session 2026-07-29_17-21-48
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HMR_DIR = os.path.join(HERE, "LiDAR-HMR")
PED_LABEL = 2


# ── crop extraction (same as reid_embed_server.py) ─────────────────────────

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


class SMPLExtractor:
    """Wraps LiDAR-HMR to regress SMPL body shape from LiDAR point crops."""

    def __init__(self, weights_path=None, device="cuda", config_path="configs/mesh/sloper4d.yaml"):
        if HMR_DIR not in sys.path:
            sys.path.insert(0, HMR_DIR)
        import torch
        self.torch = torch

        cwd = os.getcwd()
        os.chdir(HMR_DIR)  # model ctor loads files via relative paths (./models/..., ./smplx_models/...)
        try:
            from models.pmg_config import config, update_config
            update_config(config_path)
            from models.pose_mesh_net import LiDAR_HMR
            self.device = torch.device(device)
            self.model = LiDAR_HMR(pmg_cfg=config, train_pmg=True, device=str(self.device))
        finally:
            os.chdir(cwd)

        if weights_path:
            print(f"[smpl] Loading weights from {weights_path}")
            state = torch.load(weights_path, map_location="cpu")
            if "net" in state:
                state = state["net"]
            elif "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

    def extract_from_crop(self, crop_points: np.ndarray) -> dict:
        """crop_points: (n_pts, 3) numpy → dict with beta, theta, mesh (all numpy, batch dim stripped)."""
        return self.extract_from_crops(crop_points[None, ...])

    def extract_from_crops(self, crops: np.ndarray) -> dict:
        """crops: (B, n_pts, 3) numpy → dict with beta (B,10), theta (B,72), mesh (B,6890,3)."""
        torch = self.torch
        with torch.no_grad():
            pcd = torch.from_numpy(crops.astype(np.float32)).to(self.device)
            out = self.model(pcd)
        return {
            "beta":  out["pose_beta"].cpu().numpy(),
            "theta": out["pose_theta"].cpu().numpy(),
            "mesh":  out["mesh_refine"].cpu().numpy(),
            "trans": out["trans"].cpu().numpy(),
        }

    def extract_from_session(self, npz_path, frames_dir, min_score=0.2,
                             max_range=5.0, n_pts=256, batch_size=32) -> dict:
        """Runs SMPL extraction over every pedestrian detection in a session NPZ.

        Returns dict with arrays: beta (N,10), theta (N,72), fi (N,), box (N,7), score (N,)
        """
        from tqdm import tqdm

        npz = np.load(npz_path, allow_pickle=True)
        n_frames = len(npz["frame_files"])
        rng = np.random.default_rng(0)

        crops, fi_l, box_l, score_l = [], [], [], []
        for fi in tqdm(range(n_frames), desc="crops"):
            boxes  = np.asarray(npz["pred_boxes"][fi])
            labels = np.asarray(npz["pred_labels"][fi])
            scores = np.asarray(npz["pred_scores"][fi])
            mask = (labels == PED_LABEL) & (scores >= min_score)
            if not mask.any():
                continue
            b = boxes[mask]
            s = scores[mask]
            if max_range > 0:
                r = np.hypot(b[:, 0], b[:, 1])
                keep = r <= max_range
                b, s = b[keep], s[keep]
                if len(b) == 0:
                    continue
            frame_name = str(npz["frame_files"][fi])
            frame_path = os.path.join(frames_dir, frame_name)
            if not os.path.exists(frame_path):
                continue
            pts = np.load(frame_path)[:, :3].astype(float)
            for box7, score in zip(b, s):
                crop = _extract_crop(pts, box7, n_pts=n_pts, rng=rng)
                crops.append(crop)
                fi_l.append(fi)
                box_l.append(box7)
                score_l.append(score)

        if not crops:
            return {"beta": np.zeros((0, 10), dtype=np.float32),
                    "theta": np.zeros((0, 72), dtype=np.float32),
                    "fi": np.zeros((0,), dtype=np.int32),
                    "box": np.zeros((0, 7), dtype=np.float32),
                    "score": np.zeros((0,), dtype=np.float32)}

        crops_np = np.stack(crops)  # (N, n_pts, 3)
        beta_l, theta_l = [], []
        for start in tqdm(range(0, len(crops_np), batch_size), desc="SMPL"):
            out = self.extract_from_crops(crops_np[start:start + batch_size])
            beta_l.append(out["beta"])
            theta_l.append(out["theta"])

        return {
            "beta":  np.concatenate(beta_l, axis=0).astype(np.float32),
            "theta": np.concatenate(theta_l, axis=0).astype(np.float32),
            "fi":    np.array(fi_l, dtype=np.int32),
            "box":   np.array(box_l, dtype=np.float32),
            "score": np.array(score_l, dtype=np.float32),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session",    default="2026-07-29_17-21-48")
    ap.add_argument("--lidar-dir",  default="~/Projects/Thesis/Lidar Data")
    ap.add_argument("--weights",    default=None, help="Path to checkpoint .pth")
    ap.add_argument("--config",     default="configs/mesh/sloper4d.yaml", help="Path to config file")
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--min-score",  type=float, default=0.2)
    ap.add_argument("--max-range",  type=float, default=5.0,
                    help="Discard detections farther than this many metres from sensor (0=off)")
    ap.add_argument("--n-pts",      type=int, default=256, help="Points per crop")
    ap.add_argument("--output-dir", default="reid_data")
    args = ap.parse_args()

    LIDAR    = os.path.expanduser(args.lidar_dir)
    npz_p    = os.path.join(LIDAR, f"{args.session}_frames_voxelnext.npz")
    frames_d = os.path.join(LIDAR, "frames", args.session)
    if not os.path.exists(npz_p):
        sys.exit(f"detections not found: {npz_p}")
    if not os.path.isdir(frames_d):
        sys.exit(f"frames dir not found: {frames_d}")

    out_dir = os.path.join(HERE, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[smpl] Loading LiDAR-HMR model → device={args.device}")
    extractor = SMPLExtractor(weights_path=args.weights, device=args.device, config_path=args.config)

    print(f"[smpl] Extracting SMPL params for session {args.session}")
    result = extractor.extract_from_session(
        npz_p, frames_d,
        min_score=args.min_score,
        max_range=args.max_range,
        n_pts=args.n_pts)

    n = len(result["fi"])
    prefix = os.path.join(out_dir, f"smpl_{args.session}")
    np.save(prefix + "_beta.npy",  result["beta"])
    np.save(prefix + "_theta.npy", result["theta"])
    np.save(prefix + "_fi.npy",    result["fi"])
    np.save(prefix + "_box.npy",   result["box"])
    np.save(prefix + "_score.npy", result["score"])

    print(f"[smpl] {n} detections processed")
    print(f"[smpl] Saved → {prefix}_{{beta,theta,fi,box,score}}.npy")


if __name__ == "__main__":
    main()
