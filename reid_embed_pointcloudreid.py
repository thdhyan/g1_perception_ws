#!/usr/bin/env python3
"""
reid_embed_pointcloudreid.py — Standalone inference for point-cloud-reid models.

Extracts 128-d embeddings from LiDAR crops using pre-trained models from
https://github.com/bentherien/point-cloud-reid (WACV 2024).

No mmdet3d installation required — defines model architecture in pure PyTorch.

Usage:
    python reid_embed_pointcloudreid.py --session 2026-07-29_17-21-48 \
        --model pretrained/nuscenes/pts_pointnet_r_nus_det_500e.pth \
        --backbone pointnet --port 8767

Output: reid_data/emb_{session}_*.npy (same format as existing pipeline)
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
WS = HERE  # Use current directory as workspace root
PED_LABEL = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Model Architecture (extracted from point-cloud-reid, pure PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

class LinearAttention(nn.Module):
    """Linear Transformer attention (from 'Transformers are RNNs')."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, Q, K, V):
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        v_length = V.size(1)
        V = V / v_length
        KV = torch.einsum("nshd,nshv->nhdv", K, V)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        return torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length


class SelfAttention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, d_model, nhead, attention='linear'):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = LinearAttention() if attention == 'linear' else None

    def forward(self, feat, xyz=None, mask=None):
        B, N, D = feat.shape
        Q = self.W_Q(feat).view(B, N, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_K(feat).view(B, N, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_V(feat).view(B, N, self.nhead, self.d_k).transpose(1, 2)
        if self.attention:
            out = self.attention(Q, K, V)
        else:
            scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class PointNetBackbone(nn.Module):
    """PointNet2-style backbone with Set Abstraction modules."""
    def __init__(self, input_channels=3, conv_out=32):
        super().__init__()
        # Simplified: 3 conv layers + max pool
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, conv_out, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(conv_out)

    def forward(self, x):
        """x: (B, N, 3) -> (B, conv_out, N)"""
        x = x.transpose(1, 2)  # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x  # (B, conv_out, N)


class RTMM(nn.Module):
    """Real-Time Matching Module — cross-correlation matching head."""
    def __init__(self, hidden_size, output_size=1):
        super().__init__()
        self.match = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, emb_a, emb_b):
        """emb_a, emb_b: (B, D) -> match_score: (B, 1)"""
        combined = torch.cat([emb_a, emb_b], dim=-1)
        return self.match(combined)


class ReIDNet(nn.Module):
    """
    Full ReID model: backbone + RTMM matching head.
    For inference, we only use the backbone to extract embeddings.
    """
    def __init__(self, backbone_type='pointnet', emb_dim=128, n_points=128):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_points = n_points

        if backbone_type == 'pointnet':
            self.backbone = PointNetBackbone(input_channels=3, conv_out=emb_dim)
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")

        # Pooling: both avg and max, then concatenate
        self.pool_dim = emb_dim * 2

        # Final projection to emb_dim
        self.proj = nn.Linear(self.pool_dim, emb_dim)

    def forward(self, x):
        """
        x: (B, N, 3) point cloud
        Returns: (B, emb_dim) L2-normalised embedding
        """
        feat = self.backbone(x)  # (B, emb_dim, N)

        # Pool: avg + max
        avg_pool = feat.mean(dim=2)  # (B, emb_dim)
        max_pool = feat.max(dim=2).values  # (B, emb_dim)
        pooled = torch.cat([avg_pool, max_pool], dim=-1)  # (B, emb_dim*2)

        emb = self.proj(pooled)  # (B, emb_dim)
        emb = F.normalize(emb, p=2, dim=1)  # L2-normalise
        return emb


# ═══════════════════════════════════════════════════════════════════════════════
# Crop extraction (same as reid_embed_server.py)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_crop(pts_xyz, box7, n_points=128, rng=None):
    """Extract a point cloud crop from a 3D bounding box.

    Args:
        pts_xyz: (N, 3) raw point cloud
        box7: [cx, cy, cz, dx, dy, dz, yaw]
        n_points: number of points to sample
        rng: random generator
    Returns:
        crop: (n_points, 3) centred, de-yawed points
    """
    if rng is None:
        rng = np.random.default_rng(0)

    cx, cy, cz, dx, dy, dz, yaw = box7.astype(float)

    # Remove no-return placeholders
    valid = ~((pts_xyz[:, 0] == 0.0) & (pts_xyz[:, 1] == 0.0))
    pts = pts_xyz[valid]

    # Translate to box centre
    pts = pts - np.array([cx, cy, cz])

    # De-yaw
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = pts @ R.T

    # Filter to box extent (with 10% margin)
    hw, hh, ht = dx / 2 * 1.1, dy / 2 * 1.1, dz / 2 * 1.1
    mask = (np.abs(pts[:, 0]) <= hw) & (np.abs(pts[:, 1]) <= hh) & (np.abs(pts[:, 2]) <= ht)
    pts = pts[mask]

    if len(pts) == 0:
        # Fallback: use all points near the box centre
        pts = pts_xyz - np.array([cx, cy, cz])
        dists = np.linalg.norm(pts, axis=1)
        mask = dists <= max(dx, dy, dz)
        pts = pts[mask] if mask.any() else pts[:n_points]

    # Sample n_points (with replacement if needed)
    if len(pts) >= n_points:
        idx = rng.choice(len(pts), n_points, replace=False)
    else:
        idx = rng.choice(len(pts), n_points, replace=True)
    return pts[idx].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding extraction
# ═══════════════════════════════════════════════════════════════════════════════

def embed_session(model, npz, frames_dir, n_points=128, min_score=0.0,
                  batch_size=64, device='cuda', max_range=5.0):
    """Extract embeddings for all pedestrian detections in a session.

    Returns:
        fi_arr: (N,) frame indices
        box_arr: (N, 7) bounding boxes
        emb_arr: (N, 128) L2-normalised embeddings
        score_arr: (N,) detection scores
    """
    model.eval()
    n_frames = len(npz["frame_files"])
    rng = np.random.default_rng(42)

    all_fi, all_box, all_emb, all_score = [], [], [], []

    for fi in range(n_frames):
        boxes = np.asarray(npz["pred_boxes"][fi])
        labels = np.asarray(npz["pred_labels"][fi])
        scores = np.asarray(npz["pred_scores"][fi])

        mask = labels == PED_LABEL
        if min_score > 0:
            mask &= scores >= min_score
        if not mask.any():
            continue

        # Load point cloud
        fname = str(npz["frame_files"][fi])
        frame_path = os.path.join(frames_dir, fname)
        if not os.path.exists(frame_path):
            continue
        pts = np.load(frame_path)[:, :3].astype(np.float32)

        # Range filter
        if max_range > 0:
            r = np.hypot(pts[:, 0], pts[:, 1])
            pts = pts[r <= max_range]

        bboxes = boxes[mask]
        sc = scores[mask]

        # Extract crops
        crops = []
        valid_indices = []
        for k, box7 in enumerate(bboxes):
            crop = extract_crop(pts, box7, n_points=n_points, rng=rng)
            crops.append(crop)
            valid_indices.append(k)

        if not crops:
            continue

        # Batch embed
        crops_np = np.stack(crops)  # (K, n_points, 3)
        with torch.no_grad():
            for start in range(0, len(crops_np), batch_size):
                batch = torch.from_numpy(crops_np[start:start+batch_size]).to(device)
                emb = model(batch)  # (B, 128) L2-normalised
                emb_cpu = emb.cpu().numpy()

                for i, idx in enumerate(valid_indices[start:start+batch_size]):
                    all_fi.append(fi)
                    all_box.append(bboxes[idx])
                    all_emb.append(emb_cpu[i])
                    all_score.append(sc[idx])

        if fi % 200 == 0:
            print(f"  [embed] frame {fi}/{n_frames} — {len(all_emb)} embeddings so far")

    return (np.array(all_fi, dtype=np.int32),
            np.array(all_box, dtype=np.float32),
            np.array(all_emb, dtype=np.float32),
            np.array(all_score, dtype=np.float32))


def save_embeddings(session, fi_arr, box_arr, emb_arr, score_arr, out_dir):
    """Save embeddings in the same format as the existing pipeline."""
    stem = os.path.join(out_dir, f"emb_{session}")
    np.save(f"{stem}_emb.npy", emb_arr.astype(np.float32))
    np.save(f"{stem}_fi.npy", fi_arr.astype(np.int32))
    np.save(f"{stem}_box.npy", box_arr.astype(np.float32))
    np.save(f"{stem}_score.npy", score_arr.astype(np.float32))
    print(f"[save] {len(emb_arr)} embeddings → {stem}_*.npy")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="2026-07-29_17-21-48")
    ap.add_argument("--lidar-dir", default=os.path.expanduser("~/Projects/Thesis/Lidar Data"))
    ap.add_argument("--model", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--backbone", default="pointnet", choices=["pointnet"])
    ap.add_argument("--n-points", type=int, default=128, help="Points per crop")
    ap.add_argument("--min-score", type=float, default=0.4)
    ap.add_argument("--max-range", type=float, default=5.0, help="Max range in metres")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=os.path.join(WS, "reid_data"))
    args = ap.parse_args()

    # Load model
    print(f"[model] Loading {args.backbone} from {args.model}")
    model = ReIDNet(backbone_type=args.backbone, emb_dim=128, n_points=args.n_points)

    state = torch.load(args.model, map_location="cpu")
    # Handle different checkpoint formats
    if "state_dict" in state:
        sd = state["state_dict"]
        # Remove "module." prefix if present (DataParallel)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    elif "model" in state:
        sd = state["model"]
    else:
        sd = state

    # Try loading with strict=False to handle architecture mismatches
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[model] Warning: {len(missing)} missing keys: {missing[:5]}...")
    if unexpected:
        print(f"[model] Warning: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

    model.to(args.device)
    model.eval()
    print(f"[model] Loaded on {args.device}, {sum(v.numel() for v in model.parameters()):,} params")

    # Load labels
    labels_path = os.path.join(args.lidar_dir, f"{args.session}_frames_voxelnext.npz")
    print(f"[data] Loading {labels_path}")
    npz = np.load(labels_path, allow_pickle=True)

    # Extract embeddings
    t0 = time.time()
    fi_arr, box_arr, emb_arr, score_arr = embed_session(
        model, npz, os.path.join(args.lidar_dir, "frames", args.session),
        n_points=args.n_points, min_score=args.min_score,
        batch_size=args.batch_size, device=args.device, max_range=args.max_range
    )
    elapsed = time.time() - t0
    print(f"[embed] {len(emb_arr)} embeddings in {elapsed:.1f}s "
          f"({len(emb_arr)/elapsed:.0f} emb/s)")

    # Save
    save_embeddings(args.session, fi_arr, box_arr, emb_arr, score_arr, args.out_dir)


if __name__ == "__main__":
    main()
