#!/usr/bin/env python3
"""
compare_reid_embeddings.py — Compare ReID embeddings and visualize trajectories.

Compares:
  1. Existing PointNet (173K params, trained on indoor data)
  2. point-cloud-reid PointNet (109M params, WACV 2024, nuScenes)

Outputs:
  - Embedding similarity analysis
  - Trajectory plots (2D top-down view)
  - t-SNE embedding visualization
  - Per-session statistics

Filters:
  - Drop detections with confidence < 0.2
  - Drop detections with < 100 points in crop
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PED_LABEL = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Crop extraction (same as pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def count_points_in_crop(pts_xyz, box7, rng=None):
    """Count how many points fall inside a 3D bounding box."""
    if rng is None:
        rng = np.random.default_rng(0)
    cx, cy, cz, dx, dy, dz, yaw = box7.astype(float)
    valid = ~((pts_xyz[:, 0] == 0.0) & (pts_xyz[:, 1] == 0.0))
    pts = pts_xyz[valid]
    pts = pts - np.array([cx, cy, cz])
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = (R @ pts.T).T
    mask = ((np.abs(pts[:, 0]) <= dx / 2.0) &
            (np.abs(pts[:, 1]) <= dy / 2.0) &
            (np.abs(pts[:, 2]) <= dz / 2.0))
    return mask.sum()


def extract_crop(pts_xyz, box7, n_pts=256, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    cx, cy, cz, dx, dy, dz, yaw = box7.astype(float)
    valid = ~((pts_xyz[:, 0] == 0.0) & (pts_xyz[:, 1] == 0.0))
    pts = pts_xyz[valid]
    pts = pts - np.array([cx, cy, cz])
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = (R @ pts.T).T
    mask = ((np.abs(pts[:, 0]) <= dx / 2.0) &
            (np.abs(pts[:, 1]) <= dy / 2.0) &
            (np.abs(pts[:, 2]) <= dz / 2.0))
    pts_in = pts[mask]
    if len(pts_in) == 0:
        return np.zeros((n_pts, 3), dtype=np.float32)
    idx = rng.choice(len(pts_in), n_pts, replace=True)
    return pts_in[idx].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Load existing model embeddings
# ═══════════════════════════════════════════════════════════════════════════════

def load_existing_embeddings(session, reid_dir):
    """Load pre-computed embeddings from the existing PointNet model."""
    stem = os.path.join(reid_dir, f"emb_{session}")
    emb_path = f"{stem}_emb.npy"
    fi_path = f"{stem}_fi.npy"
    box_path = f"{stem}_box.npy"
    score_path = f"{stem}_score.npy"

    if not os.path.exists(emb_path):
        print(f"  [existing] No embeddings for {session}")
        return None

    emb = np.load(emb_path).astype(np.float32)
    fi = np.load(fi_path).astype(np.int32)
    box = np.load(box_path).astype(np.float32)
    score = np.load(score_path).astype(np.float32)

    # L2-normalise
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb = emb / norms

    return {"emb": emb, "fi": fi, "box": box, "score": score}


# ═══════════════════════════════════════════════════════════════════════════════
# Load new model embeddings (from npz + model inference)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_new_embeddings(session, npz, frames_dir, model, n_points=128,
                           min_score=0.2, min_points=100, max_range=15.0,
                           device='cuda', batch_size=64):
    """Compute embeddings with the new model, applying strict filters."""
    import torch

    model.eval()
    n_frames = len(npz["frame_files"])
    rng = np.random.default_rng(42)

    all_fi, all_box, all_emb, all_score, all_npts = [], [], [], [], []

    for fi in range(n_frames):
        boxes = np.asarray(npz["pred_boxes"][fi])
        labels = np.asarray(npz["pred_labels"][fi])
        scores = np.asarray(npz["pred_scores"][fi])

        mask = labels == PED_LABEL
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

        crops = []
        valid_indices = []
        for k, (box7, score) in enumerate(zip(bboxes, sc)):
            # Filter: confidence >= min_score
            if score < min_score:
                continue
            # Count points in crop
            n_pts_in = count_points_in_crop(pts, box7, rng=rng)
            if n_pts_in < min_points:
                continue
            crop = extract_crop(pts, box7, n_pts=n_points, rng=rng)
            crops.append(crop)
            valid_indices.append((k, n_pts_in))

        if not crops:
            continue

        crops_np = np.stack(crops)
        with torch.no_grad():
            for start in range(0, len(crops_np), batch_size):
                batch = torch.from_numpy(crops_np[start:start+batch_size]).to(device)
                emb = model(batch)
                emb_cpu = emb.cpu().numpy()
                for i, (idx, n_pts) in enumerate(valid_indices[start:start+batch_size]):
                    all_fi.append(fi)
                    all_box.append(bboxes[idx])
                    all_emb.append(emb_cpu[i])
                    all_score.append(sc[idx])
                    all_npts.append(n_pts)

    if not all_emb:
        return None

    return {
        "emb": np.array(all_emb, dtype=np.float32),
        "fi": np.array(all_fi, dtype=np.int32),
        "box": np.array(all_box, dtype=np.float32),
        "score": np.array(all_score, dtype=np.float32),
        "npts": np.array(all_npts, dtype=np.int32),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_similarity_matrix(emb):
    """Compute pairwise cosine similarity matrix."""
    return emb @ emb.T


def intra_inter_analysis(emb, fi, box, session_name):
    """Compute intra-frame and inter-frame similarity statistics."""
    # Group by frame
    frame_groups = {}
    for i, f in enumerate(fi):
        frame_groups.setdefault(f, []).append(i)

    intra_sims = []
    inter_sims = []

    frames = sorted(frame_groups.keys())
    for i, f1 in enumerate(frames):
        idx1 = frame_groups[f1]
        emb1 = emb[idx1]
        # Intra-frame (same frame, different detections)
        if len(idx1) > 1:
            sim = compute_similarity_matrix(emb1)
            # Off-diagonal elements
            mask = ~np.eye(len(idx1), dtype=bool)
            intra_sims.extend(sim[mask].tolist())

        # Inter-frame (consecutive frames)
        if i + 1 < len(frames):
            f2 = frames[i + 1]
            if abs(f2 - f1) <= 3:  # Within 3 frames
                idx2 = frame_groups[f2]
                emb2 = emb[idx2]
                sim = emb1 @ emb2.T
                inter_sims.extend(sim.flatten().tolist())

    return {
        "intra_mean": np.mean(intra_sims) if intra_sims else 0,
        "intra_std": np.std(intra_sims) if intra_sims else 0,
        "inter_mean": np.mean(inter_sims) if inter_sims else 0,
        "inter_std": np.std(inter_sims) if inter_sims else 0,
        "intra_count": len(intra_sims),
        "inter_count": len(inter_sims),
    }


def trajectory_analysis(box, fi, emb, session_name):
    """Extract trajectory data for plotting."""
    # Group by frame, take centroid
    frame_centroids = {}
    frame_emb = {}
    for i, f in enumerate(fi):
        cx, cy = box[i, 0], box[i, 1]
        frame_centroids.setdefault(f, []).append([cx, cy])
        frame_emb.setdefault(f, []).append(emb[i])

    frames = sorted(frame_centroids.keys())
    traj_x, traj_y, traj_emb = [], [], []
    for f in frames:
        centroids = np.array(frame_centroids[f])
        embs = np.array(frame_emb[f])
        # If multiple detections in frame, take the one closest to previous
        if len(centroids) > 1 and traj_x:
            dists = np.sqrt((centroids[:, 0] - traj_x[-1])**2 +
                           (centroids[:, 1] - traj_y[-1])**2)
            best = np.argmin(dists)
            traj_x.append(centroids[best, 0])
            traj_y.append(centroids[best, 1])
            traj_emb.append(embs[best])
        else:
            traj_x.append(centroids[0, 0])
            traj_y.append(centroids[0, 1])
            traj_emb.append(embs[0])

    return {
        "frames": np.array(frames),
        "x": np.array(traj_x),
        "y": np.array(traj_y),
        "emb": np.array(traj_emb),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trajectory_comparison(traj_old, traj_new, session, out_dir):
    """Plot trajectory comparison between old and new models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Old model trajectory
    ax = axes[0]
    if traj_old is not None and len(traj_old["x"]) > 0:
        scatter = ax.scatter(traj_old["x"], traj_old["y"],
                           c=traj_old["frames"], cmap='viridis',
                           s=20, alpha=0.7, edgecolors='none')
        ax.plot(traj_old["x"], traj_old["y"], 'k-', alpha=0.3, linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Frame Index')
    ax.set_title(f'Existing PointNet (173K params)\n{session}', fontsize=12)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # New model trajectory
    ax = axes[1]
    if traj_new is not None and len(traj_new["x"]) > 0:
        scatter = ax.scatter(traj_new["x"], traj_new["y"],
                           c=traj_new["frames"], cmap='viridis',
                           s=20, alpha=0.7, edgecolors='none')
        ax.plot(traj_new["x"], traj_new["y"], 'k-', alpha=0.3, linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Frame Index')
    ax.set_title(f'point-cloud-reid PointNet (109M params)\n{session}', fontsize=12)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"traj_comparison_{session}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_similarity_distributions(stats_old, stats_new, session, out_dir):
    """Plot similarity distribution comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Intra-frame similarity
    ax = axes[0]
    if stats_old["intra_count"] > 0:
        ax.bar(['Existing'], [stats_old["intra_mean"]],
               yerr=[stats_old["intra_std"]], capsize=5, alpha=0.7,
               label='Existing PointNet', color='steelblue')
    if stats_new["intra_count"] > 0:
        ax.bar(['point-cloud-reid'], [stats_new["intra_mean"]],
               yerr=[stats_new["intra_std"]], capsize=5, alpha=0.7,
               label='point-cloud-reid', color='coral')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'Intra-Frame Similarity\n{session}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Inter-frame similarity
    ax = axes[1]
    if stats_old["inter_count"] > 0:
        ax.bar(['Existing'], [stats_old["inter_mean"]],
               yerr=[stats_old["inter_std"]], capsize=5, alpha=0.7,
               label='Existing PointNet', color='steelblue')
    if stats_new["inter_count"] > 0:
        ax.bar(['point-cloud-reid'], [stats_new["inter_mean"]],
               yerr=[stats_new["inter_std"]], capsize=5, alpha=0.7,
               label='point-cloud-reid', color='coral')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'Inter-Frame Similarity (Δf ≤ 3)\n{session}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, f"similarity_{session}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_tsne_comparison(emb_old, emb_new, session, out_dir):
    """t-SNE visualization of both embedding spaces."""
    from sklearn.manifold import TSNE

    # Subsample for speed
    n_old = min(len(emb_old), 500)
    n_new = min(len(emb_new), 500)
    idx_old = np.random.choice(len(emb_old), n_old, replace=False)
    idx_new = np.random.choice(len(emb_new), n_new, replace=False)

    emb_combined = np.vstack([emb_old[idx_old], emb_new[idx_new]])
    labels = np.array(['Existing'] * n_old + ['point-cloud-reid'] * n_new)

    print(f"  Running t-SNE on {len(emb_combined)} embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(emb_combined)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'Existing': 'steelblue', 'point-cloud-reid': 'coral'}
    for label in ['Existing', 'point-cloud-reid']:
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=colors[label], label=label, s=15, alpha=0.6)
    ax.set_title(f't-SNE Embedding Comparison\n{session}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"tsne_{session}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_embedding_heatmap(emb_old, emb_new, session, out_dir):
    """Plot cosine similarity heatmaps for first 100 embeddings."""
    n = min(100, len(emb_old), len(emb_new))
    sim_old = compute_similarity_matrix(emb_old[:n])
    sim_new = compute_similarity_matrix(emb_new[:n])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(sim_old, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_title(f'Existing PointNet\nCosine Similarity (first {n})')
    ax.set_xlabel('Detection Index')
    ax.set_ylabel('Detection Index')
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(sim_new, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_title(f'point-cloud-reid\nCosine Similarity (first {n})')
    ax.set_xlabel('Detection Index')
    ax.set_ylabel('Detection Index')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    path = os.path.join(out_dir, f"heatmap_{session}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def cluster_unique_people(emb, fi, box, sim_threshold=0.7):
    """Identify unique people using embedding clustering (no temporal info).
    
    Uses greedy clustering: for each embedding, assign to existing cluster
    if cosine similarity > threshold, otherwise create new cluster.
    """
    n = len(emb)
    cluster_ids = np.full(n, -1, dtype=np.int32)
    cluster_centers = []
    cluster_count = 0

    for i in range(n):
        if cluster_count == 0:
            # First embedding starts first cluster
            cluster_ids[i] = 0
            cluster_centers.append(emb[i].copy())
            cluster_count = 1
            continue

        # Compute similarity to all cluster centers
        centers = np.stack(cluster_centers)
        sims = emb[i] @ centers.T
        best_cluster = np.argmax(sims)
        best_sim = sims[best_cluster]

        if best_sim > sim_threshold:
            cluster_ids[i] = best_cluster
            # Update cluster center (running average)
            count = (cluster_ids[:i] == best_cluster).sum()
            cluster_centers[best_cluster] = (
                cluster_centers[best_cluster] * count + emb[i]
            ) / (count + 1)
        else:
            cluster_ids[i] = cluster_count
            cluster_centers.append(emb[i].copy())
            cluster_count += 1

    return cluster_ids, cluster_count


def plot_unique_people_trajectories(cluster_ids, n_clusters, fi, box, emb,
                                     sim_threshold, session, out_dir):
    """Plot trajectories of identified unique people (colored by cluster)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: All trajectories colored by cluster
    ax = axes[0]
    cmap = plt.cm.tab20(np.linspace(0, 1, min(20, n_clusters)))
    for cid in range(n_clusters):
        mask = cluster_ids == cid
        color = cmap[cid % 20]
        ax.scatter(box[mask, 0], box[mask, 1], c=[color], s=25, alpha=0.7,
                  label=f'Person {cid+1} (n={mask.sum()})')
        # Connect points in frame order
        fi_cluster = fi[mask]
        order = np.argsort(fi_cluster)
        ax.plot(box[mask, 0][order], box[mask, 1][order], '-', color=color, alpha=0.4)

    ax.set_title(f'Unique People Trajectories (sim>{sim_threshold})\n{session}\n{n_clusters} people identified')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Right: Detection count per person
    ax = axes[1]
    counts = [np.sum(cluster_ids == i) for i in range(n_clusters)]
    bars = ax.bar(range(1, n_clusters+1), counts, color=cmap[:n_clusters])
    ax.set_xlabel('Person ID')
    ax.set_ylabel('Number of Detections')
    ax.set_title(f'Detection Count per Person\n{session}')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(out_dir, f"unique_people_{session}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return counts


def plot_detection_stats(session_data, out_dir):
    """Plot detection statistics across sessions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Detections per frame
    ax = axes[0, 0]
    for label, data in session_data.items():
        fi_counts = np.bincount(data["fi"], minlength=data["fi"].max()+1)
        ax.plot(fi_counts, alpha=0.7, label=label)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Detections per Frame')
    ax.set_title('Detection Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Score distribution
    ax = axes[0, 1]
    for label, data in session_data.items():
        ax.hist(data["score"], bins=30, alpha=0.5, label=label)
    ax.set_xlabel('Detection Score')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Point count distribution
    ax = axes[1, 0]
    for label, data in session_data.items():
        if "npts" in data:
            ax.hist(data["npts"], bins=30, alpha=0.5, label=label)
    ax.set_xlabel('Points in Crop')
    ax.set_ylabel('Count')
    ax.set_title('Point Count Distribution (filtered ≥100)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Embedding norm distribution
    ax = axes[1, 1]
    for label, data in session_data.items():
        norms = np.linalg.norm(data["emb"], axis=1)
        ax.hist(norms, bins=30, alpha=0.5, label=label)
    ax.set_xlabel('L2 Norm')
    ax.set_ylabel('Count')
    ax.set_title('Embedding Norm Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Detection & Embedding Statistics', fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "detection_stats.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="2026-07-29_17-21-48")
    ap.add_argument("--lidar-dir", default=os.path.expanduser("~/Projects/Thesis/Lidar Data"))
    ap.add_argument("--model", default="point-cloud-reid/pretrained/nuscenes/pts_pointnet_r_nus_det_500e.pth")
    ap.add_argument("--reid-dir", default=os.path.join(HERE, "reid_data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "reid_analysis"))
    ap.add_argument("--min-score", type=float, default=0.2)
    ap.add_argument("--min-points", type=int, default=100)
    ap.add_argument("--max-range", type=float, default=15.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load new model ──
    print(f"[model] Loading {args.model}")
    import torch
    sys.path.insert(0, HERE)
    from reid_embed_pointcloudreid import ReIDNet
    model = ReIDNet(backbone_type='pointnet', emb_dim=128, n_points=128)
    state = torch.load(args.model, map_location='cpu')
    if 'state_dict' in state:
        sd = state['state_dict']
    elif 'model' in state:
        sd = state['model']
    else:
        sd = state
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Loaded: {len(sd)} keys, {len(missing)} missing, {len(unexpected)} unexpected")
    model.to(args.device)
    model.eval()

    # ── Load data ──
    labels_path = os.path.join(args.lidar_dir, f"{args.session}_frames_voxelnext.npz")
    print(f"[data] Loading {labels_path}")
    npz = np.load(labels_path, allow_pickle=True)
    frames_dir = os.path.join(args.lidar_dir, "frames", args.session)

    # ── Load existing embeddings ──
    print(f"[existing] Loading embeddings from {args.reid_dir}")
    old_data = load_existing_embeddings(args.session, args.reid_dir)

    # ── Compute new embeddings ──
    print(f"[new] Computing embeddings (min_score={args.min_score}, min_points={args.min_points})")
    new_data = compute_new_embeddings(
        args.session, npz, frames_dir, model,
        n_points=128, min_score=args.min_score, min_points=args.min_points,
        max_range=args.max_range, device=args.device, batch_size=64
    )

    if new_data is None:
        print("[new] No embeddings computed!")
        return

    print(f"  New embeddings: {len(new_data['emb'])} detections")
    if old_data is not None:
        print(f"  Old embeddings: {len(old_data['emb'])} detections")

    # ── Analysis ──
    print("\n" + "="*60)
    print("EMBEDDING COMPARISON")
    print("="*60)

    # Intra/inter analysis
    if old_data is not None:
        stats_old = intra_inter_analysis(old_data["emb"], old_data["fi"],
                                         old_data["box"], "old")
        print(f"\nExisting PointNet:")
        print(f"  Intra-frame similarity: {stats_old['intra_mean']:.4f} ± {stats_old['intra_std']:.4f} ({stats_old['intra_count']} pairs)")
        print(f"  Inter-frame similarity: {stats_old['inter_mean']:.4f} ± {stats_old['inter_std']:.4f} ({stats_old['inter_count']} pairs)")
    else:
        stats_old = {"intra_mean": 0, "intra_std": 0, "inter_mean": 0, "inter_std": 0,
                     "intra_count": 0, "inter_count": 0}

    stats_new = intra_inter_analysis(new_data["emb"], new_data["fi"],
                                     new_data["box"], "new")
    print(f"\npoint-cloud-reid PointNet:")
    print(f"  Intra-frame similarity: {stats_new['intra_mean']:.4f} ± {stats_new['intra_std']:.4f} ({stats_new['intra_count']} pairs)")
    print(f"  Inter-frame similarity: {stats_new['inter_mean']:.4f} ± {stats_new['inter_std']:.4f} ({stats_new['inter_count']} pairs)")

    # Embedding space statistics
    print(f"\nEmbedding Space:")
    if old_data is not None:
        norms_old = np.linalg.norm(old_data["emb"], axis=1)
        print(f"  Existing:  mean_norm={norms_old.mean():.4f}, std={norms_old.std():.4f}")
    norms_new = np.linalg.norm(new_data["emb"], axis=1)
    print(f"  New:       mean_norm={norms_new.mean():.4f}, std={norms_new.std():.4f}")

    # ── Trajectory analysis ──
    print("\n" + "="*60)
    print("TRAJECTORY ANALYSIS")
    print("="*60)

    if old_data is not None:
        traj_old = trajectory_analysis(old_data["box"], old_data["fi"],
                                       old_data["emb"], "old")
        print(f"  Existing: {len(traj_old['x'])} trajectory points")
    else:
        traj_old = None

    traj_new = trajectory_analysis(new_data["box"], new_data["fi"],
                                   new_data["emb"], "new")
    print(f"  New:      {len(traj_new['x'])} trajectory points")

    # ── Plotting ──
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    # Trajectory comparison
    plot_trajectory_comparison(traj_old, traj_new, args.session, args.out_dir)

    # Similarity distributions
    plot_similarity_distributions(stats_old, stats_new, args.session, args.out_dir)

    # t-SNE
    if old_data is not None:
        plot_tsne_comparison(old_data["emb"], new_data["emb"], args.session, args.out_dir)

    # Heatmap
    if old_data is not None:
        plot_embedding_heatmap(old_data["emb"], new_data["emb"], args.session, args.out_dir)

    # Detection stats
    session_data = {}
    if old_data is not None:
        session_data["Existing PointNet"] = old_data
    session_data["point-cloud-reid"] = new_data
    plot_detection_stats(session_data, args.out_dir)

    # ── Unique people clustering (no temporal info) ──
    print("\n" + "="*60)
    print("UNIQUE PEOPLE IDENTIFICATION (Embedding Only, No Temporal)")
    print("="*60)

    if old_data is not None:
        print("\nExisting PointNet:")
        cluster_ids_old, n_old = cluster_unique_people(
            old_data["emb"], old_data["fi"], old_data["box"], sim_threshold=0.7)
        counts_old = plot_unique_people_trajectories(
            cluster_ids_old, n_old, old_data["fi"], old_data["box"], old_data["emb"],
            0.7, f"{args.session}_existing", args.out_dir)
        print(f"  Identified: {n_old} unique people")
        print(f"  Detections per person: min={min(counts_old)}, max={max(counts_old)}, mean={np.mean(counts_old):.1f}")

    print("\npoint-cloud-reid PointNet:")
    cluster_ids_new, n_new = cluster_unique_people(
        new_data["emb"], new_data["fi"], new_data["box"], sim_threshold=0.7)
    counts_new = plot_unique_people_trajectories(
        cluster_ids_new, n_new, new_data["fi"], new_data["box"], new_data["emb"],
        0.7, f"{args.session}_new", args.out_dir)
    print(f"  Identified: {n_new} unique people")
    print(f"  Detections per person: min={min(counts_new)}, max={max(counts_new)}, mean={np.mean(counts_new):.1f}")

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Session: {args.session}")
    print(f"Filter: score >= {args.min_score}, points >= {args.min_points}, range <= {args.max_range}m")
    print(f"")
    print(f"{'Metric':<35} {'Existing':>12} {'New':>12}")
    print("-"*60)
    print(f"{'Total detections':<35} {len(old_data['emb']) if old_data else 0:>12} {len(new_data['emb']):>12}")
    print(f"{'Unique frames':<35} {len(np.unique(old_data['fi'])) if old_data else 0:>12} {len(np.unique(new_data['fi'])):>12}")
    print(f"{'Unique people (sim>0.7)':<35} {n_old if old_data else 0:>12} {n_new:>12}")
    print(f"{'Intra-frame similarity':<35} {stats_old['intra_mean']:>12.4f} {stats_new['intra_mean']:>12.4f}")
    print(f"{'Inter-frame similarity':<35} {stats_old['inter_mean']:>12.4f} {stats_new['inter_mean']:>12.4f}")
    print(f"{'Embedding norm (mean)':<35} {norms_old.mean() if old_data else 0:>12.4f} {norms_new.mean():>12.4f}")
    print(f"{'Trajectory points':<35} {len(traj_old['x']) if traj_old else 0:>12} {len(traj_new['x']):>12}")
    print(f"")
    print(f"Plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
