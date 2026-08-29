"""
train_reid.py — Train PointNet ReID on mined LiDAR person crops.

Splits by session: July (session 0) → train, August (sessions 1,2) → val.
Loss: batch-hard triplet (0.7 weight) + cross-entropy (0.3 weight).

Usage:
    python3 train_reid.py [--epochs 60] [--batch 64] [--emb-dim 128] [--lr 1e-3]
                         [--data-dir reid_data] [--out reid_data/model.pt]
                         [--device cpu|cuda]

Outputs:
    reid_data/model.pt       — best checkpoint (by val triplet loss)
    reid_data/train_log.csv  — epoch, train_loss, val_loss, val_rank1
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from reid_model import ReIDModel, batch_hard_triplet_loss


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# P×K sampler — P classes per batch, K samples each (standard for triplet)
# ────────────────────────────────────────────────────────────────────────────

class PKSampler(Sampler):
    """Yields batches of P*K indices: P classes × K random samples each."""

    def __init__(self, labels: np.ndarray, P: int, K: int, rng_seed: int = 0):
        self.P   = P
        self.K   = K
        self.rng = np.random.default_rng(rng_seed)
        # Build class → indices map.
        self.cls_to_idx = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.cls_to_idx[int(lbl)].append(i)
        # Only classes with ≥ 2 samples are useful for positives.
        self.valid_cls = [c for c, idxs in self.cls_to_idx.items() if len(idxs) >= 2]
        if len(self.valid_cls) < P:
            self.P = max(1, len(self.valid_cls))

    def __iter__(self):
        cls_list = self.rng.permutation(self.valid_cls).tolist()
        batch = []
        for cl in cls_list:
            idxs = self.cls_to_idx[cl]
            chosen = self.rng.choice(idxs, self.K, replace=len(idxs) < self.K).tolist()
            batch.extend(chosen)
            if len(batch) >= self.P * self.K:
                yield batch
                batch = []

    def __len__(self):
        n_batches = max(1, len(self.valid_cls) // self.P)
        return n_batches


def augment_crop(pts: torch.Tensor, train: bool) -> torch.Tensor:
    """
    pts: (N, 3) float32 — in canonical (centred, de-yawed) frame.
    Applies random Z-rotation, jitter, and point dropout for training.
    """
    if not train:
        return pts
    # Random Z-rotation (the crop is already de-yawed, so this adds variation).
    angle = (torch.rand(1).item() - 0.5) * 2 * math.pi   # ±180°
    c, s  = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=pts.dtype)
    pts = (R @ pts.t()).t()
    # Gaussian jitter.
    pts = pts + torch.randn_like(pts) * 0.01
    # Random point dropout: replace 10% of points with random others.
    N = pts.shape[0]
    n_drop = int(N * 0.1)
    drop_idx = torch.randperm(N)[:n_drop]
    src_idx  = torch.randint(0, N, (n_drop,))
    pts[drop_idx] = pts[src_idx]
    return pts


class CropDataset(Dataset):
    def __init__(self, crops, pseudo_ids, class_map, train: bool = False):
        """
        crops:      (N, 256, 3) float32
        pseudo_ids: (N,) int — raw pseudo_id; -1 means singleton (excluded upstream)
        class_map:  dict raw_pid → 0-based class index
        train:      enable augmentation
        """
        self.crops = torch.from_numpy(crops)
        self.labels = torch.tensor(
            [class_map[p] for p in pseudo_ids], dtype=torch.long
        )
        self.train = train

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        pts = self.crops[idx].clone()
        pts = augment_crop(pts, self.train)
        return pts, self.labels[idx]


# ────────────────────────────────────────────────────────────────────────────
# Rank-1 evaluation (cosine similarity, gallery=first crop per class)
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_rank1(model, loader, device):
    """
    Standard single-shot ReID evaluation.
    Gallery: first occurrence of each class (one per class).
    Queries: ALL samples.
    For a query that IS the gallery entry: skip (no match possible).
    For others: find nearest gallery; correct if same class.
    """
    model.eval()
    all_emb, all_lbl = [], []
    for crops, labels in loader:
        crops = crops.to(device)
        emb, _ = model(crops)
        all_emb.append(emb.cpu())
        all_lbl.append(labels)
    emb = torch.cat(all_emb)            # (N, D)
    lbl = torch.cat(all_lbl)            # (N,)

    # Gallery: first occurrence per class (record the sample index).
    seen = {}
    g_emb, g_lbl, g_sample_idx = [], [], []
    for i in range(len(lbl)):
        cl = lbl[i].item()
        if cl not in seen:
            seen[cl] = True
            g_emb.append(emb[i])
            g_lbl.append(cl)
            g_sample_idx.append(i)          # which sample is the gallery entry
    g_emb = torch.stack(g_emb)             # (G, D)
    gallery_set = set(g_sample_idx)         # indices that ARE gallery entries

    correct = 0
    total   = 0
    for i in range(len(lbl)):
        if i in gallery_set:
            continue                         # skip: this sample IS the gallery entry
        cl  = lbl[i].item()
        sim = (emb[i] @ g_emb.t())         # (G,) — no masking; gallery ≠ query
        pred = g_lbl[sim.argmax().item()]
        if pred == cl:
            correct += 1
        total += 1

    return correct / max(total, 1)


# ────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────

def train(args):
    data_dir = args.data_dir
    crops      = np.load(os.path.join(data_dir, "crops.npy"))           # (N, 256, 3)
    pseudo_ids = np.load(os.path.join(data_dir, "pseudo_ids.npy"))      # (N,)
    sess_ids   = np.load(os.path.join(data_dir, "session_ids.npy"))     # (N,)
    frame_ids  = np.load(os.path.join(data_dir, "frame_ids.npy"))       # (N,)

    # Keep only crops with a valid pseudo_id (≥ 0).
    valid = pseudo_ids >= 0
    crops      = crops[valid]
    pseudo_ids = pseudo_ids[valid]
    sess_ids   = sess_ids[valid]
    frame_ids  = frame_ids[valid]
    print(f"Valid crops: {len(crops)}  (dropped {(~valid).sum()} singletons)")

    # Make globally unique class ids: sess * 100_000 + pid.
    # This prevents July pid=5 and August pid=5 (different people) from colliding.
    global_ids = sess_ids * 100_000 + pseudo_ids

    # Within-session split on July (session 0) only.
    # Random 80/20 split WITHIN each track segment (global_id) so every identity
    # contributes crops to both train and val → rank-1 is meaningful.
    july_mask  = sess_ids == 0
    july_gids  = global_ids[july_mask]
    july_crops = crops[july_mask]
    july_n     = july_mask.sum()

    rng_split  = np.random.default_rng(42)
    train_flag = np.zeros(july_n, dtype=bool)
    for gid in np.unique(july_gids):
        idx = np.where(july_gids == gid)[0]
        n_train = max(1, int(len(idx) * 0.8))
        chosen  = rng_split.choice(idx, n_train, replace=False)
        train_flag[chosen] = True
    val_flag = ~train_flag

    n_train_crops = train_flag.sum()
    n_val_crops   = val_flag.sum()
    n_train_cls   = len(np.unique(july_gids[train_flag]))
    n_val_cls     = len(np.unique(july_gids[val_flag]))
    print(f"Within-July 80/20 split: train={n_train_crops} ({n_train_cls} IDs), "
          f"val={n_val_crops} ({n_val_cls} IDs)")

    # Build class map from train class set.
    train_gids_list = sorted(set(july_gids[train_flag].tolist()))
    class_map = {gid: i for i, gid in enumerate(train_gids_list)}
    n_classes = len(class_map)
    print(f"n_classes: {n_classes}")

    # Filter classes with ≥ 2 train crops (need positives for triplet).
    cls_count = defaultdict(int)
    for gid in july_gids[train_flag]:
        cls_count[gid] += 1
    rich_cls = {gid for gid, cnt in cls_count.items() if cnt >= 2}
    print(f"Classes with ≥2 train crops: {len(rich_cls)} / {n_classes}")

    # Rebuild class_map over rich_cls only.
    class_map = {gid: i for i, gid in enumerate(sorted(rich_cls))}
    n_classes  = len(class_map)

    def make_loader_train(c, g):
        keep = np.array([gid in class_map for gid in g])
        sub_g = g[keep]
        mapped = np.array([class_map[gid] for gid in sub_g])
        ds = CropDataset(c[keep], sub_g, class_map, train=True)
        sampler = PKSampler(mapped, P=args.P, K=args.K)
        return DataLoader(ds, batch_sampler=sampler, num_workers=0)

    def make_loader_val(c, g):
        keep = np.array([gid in class_map for gid in g])
        sub_g = g[keep]
        ds = CropDataset(c[keep], sub_g, class_map, train=False)
        return DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    train_loader = make_loader_train(july_crops[train_flag], july_gids[train_flag])
    val_loader   = make_loader_val(july_crops[val_flag],     july_gids[val_flag])
    n_val_seen = sum(1 for g in july_gids[val_flag] if g in class_map)
    n_val_cls  = len(set(gid for gid in july_gids[val_flag] if gid in class_map))
    print(f"Val crops with seen identity: {n_val_seen} / {n_val_crops} ({n_val_cls} classes)")

    device = torch.device(args.device)
    model  = ReIDModel(n_classes=n_classes, emb_dim=args.emb_dim).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    log_path = os.path.join(data_dir, "train_log.csv")
    best_val   = float("inf")
    best_rank1 = 0.0
    best_path  = args.out

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_rank1"])

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for crops_b, labels_b in train_loader:
            crops_b  = crops_b.to(device)
            labels_b = labels_b.to(device)

            emb, logits = model(crops_b)

            # Remap labels to local 0-based for CE (labels are already 0-based).
            trip_loss = batch_hard_triplet_loss(emb, labels_b, margin=0.3)
            ce_loss   = F.cross_entropy(logits, labels_b)
            loss = 0.7 * trip_loss + 0.3 * ce_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_sum += loss.item()
            n_batches += 1

        sched.step()
        train_loss = train_loss_sum / max(n_batches, 1)

        # ── Val ────────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_batches  = 0
        with torch.no_grad():
            for crops_b, labels_b in val_loader:
                crops_b  = crops_b.to(device)
                labels_b = labels_b.to(device)
                emb, logits = model(crops_b)
                trip_loss = batch_hard_triplet_loss(emb, labels_b, margin=0.3)
                ce_loss   = F.cross_entropy(logits, labels_b)
                loss = 0.7 * trip_loss + 0.3 * ce_loss
                val_loss_sum += loss.item()
                val_batches  += 1

        val_loss  = val_loss_sum / max(val_batches, 1)
        val_rank1 = eval_rank1(model, val_loader, device)

        if val_rank1 >= best_rank1:
            best_rank1 = val_rank1
            torch.save(model.state_dict(), best_path)
        if val_loss < best_val:
            best_val = val_loss

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"rank1={val_rank1:.3f}  "
              f"{'*' if val_loss == best_val else ''}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_rank1:.3f}"])

    print(f"\nBest val rank-1: {best_rank1:.4f}  → {best_path}")
    print(f"Log: {log_path}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",   type=int,   default=60)
    p.add_argument("--batch",    type=int,   default=64,  help="val batch size")
    p.add_argument("--P",        type=int,   default=16,  help="classes per train batch")
    p.add_argument("--K",        type=int,   default=4,   help="samples per class per batch")
    p.add_argument("--emb-dim",  type=int,   default=128)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--data-dir", type=str,   default="reid_data")
    p.add_argument("--out",      type=str,   default="reid_data/model.pt")
    p.add_argument("--device",   type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    print(f"Device: {args.device}")
    train(args)
