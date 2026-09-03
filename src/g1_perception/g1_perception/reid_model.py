"""
reid_model.py — PointNet-based person ReID embedding for LiDAR crops.

Input:  (B, N, 3) float32 — centred, de-yawed xyz  (N = 256 by default)
Output: (B, 128)  float32 — L2-normalised embedding

Loss helpers:
  batch_hard_triplet_loss(emb, labels, margin)  — batch-hard semi-hard triplet
  focal_cross_entropy(emb, labels, weight)       — optional classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# PointNet encoder
# ────────────────────────────────────────────────────────────────────────────

class PointNetEncoder(nn.Module):
    """Shared-MLP PointNet, global max-pool → 128-d L2-normalised embedding."""

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        # Shared MLP via 1D-conv (each point processed independently).
        self.conv1 = nn.Conv1d(3,   64,  1)
        self.conv2 = nn.Conv1d(64,  128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(256)

        # Global descriptor → embedding.
        self.fc1   = nn.Linear(256, 256)
        self.bn4   = nn.BatchNorm1d(256)
        self.fc2   = nn.Linear(256, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 3) → (B, emb_dim) L2-normalised."""
        x = x.transpose(1, 2)                     # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2).values                   # global max-pool → (B, 256)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return F.normalize(x, dim=1)               # L2 unit sphere


# ────────────────────────────────────────────────────────────────────────────
# Optional classification head (cross-entropy auxiliary loss)
# ────────────────────────────────────────────────────────────────────────────

class ReIDModel(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int = 128):
        super().__init__()
        self.encoder = PointNetEncoder(emb_dim)
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x: torch.Tensor):
        """Returns (emb, logits). emb is L2-normalised."""
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return emb, logits


# ────────────────────────────────────────────────────────────────────────────
# Batch-hard triplet loss
# ────────────────────────────────────────────────────────────────────────────

def batch_hard_triplet_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Batch-hard triplet loss.
    emb:    (B, D) L2-normalised embeddings.
    labels: (B,)   int — class label (pseudo_id, remapped to 0-based).
    Returns scalar loss (mean over valid anchors).
    """
    # Pairwise cosine similarity matrix (embs are unit-normalised → dot = cosine).
    sim = emb @ emb.t()                            # (B, B)
    dist = 1.0 - sim                               # cosine distance ∈ [0, 2]

    labels = labels.view(-1, 1)
    same = labels.eq(labels.t())                   # (B, B) bool

    # Hardest positive: same class, max distance.
    pos_dist = dist.masked_fill(~same, -1e9).max(dim=1).values

    # Hardest negative: different class, min distance.
    neg_dist = dist.masked_fill(same, 1e9).min(dim=1).values

    loss = F.relu(pos_dist - neg_dist + margin)

    # Only count anchors that have at least one positive AND one negative.
    valid = same.sum(dim=1).gt(1) & (~same).sum(dim=1).gt(0)
    if valid.sum() == 0:
        return loss.mean() * 0.0
    return loss[valid].mean()
