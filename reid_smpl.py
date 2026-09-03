#!/usr/bin/env python3
"""
reid_smpl.py — SMPL β-shape-based person Re-ID.

The learned 128-d PointNet embeddings collapse to ~99.6% cosine similarity
across distinct identities (useless for ReID). SMPL β (shape) parameters —
10-d vectors that encode body proportions such as height, weight, and
shoulder width — are physically discriminative between people and can be
regressed from LiDAR point clouds independently of pose. This module tracks
and re-identifies persons using β instead of learned embeddings.

Pure numpy/scipy — no torch dependency (β extraction happens elsewhere;
this module only consumes pre-computed β vectors).

Run:
    python3 reid_smpl.py --session 2026-07-29_17-21-48 --data-dir reid_data
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment


# ── utility functions ──────────────────────────────────────────────────────

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def beta_distance_matrix(betas_a, betas_b):
    """Pairwise 1 - cosine_sim matrix. Shape: (len(a), len(b))."""
    A = np.asarray(betas_a, dtype=float)
    B = np.asarray(betas_b, dtype=float)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    cos = A_norm @ B_norm.T
    return 1.0 - cos


def estimate_height_from_beta(beta):
    """Rough height estimate from β[0] (SMPL scale component).
    β[0] ≈ 0 → ~1.7m average. Each unit ≈ 0.1m change.
    Returns estimated height in meters."""
    beta = np.asarray(beta, dtype=float)
    return 1.7 + 0.1 * float(beta[0])


def summarize_tracks(tracks):
    """Print summary: track count, lengths, avg β distances between tracks."""
    print(f"[reid_smpl] {len(tracks)} tracks")
    for t in tracks:
        h = estimate_height_from_beta(t["avg_beta"])
        print(f"  track {t['id']:3d}  len={len(t['fi']):4d}  "
              f"frames=[{t['fi'][0]}..{t['fi'][-1]}]  "
              f"est_height={h:.2f}m")
    if len(tracks) > 1:
        avg_betas = np.array([t["avg_beta"] for t in tracks])
        dist = beta_distance_matrix(avg_betas, avg_betas)
        n = len(tracks)
        iu = np.triu_indices(n, k=1)
        print(f"[reid_smpl] pairwise avg-β distance: "
              f"mean={dist[iu].mean():.4f}  min={dist[iu].min():.4f}  "
              f"max={dist[iu].max():.4f}")


# ── gallery ─────────────────────────────────────────────────────────────────

class SMPLReID:
    """Gallery of known persons keyed by averaged SMPL β vector."""

    def __init__(self, gallery_betas=None, gallery_ids=None):
        """Initialize with optional pre-existing gallery.

        Args:
            gallery_betas: (M, 10) array of β vectors, one per known person.
            gallery_ids:   list[int] of length M, person ids matching rows.
        """
        self._betas = {}   # person_id -> running-average beta (np.ndarray)
        self._counts = {}  # person_id -> number of observations averaged in
        if gallery_betas is not None and gallery_ids is not None:
            for beta, pid in zip(gallery_betas, gallery_ids):
                self.add_to_gallery(beta, pid)

    def add_to_gallery(self, beta, person_id):
        """Add a person's β to gallery. If person_id exists, update running average."""
        beta = np.asarray(beta, dtype=float)
        if person_id in self._betas:
            n = self._counts[person_id]
            self._betas[person_id] = (self._betas[person_id] * n + beta) / (n + 1)
            self._counts[person_id] = n + 1
        else:
            self._betas[person_id] = beta.copy()
            self._counts[person_id] = 1

    def identify(self, query_beta, threshold=0.85):
        """Match query β against gallery.

        Returns (person_id, similarity) — best match if similarity ≥ threshold,
        else (-1, best_sim).
        """
        if not self._betas:
            return -1, 0.0
        best_id, best_sim = -1, -1.0
        for pid, beta in self._betas.items():
            sim = cosine_similarity(query_beta, beta)
            if sim > best_sim:
                best_id, best_sim = pid, sim
        if best_sim >= threshold:
            return best_id, best_sim
        return -1, best_sim

    def gallery_size(self):
        """Number of unique persons in gallery."""
        return len(self._betas)


# ── tracker ─────────────────────────────────────────────────────────────────

class SMPLTracker:
    """Hungarian-assignment multi-person tracker using SMPL β for identity."""

    def __init__(self, beta_weight=0.5, pos_gate=2.0, cos_thresh=0.85,
                 short_gap=5, min_len=5, max_range=5.0):
        """
        Args:
            beta_weight: weight of β similarity in cost (vs position)
            pos_gate: max xy distance (m) to consider position match
            cos_thresh: cosine threshold for long-gap re-ID
            short_gap: frames gap threshold for short vs long gap
            min_len: minimum track length to keep
            max_range: discard detections beyond this range from sensor
        """
        self.beta_weight = beta_weight
        self.pos_gate = pos_gate
        self.cos_thresh = cos_thresh
        self.short_gap = short_gap
        self.min_len = min_len
        self.max_range = max_range

    def track(self, betas, boxes, scores, frame_indices):
        """Run full tracking over a session.

        Args:
            betas: (N, 10) array of β vectors per detection
            boxes: (N, 7) array of 3D boxes per detection
            scores: (N,) array of confidence scores
            frame_indices: (N,) array of frame indices

        Returns:
            tracks: list of dicts, each with:
                - id: track id
                - fi: list of frame indices
                - box: list of 7-d boxes
                - beta: list of β vectors
                - avg_beta: averaged β for this track
                - score: list of scores
        """
        betas = np.asarray(betas, dtype=float)
        boxes = np.asarray(boxes, dtype=float)
        scores = np.asarray(scores, dtype=float)
        frame_indices = np.asarray(frame_indices, dtype=int)

        if self.max_range > 0 and len(boxes):
            r = np.hypot(boxes[:, 0], boxes[:, 1])
            keep = r <= self.max_range
            betas, boxes, scores, frame_indices = (
                betas[keep], boxes[keep], scores[keep], frame_indices[keep])

        if len(frame_indices) == 0:
            return []

        n_frames = int(frame_indices.max()) + 1
        # Bucket detections by frame.
        frame_idx_of = {}
        for i, fi in enumerate(frame_indices):
            frame_idx_of.setdefault(int(fi), []).append(i)

        tracks = {}
        next_tid = 1
        EMA_ALPHA = 0.3   # β anchor update rate

        def new_track(fi, box7, beta, score):
            nonlocal next_tid
            tid = next_tid
            next_tid += 1
            tracks[tid] = {
                "id": tid,
                "last_fi": fi,
                "last_xy": box7[:2].copy(),
                "anchor": beta.copy(),
                "fi": [fi],
                "box": [box7.tolist()],
                "beta": [beta.tolist()],
                "score": [float(score)],
            }

        for fi in range(n_frames):
            det_i = frame_idx_of.get(fi)
            if not det_i:
                continue
            B = boxes[det_i]      # (K, 7)
            E = betas[det_i]      # (K, 10)
            S = scores[det_i]     # (K,)
            K = len(det_i)

            active_tids = [t for t, tr in tracks.items()
                          if fi - tr["last_fi"] <= self.short_gap]
            dormant_tids = [t for t, tr in tracks.items()
                            if fi - tr["last_fi"] > self.short_gap]

            # ── Phase 1: active tracks — Hungarian (pos + β cost) ────────────
            used = set()
            if active_tids and K:
                A_xy = np.array([tracks[t]["last_xy"] for t in active_tids])   # (A, 2)
                A_beta = np.array([tracks[t]["anchor"] for t in active_tids])  # (A, 10)
                det_xy = B[:, :2]   # (K, 2)

                pos_dist = np.linalg.norm(A_xy[:, None, :] - det_xy[None, :, :], axis=-1)
                beta_dist = beta_distance_matrix(A_beta, E)   # (A, K), 1 - cos ∈ [0, 2]

                gate_ok = pos_dist < self.pos_gate
                pos_norm = pos_dist / self.pos_gate
                cost = ((1 - self.beta_weight) * pos_norm +
                        self.beta_weight * beta_dist * 0.5)
                cost[~gate_ok] = 1e9

                row_ind, col_ind = linear_sum_assignment(cost)
                for ri, ci in zip(row_ind, col_ind):
                    if cost[ri, ci] >= 1e6:
                        continue
                    tid = active_tids[ri]
                    tr = tracks[tid]
                    tr["last_fi"] = fi
                    tr["last_xy"] = B[ci, :2].copy()
                    tr["anchor"] = (1 - EMA_ALPHA) * tr["anchor"] + EMA_ALPHA * E[ci]
                    tr["fi"].append(fi)
                    tr["box"].append(B[ci].tolist())
                    tr["beta"].append(E[ci].tolist())
                    tr["score"].append(float(S[ci]))
                    used.add(ci)

            # ── Phase 2: dormant tracks — re-ID by β cosine only ──────────────
            remaining = [j for j in range(K) if j not in used]
            if dormant_tids and remaining:
                D_beta = np.array([tracks[t]["anchor"] for t in dormant_tids])  # (D, 10)
                R_beta = E[remaining]                                            # (R, 10)
                cost = beta_distance_matrix(D_beta, R_beta)
                cost[cost > (1.0 - self.cos_thresh)] = 1e9

                row_ind, col_ind = linear_sum_assignment(cost)
                matched_r = set()
                for ri, ci in zip(row_ind, col_ind):
                    if cost[ri, ci] >= 1e6:
                        continue
                    tid = dormant_tids[ri]
                    tr = tracks[tid]
                    j = remaining[ci]
                    tr["last_fi"] = fi
                    tr["last_xy"] = B[j, :2].copy()
                    tr["anchor"] = (1 - EMA_ALPHA) * tr["anchor"] + EMA_ALPHA * E[j]
                    tr["fi"].append(fi)
                    tr["box"].append(B[j].tolist())
                    tr["beta"].append(E[j].tolist())
                    tr["score"].append(float(S[j]))
                    matched_r.add(ci)
                remaining = [remaining[ci] for ci in range(len(remaining))
                            if ci not in matched_r]

            # ── Phase 3: new tracks for unmatched detections ───────────────────
            for j in remaining:
                new_track(fi, B[j], E[j], S[j])

        # ── Build output, drop short tracks ─────────────────────────────────
        out = []
        for tid, tr in sorted(tracks.items()):
            if len(tr["fi"]) < self.min_len:
                continue
            avg_beta = np.mean(np.array(tr["beta"]), axis=0)
            out.append({
                "id": tid,
                "fi": tr["fi"],
                "box": tr["box"],
                "beta": tr["beta"],
                "avg_beta": avg_beta.tolist(),
                "score": tr["score"],
            })

        out.sort(key=lambda t: -len(t["fi"]))
        for rank, t in enumerate(out, 1):
            t["id"] = rank

        return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def _load_session_betas(data_dir, session):
    """Load pre-computed β / box / score / frame-index arrays for a session.

    Expects files named smpl_<session>_beta.npy, _box.npy, _score.npy, _fi.npy
    under data_dir (same convention as the emb_<session>_*.npy files produced
    by reid_embed_server.py, but for SMPL β instead of learned embeddings).
    """
    prefix = os.path.join(data_dir, f"smpl_{session}")
    beta_p = prefix + "_beta.npy"
    box_p = prefix + "_box.npy"
    score_p = prefix + "_score.npy"
    fi_p = prefix + "_fi.npy"
    for p in (beta_p, box_p, fi_p):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing pre-computed SMPL data: {p}")
    betas = np.load(beta_p)
    boxes = np.load(box_p)
    frame_indices = np.load(fi_p)
    if os.path.exists(score_p):
        scores = np.load(score_p)
    else:
        scores = np.ones(len(frame_indices), dtype=float)
    return betas, boxes, scores, frame_indices


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="2026-07-29_17-21-48")
    ap.add_argument("--data-dir", default="reid_data")
    ap.add_argument("--beta-weight", type=float, default=0.5,
                    help="Weight of β similarity in cost (vs position)")
    ap.add_argument("--pos-gate", type=float, default=2.0,
                    help="Max xy distance (m) to consider a position match")
    ap.add_argument("--cos-thresh", type=float, default=0.85,
                    help="Cosine threshold for long-gap re-ID")
    ap.add_argument("--short-gap", type=int, default=5,
                    help="Frames gap threshold for short vs long gap")
    ap.add_argument("--min-len", type=int, default=5,
                    help="Minimum track length (observed frames) to keep")
    ap.add_argument("--max-range", type=float, default=5.0,
                    help="Discard detections farther than this many metres (0=off)")
    ap.add_argument("--save", default=None,
                    help="Optional path to save tracks as JSON")
    args = ap.parse_args()

    betas, boxes, scores, frame_indices = _load_session_betas(
        args.data_dir, args.session)
    print(f"[reid_smpl] loaded {len(frame_indices)} detections for session {args.session}")

    tracker = SMPLTracker(beta_weight=args.beta_weight, pos_gate=args.pos_gate,
                          cos_thresh=args.cos_thresh, short_gap=args.short_gap,
                          min_len=args.min_len, max_range=args.max_range)
    tracks = tracker.track(betas, boxes, scores, frame_indices)

    summarize_tracks(tracks)

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"session": args.session, "tracks": tracks}, f)
        print(f"[reid_smpl] saved {len(tracks)} tracks to {args.save}")


if __name__ == "__main__":
    main()
