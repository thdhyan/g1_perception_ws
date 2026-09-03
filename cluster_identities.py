#!/usr/bin/env python3
"""
cluster_identities.py — estimate the number of distinct people in a session
by clustering per-track SMPL beta anchors (fixes track fragmentation: a
person who was lost and re-tracked as multiple track ids still collapses
to one identity cluster).

Run against a live reid_embed_server.py --smpl-mode instance:
    python3 cluster_identities.py --port 8767
"""
import argparse
import json
import urllib.request

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


def cluster_tracks(tracks, k_max=None):
    anchors = np.array([t["anchor"] for t in tracks])
    norm = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)
    dist = np.clip(1 - norm @ norm.T, 0, None)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")

    k_max = k_max or (len(tracks) - 1)
    best_k, best_s, scores = 2, -1, {}
    for k in range(2, min(k_max, len(tracks) - 1) + 1):
        labels = fcluster(Z, k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(dist, labels, metric="precomputed")
        scores[k] = s
        if s > best_s:
            best_s, best_k = s, k

    labels = fcluster(Z, best_k, criterion="maxclust")
    return best_k, best_s, labels, scores


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--k-max", type=int, default=None)
    args = ap.parse_args()

    d = json.load(urllib.request.urlopen(f"http://{args.host}:{args.port}/api/tracks"))
    tracks = d["tracks"]
    if len(tracks) < 3:
        print(f"only {len(tracks)} tracks — too few to cluster meaningfully")
        return

    best_k, best_s, labels, scores = cluster_tracks(tracks, args.k_max)
    print(f"session: {d['session']}  ({len(tracks)} tracks)")
    for k, s in sorted(scores.items()):
        marker = "  <-- best" if k == best_k else ""
        print(f"  k={k}: silhouette={s:.3f}{marker}")
    print(f"\n>>> estimated {best_k} distinct people (silhouette={best_s:.3f})")
    by_cluster = {}
    for t, lab in zip(tracks, labels):
        by_cluster.setdefault(int(lab), []).append((t["tid"], t["len"]))
    for lab, members in sorted(by_cluster.items()):
        members.sort(key=lambda m: -m[1])
        tag = ", ".join(f"T{tid}(len={l})" for tid, l in members)
        print(f"   person {lab}: {tag}")


if __name__ == "__main__":
    main()
