# Handoff: SMPL-Driven LiDAR Person ReID

## Status: READY FOR IMPLEMENTATION

**Created**: 2026-09-02
**Priority**: HIGH
**Estimated Effort**: 7 days

---

## Problem Summary

The current 128-d embedding approach fails completely:
- All embeddings are 99.6% similar (USELESS for ReID)
- Cannot distinguish between different people
- Result: All detections cluster as 1 "person"

**Root Cause**: Models trained on outdoor driving data learn "human shape" features, not individual identity.

---

## Solution: SMPL Shape Parameters

Use SMPL (Skinned Multi-Person Linear Model) which encodes body shape as a 10-d vector β:
- β[0]: Overall scale (height)
- β[1-3]: Body proportions (weight, torso length, limb ratios)
- β[4-6]: Shape variations (shoulder width, hip width, etc.)
- β[7-9]: Fine-grained shape details

**These are PHYSICAL PROPERTIES that differ between people and are stable across poses.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  SMPL-DRIVEN REID PIPELINE                                  │
│                                                              │
│  Phase 1: Detection (KEEP VoxelNext)                        │
│    LiDAR → VoxelNext → 3D Bounding Boxes                    │
│                                                              │
│  Phase 2: SMPL Estimation (NEW - LiDAR-HMR)                │
│    Person Crop → LiDAR-HMR → SMPL (β, θ, mesh)             │
│                                                              │
│  Phase 3: ReID (NEW)                                        │
│    SMPL β (10-d) → Compare with Gallery → Identity          │
│                                                              │
│  Phase 4: Pose Estimation (BONUS)                           │
│    SMPL θ (72-d) → 3D Skeleton (24 joints)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components

| Component | Purpose | Status | Action |
|-----------|---------|--------|--------|
| **VoxelNext** | 3D detection | ✅ Working | KEEP |
| **LiDAR-HMR** | SMPL estimation | ❌ Not installed | CLONE & SETUP |
| **SMPL ReID** | Identity matching | ❌ Not implemented | CREATE |

---

## Implementation Steps

### Step 1: Environment Setup (Day 1)

```bash
# Clone LiDAR-HMR
cd ~/Projects/thesis/g1_perception_ws
git clone https://github.com/soullessrobot/LiDAR-HMR.git
cd LiDAR-HMR

# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install -r requirements.txt

# Install Point Transformer-v2
git clone https://github.com/Pointcept/PointTransformerV2.git
cd PointTransformerV2
pip install -e .
cd ..

# Install ChamferDistancePytorch
git clone https://github.com/ThibaultGROUEIX/ChamferDistancePytorch.git
cd ChamferDistancePytorch
pip install -e .
cd ..

# Download SMPL model
# Visit: https://smpl.is.tue.mpg.de/
# Download SMPL model files
# Place in: smplx_models/ folder
```

### Step 2: Create Scripts (Day 2-3)

**Script 1: extract_smpl.py**

```python
#!/usr/bin/env python3
"""
extract_smpl.py — Extract SMPL parameters from LiDAR detections.

Input: NPZ file with VoxelNext detections + point cloud frames
Output: SMPL β (10-d) + θ (72-d) for each detection
"""
import os
import torch
import numpy as np

class SMPLExtractor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        # Import LiDAR-HMR model
        import sys
        sys.path.insert(0, 'LiDAR-HMR')
        from models.lidar_hmr import LiDAR_HMR
        self.model = LiDAR_HMR()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def extract_from_crop(self, crop_points):
        """
        Extract SMPL parameters from a person crop.

        Args:
            crop_points: (128, 3) numpy array

        Returns:
            beta: (10,) SMPL shape parameters
            theta: (72,) SMPL pose parameters
            mesh: (6890, 3) 3D mesh vertices
        """
        # Convert to tensor
        points_tensor = torch.from_numpy(crop_points).float().unsqueeze(0)
        points_tensor = points_tensor.to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(points_tensor)

        beta = output['beta'].cpu().numpy().squeeze()
        theta = output['theta'].cpu().numpy().squeeze()
        mesh = output['mesh'].cpu().numpy().squeeze()

        return beta, theta, mesh

    def extract_from_session(self, npz_path, frames_dir, min_score=0.2):
        """
        Extract SMPL parameters for all detections in a session.

        Returns:
            dict with keys: 'beta', 'theta', 'mesh', 'fi', 'box', 'score'
        """
        npz = np.load(npz_path, allow_pickle=True)
        n_frames = len(npz['frame_files'])

        all_beta, all_theta, all_mesh = [], [], []
        all_fi, all_box, all_score = [], [], []

        for fi in range(n_frames):
            boxes = np.asarray(npz['pred_boxes'][fi])
            labels = np.asarray(npz['pred_labels'][fi])
            scores = np.asarray(npz['pred_scores'][fi])

            # Filter for pedestrians
            mask = (labels == 2) & (scores >= min_score)
            if not mask.any():
                continue

            # Load point cloud
            frame_path = os.path.join(frames_dir, str(npz['frame_files'][fi]))
            if not os.path.exists(frame_path):
                continue
            pts = np.load(frame_path)[:, :3].astype(np.float32)

            # Extract crops and SMPL for each detection
            for box, score in zip(boxes[mask], scores[mask]):
                crop = self._extract_crop(pts, box, n_pts=128)
                beta, theta, mesh = self.extract_from_crop(crop)

                all_beta.append(beta)
                all_theta.append(theta)
                all_mesh.append(mesh)
                all_fi.append(fi)
                all_box.append(box)
                all_score.append(score)

        return {
            'beta': np.array(all_beta) if all_beta else np.array([]),
            'theta': np.array(all_theta) if all_theta else np.array([]),
            'mesh': np.array(all_mesh) if all_mesh else np.array([]),
            'fi': np.array(all_fi) if all_fi else np.array([]),
            'box': np.array(all_box) if all_box else np.array([]),
            'score': np.array(all_score) if all_score else np.array([]),
        }

    def _extract_crop(self, pts_xyz, box7, n_pts=128, rng=None):
        """Extract a point cloud crop from a 3D bounding box."""
        if rng is None:
            rng = np.random.default_rng(0)
        cx, cy, cz, dx, dy, dz, yaw = box7.astype(float)
        valid = ~((pts_xyz[:, 0] == 0.0) & (pts_xyz[:, 1] == 0.0))
        pts = pts_xyz[valid]
        pts = pts - np.array([cx, cy, cz])
        c, s = np.cos(-yaw), np.sin(-yaw)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="2026-07-29_17-21-48")
    parser.add_argument("--lidar-dir", default=os.path.expanduser("~/Projects/Thesis/Lidar Data"))
    parser.add_argument("--model", required=True, help="Path to LiDAR-HMR weights")
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--out-dir", default="reid_data")
    args = parser.parse_args()

    # Extract SMPL
    extractor = SMPLExtractor(args.model)
    npz_path = os.path.join(args.lidar_dir, f"{args.session}_frames_voxelnext.npz")
    frames_dir = os.path.join(args.lidar_dir, "frames", args.session)

    print(f"Extracting SMPL from {args.session}...")
    smpl_data = extractor.extract_from_session(npz_path, frames_dir, args.min_score)

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"smpl_{args.session}")
    np.save(f"{stem}_beta.npy", smpl_data['beta'])
    np.save(f"{stem}_theta.npy", smpl_data['theta'])
    np.save(f"{stem}_mesh.npy", smpl_data['mesh'])
    np.save(f"{stem}_fi.npy", smpl_data['fi'])
    np.save(f"{stem}_box.npy", smpl_data['box'])
    np.save(f"{stem}_score.npy", smpl_data['score'])

    print(f"Saved {len(smpl_data['beta'])} SMPL parameters to {stem}_*.npy")
```

**Script 2: reid_smpl.py**

```python
#!/usr/bin/env python3
"""
reid_smpl.py — Person ReID using SMPL shape parameters.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SMPLReID:
    def __init__(self, gallery_betas=None, gallery_ids=None):
        """
        Args:
            gallery_betas: (N, 10) SMPL shape parameters for gallery
            gallery_ids: (N,) person IDs for gallery
        """
        self.gallery_betas = gallery_betas
        self.gallery_ids = gallery_ids

    def add_to_gallery(self, beta, person_id):
        """Add a person to the gallery."""
        if self.gallery_betas is None:
            self.gallery_betas = beta.reshape(1, -1)
            self.gallery_ids = np.array([person_id])
        else:
            self.gallery_betas = np.vstack([self.gallery_betas, beta.reshape(1, -1)])
            self.gallery_ids = np.append(self.gallery_ids, person_id)

    def identify(self, query_beta, threshold=0.9):
        """
        Identify a person based on SMPL β.

        Returns:
            person_id: ID of matched person, or -1 if no match
            similarity: Best similarity score
        """
        if self.gallery_betas is None:
            return -1, 0.0

        # Compute similarities
        sims = cosine_similarity(query_beta.reshape(1, -1), self.gallery_betas)[0]
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]

        if best_sim > threshold:
            return self.gallery_ids[best_idx], best_sim
        else:
            return -1, best_sim

    def track_with_smpl(self, detections, threshold=0.9):
        """
        Track persons using SMPL β.

        Args:
            detections: list of dicts with 'beta', 'box', 'score', 'fi'

        Returns:
            tracks: dict mapping track_id -> list of detection indices
        """
        tracks = {}
        track_betas = {}  # track_id -> list of β vectors (for averaging)

        for det_idx, det in enumerate(detections):
            beta = det['beta']
            box = det['box']
            fi = det['fi']

            best_track = -1
            best_score = -1

            # Compare with existing tracks
            for track_id, betas in track_betas.items():
                # Average β for this track
                avg_beta = np.mean(betas, axis=0)
                sim = cosine_similarity(beta.reshape(1, -1), avg_beta.reshape(1, -1))[0, 0]

                if sim > best_score:
                    best_score = sim
                    best_track = track_id

            # Assign to track or create new
            if best_score > threshold and best_track >= 0:
                # Assign to existing track
                tracks[best_track].append(det_idx)
                track_betas[best_track].append(beta)
            else:
                # Create new track
                new_track_id = len(tracks)
                tracks[new_track_id] = [det_idx]
                track_betas[new_track_id] = [beta]

        return tracks
```

**Script 3: evaluate_smpl_reid.py**

```python
#!/usr/bin/env python3
"""
evaluate_smpl_reid.py — Evaluate SMPL-based ReID accuracy.
"""
import numpy as np
import os
import sys
sys.path.insert(0, '.')
from reid_smpl import SMPLReID

def evaluate_reid(smpl_data, ground_truth_ids):
    """
    Evaluate ReID accuracy using SMPL β.

    Metrics:
    - Rank-1 accuracy
    - mAP
    - CMC curve
    """
    reid = SMPLReID()

    # Build gallery (first appearance of each person)
    gallery_added = set()
    for idx, (beta, person_id) in enumerate(zip(smpl_data['beta'], ground_truth_ids)):
        if person_id not in gallery_added:
            reid.add_to_gallery(beta, person_id)
            gallery_added.add(person_id)

    # Query all detections
    correct = 0
    total = 0
    for idx, (beta, true_id) in enumerate(zip(smpl_data['beta'], ground_truth_ids)):
        pred_id, sim = reid.identify(beta, threshold=0.9)
        if pred_id == true_id:
            correct += 1
        total += 1

    rank1_accuracy = correct / total
    print(f"Rank-1 Accuracy: {rank1_accuracy:.2%}")
    return rank1_accuracy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="2026-07-29_17-21-48")
    parser.add_argument("--smpl-dir", default="reid_data")
    parser.add_argument("--gt-file", required=True, help="Ground truth person IDs")
    args = parser.parse_args()

    # Load SMPL data
    stem = os.path.join(args.smpl_dir, f"smpl_{args.session}")
    smpl_data = {
        'beta': np.load(f"{stem}_beta.npy"),
        'theta': np.load(f"{stem}_theta.npy"),
        'fi': np.load(f"{stem}_fi.npy"),
        'box': np.load(f"{stem}_box.npy"),
    }

    # Load ground truth
    ground_truth_ids = np.load(args.gt_file)

    # Evaluate
    evaluate_reid(smpl_data, ground_truth_ids)
```

---

## Key Design Decisions

### Decision 1: Keep VoxelNext for Detection

**Rationale**: VoxelNext is optimized for 3D object detection. LiDAR-HMR is optimized for SMPL estimation. They serve different purposes and should be used together.

**Flow**: VoxelNext (detection) → LiDAR-HMR (SMPL) → SMPL ReID (identity)

### Decision 2: Use SMPL β for ReID, Not 128-d Embeddings

**Rationale**: SMPL β encodes body proportions (height, weight, limb lengths) that are physically discriminative. The 128-d embeddings fail because they're trained to recognize "human shape" not individual identity.

**Evidence**: ReID3D achieves 94% Rank-1 accuracy using SMPL pre-training.

### Decision 3: Temporal Tracking for Training, Not Inference

**Rationale**: Temporal tracking provides free supervision (pseudo-labels) for contrastive learning. At inference, we want single-shot ReID without tracking dependencies.

**Implementation**: Train with InfoNCE loss using tracking pseudo-labels. Inference uses only β comparison.

### Decision 4: Fine-tune on Indoor Data

**Rationale**: LiDAR-HMR is trained on outdoor data (Waymo, SLOPER4D). Indoor environments have different characteristics (closer range, different point density). Fine-tuning improves accuracy.

---

## Success Metrics

| Metric | Current (128-d) | Target (SMPL β) | Improvement |
|--------|-----------------|-----------------|-------------|
| **Rank-1 Accuracy** | ~0% (1 cluster) | >80% | ✅ Critical |
| **mAP** | ~0% | >60% | ✅ Critical |
| **Unique People Detected** | 1 | N (ground truth) | ✅ Critical |
| **Pose Estimation** | ❌ None | ✅ 24 joints | ✅ Bonus |
| **Interpretability** | ❌ Black box | ✅ Height, weight | ✅ Bonus |

---

## Dependencies

### Required Software
- Python 3.12
- PyTorch 2.0+ with CUDA
- Point Transformer-v2
- ChamferDistancePytorch
- SMPL model files

### Required Data
- LiDAR-HMR pre-trained weights (Waymo/SLOPER4D)
- Indoor dataset with person ID annotations (for fine-tuning)
- Your existing VoxelNext detection data

### Hardware
- GPU: RTX 4060 (8GB) or better
- RAM: 16GB minimum
- Storage: 10GB for models and data

---

## Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Environment setup, clone LiDAR-HMR | Working LiDAR-HMR installation |
| 2 | Create extract_smpl.py | SMPL extraction from detections |
| 3 | Create reid_smpl.py | SMPL-based ReID module |
| 4 | Integrate with web server | Updated reid_web_server.py |
| 5 | Test on existing data | Initial evaluation results |
| 6 | Fine-tune on indoor data | Improved indoor accuracy |
| 7 | Evaluation & documentation | Final metrics, handoff docs |

---

## References

1. **LiDAR-HMR**: https://github.com/soullessrobot/LiDAR-HMR
2. **ReID3D**: https://github.com/GWxuan/ReID3D
3. **SMPL Model**: https://smpl.is.tue.mpg.de/
4. **Point Transformer-v2**: https://github.com/Pointcept/PointTransformerV2

---

**Document Version**: 1.0
**Created**: 2026-09-02
**Status**: Ready for implementation
