# Plan: SMPL-Driven LiDAR Person ReID Integration

## Executive Summary

Replace the failing 128-d embedding approach with **SMPL body shape parameters** for person ReID. This simultaneously solves:
1. **Person discrimination** (SMPL β encodes body proportions)
2. **Pose estimation** (SMPL θ gives skeleton)
3. **ReID** (compare β vectors across frames/sessions)

**Key insight**: SMPL shape parameters (β: 10-d) are discriminative because they encode height, weight, shoulder width, hip width, and limb lengths - physical properties that differ between people.

---

## Problem Statement

### Current System Failure

```
Current Pipeline:
  LiDAR → VoxelNext Detection → Crop → PointNet → 128-d embedding → ReID

Failure Mode:
  - All 128-d embeddings are 99.6% similar (USELESS for ReID)
  - Cannot distinguish between different people
  - Result: All detections cluster as 1 "person"
```

### Root Cause

The existing PointNet model (173K params) and point-cloud-reid model (109M params) were both trained on **outdoor driving data** (nuScenes/Waymo). They learn to recognize "human shape" features, not individual identity. When applied to indoor data, all pedestrians map to nearly the same embedding.

### Solution: SMPL Shape Parameters

SMPL (Skinned Multi-Person Linear Model) encodes body shape as a 10-d vector β:
- β[0]: Overall scale (height)
- β[1-3]: Body proportions (weight, torso length, limb ratios)
- β[4-6]: Shape variations (shoulder width, hip width, etc.)
- β[7-9]: Fine-grained shape details

**These are PHYSICAL PROPERTIES that differ between people and are stable across poses.**

---

## Architecture: SMPL-Driven Pipeline

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    SMPL-DRIVEN REID PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

Phase 1: Detection (KEEP VoxelNext)
────────────────────────────────────
  LiDAR Point Cloud (.npy)
      ↓
  VoxelNext Detection
      ↓
  3D Bounding Boxes + Scores + Labels
      ↓
  Filter (label=2, score≥0.2, range≤15m)

Phase 2: SMPL Estimation (NEW)
──────────────────────────────
  Person Crop (128 × 3 points)
      ↓
  LiDAR-HMR Network
      ↓
  SMPL Parameters:
    - β: Shape (10-d) → IDENTITY
    - θ: Pose (72-d) → SKELETON
    - Mesh: 3D Body Surface (6890 × 3)

Phase 3: ReID (NEW)
───────────────────
  SMPL β Vector (10-d)
      ↓
  Compare with Gallery (cosine similarity)
      ↓
  Person Identity

Phase 4: Pose Estimation (BONUS)
────────────────────────────────
  SMPL θ Vector (72-d)
      ↓
  3D Skeleton (24 joints)
      ↓
  Human-Robot Interaction
```

### Why VoxelNext is Still Needed

| Component | VoxelNext | LiDAR-HMR | Can Replace? |
|-----------|-----------|-----------|--------------|
| **3D Detection** | ✅ Primary task | ❌ Not designed for this | **NO** |
| **Person Localization** | ✅ Bounding boxes | ❌ Needs crop input | **NO** |
| **SMPL Estimation** | ❌ Not supported | ✅ Primary task | **YES** |
| **Pose Estimation** | ❌ Limited | ✅ Full skeleton | **YES** |
| **ReID** | ❌ No | ✅ Via β parameters | **YES** |

**Conclusion**: VoxelNext remains essential for **detection and localization**. LiDAR-HMR handles **SMPL estimation and ReID**. They are complementary, not competing.

---

## Detailed Architecture

### Component 1: Detection (VoxelNext - KEEP)

**Input**: Raw LiDAR point cloud (N × 4: x, y, z, intensity)
**Output**: 3D bounding boxes (K × 7: cx, cy, cz, dx, dy, dz, yaw)

**Key Parameters**:
- Score threshold: 0.2 (adjustable)
- Range filter: 15m (indoor)
- Class labels: 1=car, 2=pedestrian, 3=cyclist

**Status**: ✅ Already working, keep as-is

### Component 2: SMPL Estimation (LiDAR-HMR - NEW)

**Input**: Person crop (128 × 3 points from bounding box)
**Output**: SMPL parameters (β: 10-d, θ: 72-d, mesh: 6890 × 3)

**Architecture** (from LiDAR-HMR paper):
```
Point Cloud (128 × 3)
    ↓
Point Transformer-v2 (feature extractor)
    ↓
Pose Regression Network (PRN)
    ↓
Template 3D Pose (24 joints × 3)
    ↓
Mesh Reconstruction Network (MRN)
    ↓
Fine 3D Mesh (6890 × 3 vertices)
    ↓
MeshIK (Inverse Kinematics)
    ↓
SMPL Parameters:
  - β: Shape (10-d) ← IDENTITY FEATURES
  - θ: Pose (72-d) ← JOINT ANGLES
```

**Dependencies**:
- Point Transformer-v2 (install from GitHub)
- ChamferDistancePytorch (for training)
- SMPL model files (download from smpl.is.tue.mpg.de)

**Performance**:
- Works with sparse point clouds (30-60 points/person)
- Handles noise and incompleteness
- Tested on SLOPER4D, Waymo-v2, 3DPW

### Component 3: ReID (NEW)

**Input**: SMPL β vector (10-d)
**Output**: Person identity (matched to gallery)

**Method 1: Direct Comparison**
```python
# Compute cosine similarity between β vectors
def reid_beta(beta_query, gallery_betas):
    similarities = []
    for beta_gallery in gallery_betas:
        sim = cosine_similarity(beta_query, beta_gallery)
        similarities.append(sim)
    return np.argmax(similarities)
```

**Method 2: Learned Metric**
```python
# Train a small network to map β to ReID space
class BetaReID(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)  # ReID embedding

    def forward(self, beta):
        x = F.relu(self.fc1(beta))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.normalize(x, p=2, dim=1)
```

**Training Strategy**:
1. Use temporal tracking for pseudo-labels
2. Same track → same person (positive pair)
3. Different tracks → different people (negative pair)
4. Train with contrastive loss (InfoNCE)

### Component 4: Pose Estimation (BONUS)

**Input**: SMPL θ vector (72-d)
**Output**: 3D skeleton (24 joints × 3)

**Joint Mapping**:
```python
SMPL_JOINTS = {
    0: 'pelvis',
    1: 'left_hip',
    2: 'right_hip',
    3: 'spine1',
    4: 'left_knee',
    5: 'right_knee',
    6: 'spine2',
    7: 'left_ankle',
    8: 'right_ankle',
    9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'left_hand',
    23: 'right_hand'
}
```

---

## Implementation Plan

### Phase 1: Environment Setup (Day 1)

**Task 1.1: Clone and Setup LiDAR-HMR**
```bash
# Clone repository
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

**Task 1.2: Download Pre-trained Weights**
```bash
# Download LiDAR-HMR pre-trained weights
# Check GitHub releases: https://github.com/soullessrobot/LiDAR-HMR/releases
# Or train from scratch on SLOPER4D dataset
```

**Task 1.3: Test Basic Inference**
```bash
# Test on sample data
python scripts/lidar_hmr/test_lidarhmr.py \
  --dataset sloper4d \
  --cfg configs/mesh/sloper4d.yaml \
  --state_dict /path/to/weights.pth
```

### Phase 2: Integration with Existing Pipeline (Day 2-3)

**Task 2.1: Create SMPL Extraction Script**

Create `extract_smpl.py`:
```python
#!/usr/bin/env python3
"""
extract_smpl.py — Extract SMPL parameters from LiDAR detections.

Input: NPZ file with VoxelNext detections + point cloud frames
Output: SMPL β (10-d) + θ (72-d) for each detection
"""
import torch
import numpy as np
from lidar_hmr import LiDAR_HMR  # Import from LiDAR-HMR

class SMPLExtractor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
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
                crop = extract_crop(pts, box, n_pts=128)
                beta, theta, mesh = self.extract_from_crop(crop)

                all_beta.append(beta)
                all_theta.append(theta)
                all_mesh.append(mesh)
                all_fi.append(fi)
                all_box.append(box)
                all_score.append(score)

        return {
            'beta': np.array(all_beta),
            'theta': np.array(all_theta),
            'mesh': np.array(all_mesh),
            'fi': np.array(all_fi),
            'box': np.array(all_box),
            'score': np.array(all_score),
        }
```

**Task 2.2: Create SMPL ReID Module**

Create `reid_smpl.py`:
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

    def track_with_smpl(self, detections, iou_threshold=0.3):
        """
        Track persons using SMPL β + IoU.

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
            if best_score > 0.9 and best_track >= 0:
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

**Task 2.3: Integrate with Web Server**

Modify `reid_web_server.py` to add SMPL mode:
```python
# Add argument
parser.add_argument("--smpl-mode", action="store_true",
                    help="Use SMPL β for ReID instead of embeddings")

# In main loop
if args.smpl_mode:
    from extract_smpl import SMPLExtractor
    from reid_smpl import SMPLReID

    extractor = SMPLExtractor(model_path="path/to/lidar_hmr.pth")
    reid = SMPLReID()

    # Extract SMPL for all detections
    smpl_data = extractor.extract_from_session(npz_path, frames_dir)

    # Track using SMPL β
    tracks = reid.track_with_smpl(smpl_data)
```

### Phase 3: Training & Evaluation (Day 4-7)

**Task 3.1: Fine-tune LiDAR-HMR on Indoor Data**

Since LiDAR-HMR is trained on outdoor data (Waymo, SLOPER4D), we need to fine-tune on indoor data:

```bash
# Create indoor dataset
# Collect 20+ minutes of MID-360 data with multiple people
# Annotate person IDs (manual or semi-automatic)

# Fine-tune
python scripts/lidar_hmr/train_lidarhmr.py \
  --dataset indoor \
  --cfg configs/mesh/indoor_finetune.yaml \
  --pretrained /path/to/lidar_hmr.pth \
  --epochs 50
```

**Task 3.2: Evaluate SMPL ReID Accuracy**

Create `evaluate_smpl_reid.py`:
```python
#!/usr/bin/env python3
"""
evaluate_smpl_reid.py — Evaluate SMPL-based ReID accuracy.
"""
import numpy as np
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
```

**Task 3.3: Compare with Baseline**

Compare SMPL ReID vs. 128-d embedding ReID:
- Same detections
- Same ground truth
- Different feature representations

Expected result: SMPL β significantly outperforms 128-d embeddings.

---

## File Structure

```
~/Projects/thesis/g1_perception_ws/
├── LiDAR-HMR/                    # Cloned repository
│   ├── scripts/                   # Training/testing scripts
│   ├── configs/                   # Configuration files
│   └── smplx_models/              # SMPL model files
├── extract_smpl.py                # SMPL extraction script (NEW)
├── reid_smpl.py                   # SMPL-based ReID (NEW)
├── evaluate_smpl_reid.py          # Evaluation script (NEW)
├── reid_web_server.py             # Modified for SMPL mode
├── reid_data/                     # Existing data
│   ├── emb_*.npy                  # Old 128-d embeddings (deprecated)
│   └── smpl_*.npy                 # New SMPL β, θ, mesh (NEW)
└── docs/
    └── PLAN_SMPL_LIDAR_REID.md    # This document
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

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LiDAR-HMR fails on MID-360 data | Medium | High | Fine-tune on indoor data |
| SMPL β not discriminative enough | Low | High | Add contrastive learning |
| Too slow for real-time | Medium | Medium | Optimize with TensorRT |
| SMPL model download issues | Low | Low | Use pre-trained weights |

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

## Handoff Instructions for Coding Agents

### Step 1: Clone and Setup
```bash
cd ~/Projects/thesis/g1_perception_ws
git clone https://github.com/soullessrobot/LiDAR-HMR.git
cd LiDAR-HMR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Download SMPL Model
```bash
# Visit: https://smpl.is.tue.mpg.de/
# Download SMPL model files
# Place in: LiDAR-HMR/smplx_models/
```

### Step 3: Create Scripts
- Create `extract_smpl.py` (see Task 2.1)
- Create `reid_smpl.py` (see Task 2.2)
- Create `evaluate_smpl_reid.py` (see Task 3.2)

### Step 4: Test
```bash
# Test SMPL extraction
python extract_smpl.py --session 2026-07-29_17-21-48

# Test ReID
python evaluate_smpl_reid.py --session 2026-07-29_17-21-48

# Compare with baseline
python compare_reid_embeddings.py --smpl-mode
```

### Step 5: Integrate
- Modify `reid_web_server.py` for SMPL mode
- Add SMPL visualization to web UI
- Update documentation

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
