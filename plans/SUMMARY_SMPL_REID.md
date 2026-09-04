# Quick Reference: SMPL-Driven LiDAR ReID

## The Problem (One Sentence)
The 128-d embeddings are 99.6% similar for ALL people because they were trained to recognize "human shape" not individual identity.

## The Solution (One Sentence)
Use SMPL shape parameters (β: 10-d) which encode body proportions (height, weight, limb lengths) that are physically discriminative between people.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  CURRENT (BROKEN)                                          │
│  LiDAR → VoxelNext → Crop → PointNet → 128-d → ReID       │
│                                    ↓                        │
│                              ALL IDENTICAL (99.6%)          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PROPOSED (SMPL-BASED)                                     │
│  LiDAR → VoxelNext → Crop → LiDAR-HMR → SMPL β (10-d)     │
│                                    ↓                        │
│                              DISCRIMINATIVE                 │
│                              (body proportions)             │
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

## What SMPL Gives You

| Output | Dimension | Use Case |
|--------|-----------|----------|
| **β (shape)** | 10-d | Person ReID (identity) |
| **θ (pose)** | 72-d | Skeleton tracking |
| **Mesh** | 6890 × 3 | Full 3D body |

**Two birds, one stone**: ReID + Pose Estimation

---

## Implementation Steps

### Day 1: Setup
```bash
cd ~/Projects/thesis/g1_perception_ws
git clone https://github.com/soullessrobot/LiDAR-HMR.git
cd LiDAR-HMR
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Download SMPL model from https://smpl.is.tue.mpg.de/
```

### Day 2-3: Create Scripts
1. `extract_smpl.py` — Extract β, θ, mesh from detections
2. `reid_smpl.py` — ReID using β comparison
3. `evaluate_smpl_reid.py` — Test accuracy

### Day 4-7: Integrate & Evaluate
1. Modify web server for SMPL mode
2. Fine-tune on indoor data
3. Compare with baseline (128-d embeddings)

---

## Expected Results

| Metric | Current | Target |
|--------|---------|--------|
| Rank-1 Accuracy | ~0% | >80% |
| Unique People Detected | 1 | N (ground truth) |
| Pose Estimation | ❌ | ✅ 24 joints |

---

## Key Files

| File | Purpose |
|------|---------|
| `PLAN_SMPL_LIDAR_REID.md` | Full implementation plan |
| `extract_smpl.py` | SMPL extraction (TO CREATE) |
| `reid_smpl.py` | SMPL ReID (TO CREATE) |
| `LiDAR-HMR/` | Cloned repository (TO CLONE) |

---

## Dependencies

- LiDAR-HMR: https://github.com/soullessrobot/LiDAR-HMR
- SMPL Model: https://smpl.is.tue.mpg.de/
- Point Transformer-v2: https://github.com/Pointcept/PointTransformerV2

---

## Success Criteria

- [ ] LiDAR-HMR installed and working
- [ ] SMPL β extraction working on sample data
- [ ] SMPL ReID accuracy > 80% (vs. current ~0%)
- [ ] Pose estimation working (24 joints)
- [ ] Integrated with web server
- [ ] Documentation complete

---

**Status**: Ready for implementation
**Next Step**: Clone LiDAR-HMR and setup environment
