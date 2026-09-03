# VoxelNeXt patches

Handwritten patches applied on top of `JIA-Lab-research/VoxelNeXt` (upstream commit `b5b7d39`).
Apply after cloning VoxelNeXt — see workspace `SETUP.md §4b`.

---

## centernet_utils_topk_sparse_fix.patch

**File**: `pcdet/models/model_utils/centernet_utils.py`  
**Function**: `_topk_1d`

### Problem
Two bugs triggered by sparse indoor LiDAR (Mid-360, few active voxels):

1. **RuntimeError on `scores.view(batch_size, K)`** — sparse scenes may have
   fewer than `K` active voxels per class. The original code assumed ≥ K
   candidates, causing a shape mismatch.

2. **Wrong class IDs** — `topk_classes = (topk_ind // K)` assumed the flat
   `topk_ind` layout was `(C × K)`, but when a class has fewer than K voxels
   the block size differs from K, so integer division gives wrong class labels.

### Fix
- Pad `topk_score` / `topk_ind` to exactly K with `-inf` scores and a
  harmless repeated index; padded entries are dropped by `score_thresh`.
- Compute class IDs from the real block size `_block = topk_scores.shape[-1]`
  before padding, not from `K`.
- Guard: only use the new class-id path when `obj.shape[-1] != 1 or nuscenes`
  (the original single-class path is unchanged).

---

## datasets_argo2_optional.patch

**File**: `pcdet/datasets/__init__.py`

### Problem
`from .argo2.argo2_dataset import Argo2Dataset` raises `ImportError` on systems
without the Argoverse 2 SDK installed (our robot + laptop setups).

### Fix
Wrap in `try/except ImportError`; set `Argo2Dataset = None` on failure and
exclude it from `__all__` conditionally. No behaviour change when the SDK is present.

---

## How to apply

```bash
cd VoxelNeXt
git apply ../patches/voxelnext/centernet_utils_topk_sparse_fix.patch
git apply ../patches/voxelnext/datasets_argo2_optional.patch
```

Verify:
```bash
git diff --stat
# expect: 2 files changed, 35 insertions(+), 3 deletions(-)
```
