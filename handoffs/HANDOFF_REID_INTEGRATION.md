# Handoff: ReID Integration with point-cloud-reid

## Current Status

**Date**: 2026-09-01
**Agent**: Background agent (ses_f9fa5611affexVYu35M37QqcRl)

## What's Being Done

Integrating point-cloud-reid (WACV 2024) into the existing ReID pipeline.

### Steps in Progress
1. Cloning point-cloud-reid repository
2. Installing dependencies
3. Downloading pre-trained models
4. Creating new embedding extraction script

### Expected Outcome
- New model: Point-Transformer + RTMM matching head
- Pre-trained on nuScenes/Waymo ReID datasets
- 128-d embeddings (compatible with existing pipeline)
- 10 Hz inference speed

## How to Verify

```bash
# Check if repo was cloned
ls ~/Projects/thesis/g1_perception_ws/point-cloud-reid/

# Check if pre-trained models downloaded
ls ~/Projects/thesis/g1_perception_ws/point-cloud-reid/pretrained/

# Test embedding extraction
cd ~/Projects/thesis/g1_perception_ws
source .venv/bin/activate
python reid_embed_pointcloudreid.py --session 2026-07-29_17-21-48 --test
```

## What's Left

1. Verify embedding format compatibility
2. Test with pair annotator (port 8765)
3. Compare with current PointNet embeddings
4. Fine-tune on our data (if needed)

## Files Created

- `reid_embed_pointcloudreid.py` — new embedding extraction script
- `docs/research_reid_comparison.md` — model comparison

## Rollback

If new model doesn't work, the existing PointNet model (`reid_model.py`) remains functional. No changes to existing pipeline.
