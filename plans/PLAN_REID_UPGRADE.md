# Plan: ReID Model Upgrade (point-cloud-reid)

## Objective

Upgrade from custom PointNet to point-cloud-reid (WACV 2024) for better person re-identification accuracy.

## Phase 1: Setup (NOW)

- [x] Clone point-cloud-reid repository
- [ ] Install dependencies (torchpack, etc.)
- [ ] Download pre-trained models (Point-Transformer on nuScenes)
- [ ] Test inference on sample data

## Phase 2: Integration (Week 1)

- [ ] Create `reid_embed_pointcloudreid.py` — new embedding extraction script
- [ ] Handle input format differences (128 vs 256 points)
- [ ] Generate embeddings for session 2026-07-29_17-21-48
- [ ] Verify embeddings work with pair annotator (port 8765)
- [ ] Compare with current PointNet embeddings (visualize t-SNE, cosine similarity distributions)

## Phase 3: Fine-tuning (Week 2, if needed)

- [ ] Extract training data from our annotated pairs (59 same-person pairs)
- [ ] Fine-tune Point-Transformer on our data
- [ ] Benchmark accuracy improvement

## Phase 4: Deployment (Week 3)

- [ ] Optimize model for Jetson Orin (TensorRT FP16)
- [ ] Update reid_embed_server.py to support both models
- [ ] Update reid_web_server.py to load new embeddings

## Success Criteria

1. Embedding extraction works on RTX 4060 (8 GB VRAM)
2. Pair annotator shows improved ranking (higher cosine similarity for same-person pairs)
3. Model runs at ≥10 FPS on laptop
4. Model runs at ≥5 FPS on Jetson Orin (after TensorRT optimization)

## Files to Modify

- `reid_embed_server.py` — add new model loader
- `reid_web_server.py` — no changes needed (works with any embedding format)
- `reid_model.py` — keep as fallback

## Risks

1. **Indoor vs outdoor**: Pre-trained on driving data, may not transfer to indoor
2. **Point density**: MID-360 indoor may have different point distributions
3. **JetPack 4.6**: PyTorch version constraints on Jetson
