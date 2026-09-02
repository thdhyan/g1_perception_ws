# Hardware Constraints & Deployment Targets

## Current Development: RTX 4060 Laptop

| Spec | Value |
|---|---|
| GPU | RTX 4060 Laptop, 8 GB VRAM |
| RAM | 16 GB |
| CUDA | 12.x |
| PyTorch | 2.x |
| Use | Development, testing, annotation |

### Limitations
- 8 GB VRAM limits model size
- Cannot run multiple large models simultaneously
- Training possible but slow (batch size 1-2 for 3D models)

## Target Deployment: Jetson Orin 16GB

| Spec | Value |
|---|---|
| GPU | NVIDIA Ampere, 1024 CUDA cores |
| VRAM | 16 GB unified (CPU+GPU) |
| AI Performance | 100 TOPS (INT8) |
| JetPack | 4.6 (L4T 35.x) |
| Power | 15-60W configurable |

### JetPack 4.6 Constraints
- **CUDA**: 11.4 (not 12.x)
- **PyTorch**: 1.12-1.14 (not 2.x)
- **TensorRT**: 8.4
- **cuDNN**: 8.6
- **Python**: 3.8
- **Ubuntu**: 20.04

### Key Considerations
1. **Unified memory**: 16 GB shared between CPU and GPU — must be careful with memory allocation
2. **TensorRT optimization**: Essential for real-time inference — FP16/INT8 quantization required
3. **Power budget**: Mobile robot → prefer lower power (15-30W)
4. **Thermal**: Passive cooling may throttle performance

## LiDAR: Livox MID-360

| Spec | Value |
|---|---|
| FoV | 360° (non-repetitive scanning) |
| Range | 0.1-40m |
| Points/sec | ~200,000 |
| Channels | 32 (effective) |
| Update rate | 10 Hz |

### Comparison with Livox Mid-100
- MID-360 is the successor to Mid-100
- Same non-repetitive scanning pattern
- Wider FoV (360° vs 200°)
- Higher point density
- **ReID3D was trained on Mid-100 — compatible**

### Indoor Point Density (estimated)
| Distance | Points/Human | Points/Frame |
|---|---|---|
| 2m | ~1000 | ~200k |
| 5m | ~500 | ~200k |
| 10m | ~200 | ~200k |
| 15m | ~100 | ~200k |

## ROS2 Node Budget

For a mobile robot, we need to fit all nodes within the compute budget:

| Node | Est. VRAM | Est. FPS | Priority |
|---|---|---|---|
| LiDAR driver | ~0.5 GB | 10 Hz | Critical |
| 3D Detection | ~2 GB | 10-15 Hz | Critical |
| ReID Embedding | ~1 GB | 10 Hz | High |
| Tracking | ~0.5 GB | 10 Hz | High |
| Pose Estimation | ~2 GB | 10 Hz | Medium |
| Navigation | ~1 GB | 10 Hz | Critical |
| **Total** | **~7 GB** | — | — |

**Buffer**: ~9 GB留给其他节点和系统开销

## Model Optimization Pipeline

```
Training (RTX 4060 / cloud)
    ↓
Export to ONNX
    ↓
TensorRT optimization (FP16/INT8)
    ↓
Deploy on Jetson Orin
```

### Quantization Strategy
- **Detection model**: INT8 (TensorRT PTQ or QAT)
- **ReID model**: FP16 (embedding quality degrades with INT8)
- **Pose model**: FP16 (keypoint regression needs precision)
