# Research & Planning Documents

## Research Documents

| File | Topic | Status |
|---|---|---|
| `research_reid_comparison.md` | ReID model comparison (ReID3D, point-cloud-reid, etc.) | ✅ Complete |
| `research_detection_models.md` | 3D detection model comparison | ✅ Complete |
| `research_hardware_constraints.md` | Jetson Orin + MID-360 deployment | ✅ Complete |
| `research_indoor_detectors.md` | Indoor-specific LiDAR detectors | 🔄 In progress |
| `research_pose_estimators.md` | LiDAR pose estimation models | 🔄 In progress |

## Plans

| File | Objective | Timeline |
|---|---|---|
| `../plans/PLAN_REID_UPGRADE.md` | Upgrade to point-cloud-reid | 3 weeks |
| `../plans/PLAN_DETECTION_UPGRADE.md` | Indoor detection evaluation | 4 weeks |
| `../plans/PLAN_POSE_ESTIMATION.md` | LiDAR pose estimation | Deferred |

## Handoffs

| File | Status |
|---|---|
| `../handoffs/HANDOFF_REID.md` | Current ReID pipeline (working) |
| `../handoffs/HANDOFF_REID_INTEGRATION.md` | point-cloud-reid integration (in progress) |

## Hardware Context

- **Current**: RTX 4060 Laptop (8 GB VRAM), 16 GB RAM
- **Target**: Jetson Orin 16 GB (JetPack 4.6)
- **LiDAR**: Livox MID-360 (360° FoV, ~200k points/sec)
- **Environment**: 99% indoor, range ≤15m
