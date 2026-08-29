# Workspace rules — g1_perception_ws

## Identity / paths (READ FIRST)
- **The user account is `thakk100`** (lowercase: t-h-a-k-k-1-0-0). Home: `/home/thakk100`.
- Home is NOT `thabb100` (that is a common typo of "thakk100"). Never write that.
- **Prefer tokens over hardcoded usernames in all shell commands and scripts:**
  use `$HOME`, `$USER`, or `~/...` in place of `/home/thakk100`.
- When a hardcoded path is unavoidable, double-check against `whoami` before running.
- Canonical paths:
  - Workspace: `~/Projects/thesis/g1_perception_ws`
  - August LiDAR sessions (LVX2): `~/Downloads/`
  - VoxelKP: `./VoxelKP/`, VoxelNeXt: `./VoxelNext/`

## Environment
- Python venv: `./.venv` (python 3.12). Activate before running repo scripts.
- ROS2: noetic (system), built via colcon into `./build` + `./install`.
- `LD_LIBRARY_PATH` must include the venv torch lib for VoxelKP CUDA ops (see HANDOFF_REID.md).

## Build safety
- **Never run `python3 setup.py develop` in a subagent for VoxelKP** — it has crashed
  subagent processes repeatedly. Run it in the main session only.
