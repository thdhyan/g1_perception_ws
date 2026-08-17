# HANDOFF: G1 Sim Data Collection — SLAM, TF, Detection

**Date:** 2026-08-16/17. **Repo:** `g1_perception_ws` on `main`, all work committed
and buildable as of this handoff (see "Current state" below for the one
exception, already cleaned up).

## Goal

Reliable G1-humanoid data collection in Gazebo Harmonic sim: continuous,
unbroken TF (`map → odom → pelvis → ... → mid360_link/livox_frame`, plus a
separate `warehouse → g1 → gt_base_link → gt_pelvis` ground-truth tree),
working LiDAR-based human detection, and — the active next-phase goal —
GPU-accelerated SLAM (NVIDIA Isaac ROS Visual SLAM / cuVSLAM + nvblox)
running **alongside** the existing CPU-only LIO for direct comparison.

## Current Progress (this session, all committed)

Commit sequence on `main`: `f1d4a83` (pre-session baseline) →
`18e5a33` → `cb69507` → `e58619c` → `32b483d` → `2dea621` (HEAD). Full
narrative already written up in [`sim_workflow/README.md`](sim_workflow/README.md)
§3/§6 — read that for the blow-by-blow; this doc is the delta + what's next.

1. **Submodules were broken for fresh clones.** `g1_description`,
   `livox_laser_simulation_RO2`, `Ultra-Fusion`, `plain_slam_ros2` are
   gitlinks (mode `160000`) but `.gitmodules` was missing — a fresh clone
   got 4 empty directories. Fixed: `.gitmodules` created and committed.
   `g1_description` and `plain_slam_ros2` (the two with local fixes) now
   point at **your own forks** (`thdhyan/g1_description`,
   `thdhyan/plain_slam_ros2` — you re-pointed these yourself mid-session,
   both the file and each submodule's own `origin` remote) so the fixed
   commits (below) are actually reachable on a fresh clone. The other two
   (`livox_laser_simulation_RO2`, `Ultra-Fusion`) still point at upstream —
   unmodified, no fork needed.
2. **TF tree: multi-parent conflict + no startup fallback.** `pelvis` had
   two simultaneous parent claims (`base_link` static + `odom` dynamic from
   `lio_3d_node`). Flipped to `pelvis → base_link`; added identity fallback
   statics for `map → odom` / `odom → pelvis` so the tree is connected from
   `t=0` instead of only after `lio_3d_node`'s first scan. Also deleted a
   dead `g1_29dof → gt_base_link` publisher that duplicated the real
   `g1 → gt_base_link` ground-truth parent.
3. **`lio_3d_node` SIGABRT — root-caused via `gdb` against live sim
   topics** (not guesswork): `JointOptimizer::Estimate()` produces a
   non-finite Gauss-Newton step when scan-to-map correspondences are too
   thin (near-singular `H`/`P` inversion); the NaN/Inf reaches
   `Sophus::SO3::exp()`, which hard-`abort()`s — a raw signal, invisible to
   the try/catch already wrapping `SetScanCloud()`. Fixed with a finite
   check before `exp()`. Verified clean (no aborts) on a 60s live soak
   against the running sim.
4. **`lio_3d_node`'s real accuracy bug (separate from the crash): wrong
   IMU-to-LiDAR extrinsic.** `lio_3d_params.yaml` was calibrated for an IMU
   at `torso_link`'s origin, but the launch files feed it the **pelvis**
   IMU — a different link, ~14cm/5cm off in Z/X. Recomputed from the URDF
   kinematic chain (all joints between pelvis/torso/lidar are `type="fixed"`
   in this URDF, so a constant extrinsic is valid).
5. **Added onboard IMU sensors** for Mid-360 (`/livox/imu`) and D435i
   (`/camera/imu`) to the URDF — real-robot topic parity (documented in
   [`PLAN.md`](PLAN.md) §4, wasn't previously simulated). Neither is
   consumed by any node yet (no VIO in the stack until cuVSLAM lands) — the
   D435i one is exactly what cuVSLAM will eventually want.
6. **VoxelNeXt detector fixed and verified.** Was silently falling back to
   a PointPillar clustering heuristic — launch files passed it the
   CenterPoint/PointPillar checkpoint regardless of algorithm, so
   `load_params_from_file()` threw and got swallowed. Fixed
   (`VOXELNEXT_CHECKPOINT_PATH`). Verified against a live headless sim run:
   real per-frame varying scores/counts. Added `class_filter` (default
   `"pedestrian"`) and aligned `accumulate_frames` to the `max_hz` cadence
   (2 frames = 200ms @ Mid-360's 10Hz, matching the 5Hz inference cap 1:1).
   **Achieved rate is ~4.0–4.2Hz against the 5Hz cap** — that's real
   inference latency, not further tunable via these params; see
   [`livox_detection/README.md`](src/livox_detection/README.md).
7. **Added a moving pedestrian actor** (`human_walking`) to
   `g1_warehouse.sdf` via Gazebo's `<actor>/<trajectory>` system — the
   existing 5 humans are static primitives, no motion to test detection
   tracking against. Straight 8m path, ~1.2 m/s, loops. First launch needs
   network access to fetch the actor mesh from Fuel (cached after).

## What Worked

- **`gdb --batch -ex run -ex "bt full" -ex "thread apply all bt"` against a
  live process fed real sim topics** — this is what actually found the LIO
  crash's true cause. Reading the code and guessing (IMU `dt` validation)
  was *plausible* but wrong; only the backtrace showed the real path
  (`JointOptimizer::Estimate → Sophus::SO3::expAndTheta → abort()`). Don't
  skip the backtrace step on the next crash either.
- **Checking `ps -ef` for what's actually alive before debugging further.**
  Several dead-ends this session were caused by stale/degraded live
  sessions (a died `lio_3d_node` that `ros2 launch` never respawns, or my
  own parallel diagnostic processes stealing GPU/CPU from the real run) —
  always verify what's actually running before trusting terminal output
  that might be from a zombied session.
- **`--frame-id`/`--child-frame-id` static fallback matching the dynamic
  broadcaster's exact parent/child pair** — lets a dynamic SLAM node's
  output cleanly override a startup-time identity static with zero TF
  conflict, as long as no *other* source claims a different parent for the
  same child frame.

## What Didn't Work

- **Guessing the LIO crash was in the IMU 200Hz callback path** (unguarded
  `dt`) — reasonable hypothesis, added a real defensive fix (still in,
  doesn't hurt), but it **was not the actual crash**. The real one was in
  the LiDAR scan-matching optimizer, already inside an existing try/catch
  that couldn't help because `abort()` isn't a C++ exception. Lesson above.
- **A `d.allFinite()`-style edit made directly in the IDE got corrupted**
  mid-session (literal string `"Interrupted"` landed inside a C++ comment,
  wouldn't compile) — caught before commit by diffing against the last
  known-good commit and rebuilding. **If you see build failures on
  `lio_3d_node.cpp`, check for stray non-code text from an interrupted
  edit before assuming it's a logic bug.**

## Next Steps

**Primary: implement the approved plan at
[`/home/thakk100/.claude/plans/recursive-beaming-finch.md`](/home/thakk100/.claude/plans/recursive-beaming-finch.md)**
— Isaac ROS Visual SLAM (cuVSLAM) + nvblox in a Docker container (own CUDA
13 runtime, `--network host` to cross into the native ROS2 Jazzy graph),
RGBD mode (no stereo IR camera exists in sim — real gap, documented, RGBD
sidesteps it), added as a **third `slam_type:="isaac"` branch** alongside
the existing `slam_type:="3d"` (plain_slam_ros2) / `"2d"` (slam_toolbox)
pattern in `sim.launch.py`/`sim_teleop.launch.py`, publishing to isolated
frame names (`vslam_odom`/`vslam_map`) so it can run **side-by-side** with
plain_slam_ros2 for comparison. Read the plan file for full phasing
(container setup → standalone smoke test → package → RViz comparison) and
the one flagged uncertainty (exact `isaac_ros_common` devcontainer command
sequence — doc excerpts confirmed the repo/apt steps, not the full script).

**Explicit scope note — don't split effort across three SLAM stacks:**
cuVSLAM + nvblox is the active direction now. Two things NOT to pursue in
parallel:
- **`plain_slam_ros2` (the LIO node)** — it's fixed and working (items 2–4
  above), keep it running as the passive comparison baseline the plan calls
  for, but **don't spend further effort tuning/optimizing it**. It's
  CPU-only by architecture (no CUDA anywhere in `joint_optimizer.cpp`/
  `normal_map.cpp`) — that ceiling is why cuVSLAM is happening at all.
- **`src/Ultra-Fusion`** — a third, *separate* multi-sensor LVIO SLAM
  framework, vendored as a submodule, but **never wired into any launch
  file this entire session** (confirmed: zero references outside its own
  directory). It's ROS2 **Humble**-targeted, not Jazzy — would need porting
  work just to run here. Treat it as unused vendor code, not a live
  candidate, unless explicitly revisited later.

**Known open item, not yet investigated:** the RTX 4060's 8GB VRAM is right
at nvblox's stated minimum, shared with Gazebo rendering + WBC ONNX +
VoxelNeXt CUDA all running natively at the same time as the containerized
Isaac ROS stack. The plan's Phase 1 smoke test runs Isaac ROS with
detection **off** first specifically to measure this before combining
everything — don't skip that step.

**Environment facts as of this handoff** (re-verify if it's been a while):
GPU: RTX 4060 Laptop, 8188 MiB, driver 580.173.02. `nvcc` toolkit: 12.0
(driver supports up to CUDA 13 — toolkit and driver-max are different
numbers, don't conflate them). Docker 29.7.2 present, `nvidia-container-toolkit`
**not yet verified installed** — first real step of Phase 0.
