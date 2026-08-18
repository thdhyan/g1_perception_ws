# HANDOFF: G1 Sim Data Collection — SLAM, TF, Detection

**Date:** 2026-08-16/17, updated 2026-08-17 (second session) and 2026-08-18
(third session). **Repo:** `g1_perception_ws` on `main`, all three sessions'
work now committed. The Isaac ROS stack **builds and runs end-to-end**, but
cuVSLAM produces no odometry — root-caused in session 3 to a depth-encoding
mismatch, with the fix identified and not yet written. Read "Session 3"
first; it supersedes the open questions left by session 2.

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

## Session 2 (2026-08-17): Isaac ROS Docker — disk crisis, Phase 0 findings

**Nothing was launched end-to-end this session** — root disk hit 95–100%
full **three times**, twice crashing VSCode/tmux, and most of the session
was disk firefighting, not SLAM work. Read this section fully before
continuing; the next attempt should not repeat the same disk mistakes.

### Disk crisis — what happened and what's safe going forward

A `docker run ... apt-get install ros-jazzy-isaac-ros-visual-slam
ros-jazzy-isaac-ros-nvblox ros-jazzy-isaac-ros-unet` (into a bare container,
not a real Dockerfile build) produced a **27.7GB container writable layer**
— `apt-get clean` found nothing to reclaim (Ubuntu's Docker base already
auto-purges `.deb`s per-install; the 27.7GB is real unpacked CUDA13
toolkit + TensorRT + Triton + GUI-library apt *recommends*, not cache). Root
disk (`/dev/nvme0n1p5`, 355G total) went from ~29G free to **0 free** over
the session. Full technical detail (exact install command, why
`docker commit` doesn't help here, the Dockerfile-based fix) is in the plan
file, not repeated here — see "Next Steps" below.

**Host changes made to recover disk** (none are in git, flagging so a fresh
agent doesn't get confused by things that changed outside the repo):

- **`/home/thakk100/Projects/gemini-robotics-sdk` deleted entirely** (was
  ≈3.0G, almost entirely its own `.venv`; had untracked work not recoverable
  from git — user explicitly approved full deletion).
- **`/home/thakk100/Projects/so101_mnri` moved to
  `/Storage/Projects/so101_mnri`** (`/Storage` = `/dev/sda1`, separate
  physical drive, 300G free). Original path is now a symlink to there —
  verified transparent (44,935/44,935 files matched post-copy, `.venv`
  resolves fine through the symlink). If you ever need to move another big
  `.venv` for headroom, this rsync→verify→delete-original→symlink-back
  pattern is the safe template; `IsaacLab/.venv` (17G) is the next-biggest
  candidate on this host if more room is needed later.
- Cleared: `uv` cache (72GB, `uv cache clean`), `.drift` cache (3.9G), a
  stale TensorRT apt-repo leftover (4.2G, `/var/nv-tensorrt-local-repo-*`),
  unused docker images (`ultrafusion-ros2` + a stale smoke-test cuda image,
  ~4.6G), ~1GB via `sudo journalctl --vacuum-time=2d`.
- **The disposable ad-hoc install container (`isaac_ros_install`) was
  removed** (`docker rm`) once the decision was made to rebuild via a real
  Dockerfile instead of reusing it — see plan. Its install log is preserved
  at `~/isaac_ros_install.log` (host, outside repo) — the exact working
  apt package list/versions live there, check it before re-running the
  install from scratch.
- **Disk as of this handoff: ~36G free / 90% used.** Still worth checking
  `df -h /` before any docker build — this host runs chronically tight
  (other large, unrelated projects under `~/Projects/` eat most of it) and
  has now hit 100% three separate times across two sessions.
- A stray full duplicate `isaac_ros_common` clone at
  `~/workspaces/isaac_ros-dev` (outside the repo, 2MB, harmless) exists from
  an early exploration step — the one that matters is the in-repo submodule
  at `isaac_ros_ws/src/isaac_ros_common` (see below). Safe to `rm -rf` the
  stray one, not urgent.

### What's actually proven vs. what's still scaffolding

**Proven working** (from the apt install that ran to completion, log at
`~/isaac_ros_install.log`): `ros-jazzy-isaac-ros-visual-slam`,
`ros-jazzy-isaac-ros-nvblox`, `ros-jazzy-isaac-ros-unet` install cleanly
against base image `nvcr.io/nvidia/isaac/ros:isaac_ros_89df02a734965ed64c227ef531c09d65-amd64`
(already cached locally as `cached_isaac_run_dev_image_local:latest`, don't
re-pull), full `ros2 pkg list` afterward showed `isaac_ros_visual_slam`,
`nvblox_ros`, `nvblox_nav2`, `isaac_ros_unet` + kernels,
`isaac_ros_tensor_rt`, `isaac_ros_triton`, etc. present.
`ros-jazzy-isaac-ros-unet` pulls in `ros-jazzy-isaac-ros-peoplenet-models-install`
— **the human-segmentation model is already covered by this one apt
command**, no extra step needed for that part of the original ask.
`nvidia-container-toolkit` + GPU passthrough into containers verified
working (`docker run --gpus all ... nvidia-smi` shows the RTX 4060).

**Not yet done, in dependency order** (all of this list was completed in
session 3 — kept for the dependency ordering only, see "Session 3" for
current state): `isaac_ros_ws/Dockerfile.g1_isaac_slam`
(doesn't exist — bakes the above install + this package into a real,
restartable image); `src/g1_isaac_slam/launch/_container_isaac_slam.launch.py`
(doesn't exist — the actual node launch that runs inside the container);
`src/g1_isaac_slam/config/isaac_slam.rviz` (doesn't exist); the
`slam_type:="isaac"` branch in `sim.launch.py`/`sim_teleop.launch.py` (not
started). `src/g1_isaac_slam/launch/isaac_slam.launch.py` itself (the
top-level, host-side orchestration launch file) **is already written and
looks complete** — it includes the sim, runs the docker container via
`ExecuteProcess`, and opens a dedicated RViz window; it just has nothing
inside the container to actually call yet.

### Sim pipeline & how the Docker container connects to it

The G1 sim publishes everything Isaac ROS needs already, no URDF/Gazebo
changes required:

| Signal | Topic | Source |
|---|---|---|
| RGB camera | `/camera/color/image_raw` | D435i sim, via `gz_bridge.yaml` |
| Depth camera | `/camera/depth/image_rect_raw` | same |
| Camera intrinsics | `/camera/color/camera_info` | same, `gz_bridge.yaml:286-289` |
| LiDAR | `/livox/mid360/points` | Mid-360 sim (not used by the RGBD-only Phase 1 smoke test; relevant later for lidar-fusion mapping) |
| Pelvis IMU | `/imu/data` | used by `plain_slam_ros2`, not cuVSLAM |

Bring the sim up standalone first (matches Phase 1 of the plan — clean VRAM
baseline, detection off):
```bash
ros2 launch g1_bringup sim_teleop.launch.py detection:=false slam:=false rviz:=false
```

**Docker↔host connection** is `--network host` (same machine, same DDS
domain — same unicast pattern already used for robot↔laptop in
`DATA_COLLECTION.md`) plus two env vars that **must match the host exactly**
or DDS discovery silently fails with no error:
- `-e ROS_DOMAIN_ID=0` (host has none set → defaults to 0)
- `-e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` (host sets this in
  `setup_g1_env.sh:17` — if the container defaults to FastRTPS instead,
  zero topics will be visible cross-container, and it fails silently, not
  with an error)

`--gpus all` for the RTX 4060 passthrough (verified working, see above).
`g1_isaac_slam/launch/isaac_slam.launch.py` already wires all of this up in
its `ExecuteProcess` — it's the right template, just needs the image built
and the in-container launch file written before it'll do anything.

## Session 3 (2026-08-18): stack runs end-to-end; cuVSLAM depth input root-caused

Everything session 2 listed as "not yet done" now exists and is committed:
`isaac_ros_ws/Dockerfile.g1_isaac_slam` (+ a `.dockerignore` that reduces the
build context from the 14G workspace root to just `src/g1_isaac_slam`),
`src/g1_isaac_slam/launch/_container_isaac_slam.launch.py`,
`src/g1_isaac_slam/config/isaac_slam.rviz`. The image builds, the container
starts, cross-container DDS discovery works, and both `visual_slam_node` and
`nvblox_node` come up and receive frames.

**The one blocking bug: cuVSLAM ignores the depth image's ROS encoding.**
`/camera/depth/image_rect_raw` is Gazebo's `rgbd_camera` output — `32FC1`,
metres. cuVSLAM's RGBD path reinterprets the incoming buffer as `uint16`
unconditionally, so it reads float32 mantissa bytes as integers.

Proven, not inferred, via `enable_image_denoising`'s sibling
`enable_debug_mode: true` + `debug_dump_path: /tmp/cuvslam_debug` (both params
are already set in the launch file, left on deliberately — turn them off once
this is fixed, they write ~900KB/frame):

- every `depths/cam0.NNNNN.npy` is **614480 bytes = 640·480·2 + npy header**,
  i.e. 2 bytes/pixel where we send 4;
- `np.load(...)` reports `dtype uint16, shape (480,640)`, `min 0`,
  `max 65535` (the uint16 ceiling), `mean ≈24540`, with adjacent pixels like
  `[16633 16342 29888 16342 15846 16343 49642 ...]` — noise, not a depth field.

Equally important, the same dump **clears everything else**: `stereo.edex`
shows correct intrinsics (`focal 337.22/337.22`, `principal 320/240`,
`size 640×480`, `distortion_model pinhole`) and a valid camera→base rotation +
translation, and `frame_metadata.jsonl` shows steadily advancing frames at
~30Hz timestamps. So the TF/extrinsic/optical-frame work from earlier sessions
is right — **depth encoding is the only broken input.**

**`depth_scale_factor` cannot fix this** — it multiplies *after* the wrong
integer read. The fix is a conversion node upstream of cuVSLAM: subscribe to
`32FC1` metres, publish `16UC1` millimetres on a new topic, remap
`visual_slam/depth_0` (and nvblox's `depth/image`) to it, then set
`depth_scale_factor: 0.001`. **Not written yet — this is the next task.**

Also fixed this session and worth not re-discovering (recorded inline in
[`_container_isaac_slam.launch.py`](src/g1_isaac_slam/launch/_container_isaac_slam.launch.py)'s
docstring, read it before changing any param there):

- `rectified_images: True` breaks single-camera RGBD outright — it forces
  cuVSLAM's stereo path regardless of `num_cameras: 1` ("Rectified stereo
  camera mode only works with 1+ stereo cameras. Number of cameras must be
  even."). Left `False` despite the sim's zero-distortion `camera_info`.
- `camera_optical_frames` must be the REP-103 *optical*-convention frame
  (`camera_color_optical_frame`), **not** the URDF body-convention link
  (`d435_link`) that the image headers carry — that mismatch caused a
  per-frame `[CUDA] error invalid argument(1)`.
- `nvblox_ros` has **no `base_frame` param** (it's silently ignored); the real
  one is `map_clearing_frame_id`, and its default `base_link` doesn't exist in
  this sim's TF tree — set to `pelvis`.

Teardown after each run matters: `docker rm -f g1_isaac_slam_container` plus
killing the native `gz sim`/`wbc_node`/`component_container`/`parameter_bridge`
process group, or the next run inherits a half-dead stack (same
stale-session trap as session 1's `ps -ef` lesson below).

## What Worked

- **`gdb --batch -ex run -ex "bt full" -ex "thread apply all bt"` against a
  live process fed real sim topics** — this is what actually found the LIO
  crash's true cause. Reading the code and guessing (IMU `dt` validation)
  was _plausible_ but wrong; only the backtrace showed the real path
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
  conflict, as long as no _other_ source claims a different parent for the
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

**Primary: implement the approved (and, as of session 2, revised) plan at
[`/home/thakk100/.claude/plans/recursive-beaming-finch.md`](/home/thakk100/.claude/plans/recursive-beaming-finch.md)**
— but note steps 2–3 of its Phase 1 are **done** as of session 3; the live
blocker is the depth encoding. In order:

1. **Write the depth converter** (`32FC1` metres → `16UC1` millimetres) and
   put it between the sim and cuVSLAM. This is the whole blocker — see
   "Session 3" above for the evidence. Then set `depth_scale_factor: 0.001`
   in [`_container_isaac_slam.launch.py`](src/g1_isaac_slam/launch/_container_isaac_slam.launch.py)
   and remap both `visual_slam/depth_0` and nvblox's `depth/image` to the
   converted topic. Cheapest place to run it is inside the container
   (already has the Isaac ROS/NITROS stack); a plain host-side Python
   `cv_bridge` node also works and is easier to iterate on — pick one and
   note the choice, don't leave both.
2. `df -h /` — confirm ≥30G headroom before any Docker build (37G free at
   the end of session 3, this host runs chronically tight; "Session 2" has
   the cleanup playbook).
3. Rebuild + rerun: `docker build -t g1_isaac_slam:latest -f
   isaac_ros_ws/Dockerfile.g1_isaac_slam .` from the workspace root, then
   `ros2 launch g1_isaac_slam isaac_slam.launch.py detection:=false`.
   Tear down fully between runs (see end of "Session 3").
4. **Verify** per the plan's end-to-end section: `ros2 topic hz
   /visual_slam/tracking/odometry` (currently silent — that's the symptom to
   clear), `tf2_echo vslam_odom camera_color_optical_frame` shows live
   values, nvblox mesh visible in RViz, `nvidia-smi` VRAM checked with
   detection off *then* on (the real go/no-go given the 8GB budget). Re-check
   `/tmp/cuvslam_debug/depths/*.npy` once after the converter lands — the
   file should now be ~614KB of *plausible* millimetre values, not noise —
   then set `enable_debug_mode: false` to stop the per-frame dump writes.
5. Only once that's solid: `people_segmentation` (the arg is declared but
   deliberately not wired — `isaac_ros_unet`/PeopleSemSegNet feeding nvblox's
   mask input), the `slam_type:="isaac"` branch in
   `sim.launch.py`/`sim_teleop.launch.py` to unify entry points (not
   blocking — `g1_isaac_slam`'s own launch file is a fine interim entry
   point), and the lidar-fusion piece of the original ask (nvblox does
   support a lidar integrator alongside the camera one; not scoped in
   Phase 1/2 above, RGBD-only — revisit once RGBD is proven).

RGBD mode (not stereo) is a deliberate choice — no stereo IR camera pair
exists in sim (real gap, documented in the plan), RGBD reuses the existing
D435i topics unmodified.

**Explicit scope note — don't split effort across three SLAM stacks:**
cuVSLAM + nvblox is the active direction now. Two things NOT to pursue in
parallel:

- **`plain_slam_ros2` (the LIO node)** — it's fixed and working (items 2–4
  above), keep it running as the passive comparison baseline the plan calls
  for, but **don't spend further effort tuning/optimizing it**. It's
  CPU-only by architecture (no CUDA anywhere in `joint_optimizer.cpp`/
  `normal_map.cpp`) — that ceiling is why cuVSLAM is happening at all.
- **`src/Ultra-Fusion`** — a third, _separate_ multi-sensor LVIO SLAM
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
