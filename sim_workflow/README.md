# G1 Gazebo Sim — Workflow & Handoff

**Status as of 2026-08-16.** Goal: G1 stands/balances in Gazebo Harmonic under
the GR00T `decoupled_wbc` Balance/Walk ONNX policies.

Original symptom: Gazebo loads the robot fine, but it **falls / collapses face
first**. Four independent bugs were found; all four are fixed in the source.
The last fix (TF tree) was applied but **its end-to-end verification run was
interrupted — see [Open / unverified](#open--unverified)**.

---

## 1. Quick start

```bash
cd ~/Projects/thesis/g1_perception_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source .venv/bin/activate          # venv now has onnxruntime; see §5

ros2 launch g1_bringup sim.launch.py rviz:=true detection:=false
```

No rebuild needed for these fixes: `install/` is `--symlink-install`, the WBC
node loads from `build/`, and the Gazebo URDF is regenerated from `src/` at
every launch.

Flags: `headless:=true` (no GUI), `rviz:=false`, `detection:=true` (needs torch
in the running interpreter), `paused:=true`.

**Walk command** (nothing walks by default — balance-only until commanded):
```bash
ros2 topic pub -r 1 /g1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.4}}"
ros2 topic pub --once /g1/cmd_vel geometry_msgs/msg/Twist "{}"   # stop
```

**Kill everything** (leaves no orphans; plain `pkill -f "gz sim"` misses some):
```bash
ps -ef | grep -E "gz sim|wbc_node|parameter_bridge|rviz2|robot_state_pub|static_transform|sim\.launch" \
  | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

---

## 2. Signal chain

```
/g1/cmd_vel ─┐
/joint_states├─► wbc_node (50 Hz, ONNX) ─► /g1/joint/<name> (Float64 ×29)
/imu/data ───┘         516-dim obs → 15-dim action    │ ros_gz_bridge
                                                       ▼
                                    /model/g1/<name> (gz.msgs.Double)
                                                       │
                                                       ▼
                              29× JointPositionController (per-joint kp/kd)
```

TF: `world → g1` (Gazebo PosePublisher, model pose) `→ base_link → pelvis`
(static TFs) `→ all URDF links` (robot_state_publisher).

---

## 3. The four bugs (all fixed)

### Bug 1 — spawned URDF had no controller at all
`sim.launch.py` defaulted `urdf_path` to
`G1_sim/assets/robot/g1_29/g1_29dof.urdf`, which has **zero `<gazebo>` tags**:
no joint controller, no IMU, no sensors. The workspace-URDF fallback was also
broken (`parents[3]` dropped the `src/` segment). Robot spawned **unactuated**
→ fell under gravity; WBC published into the void.

**Fix:** default to `src/g1_description/urdf/g1_29dof.urdf`; fallback corrected;
spawn `z` 0.98 → 0.75 (standing pelvis ≈ 0.74, so 0.98 was a ~24 cm drop).

### Bug 2 — joint gains mismatch
One flat `JointTrajectoryController` for all 15 joints: `p=150, i=0.5, d=20`.
The policy was trained against **per-joint** PD, no integral:

| group | kp | kd |
|---|---|---|
| hips | 150 | 2 |
| knees | 200 | 4 |
| ankles | 40 | 2 |
| waist | 250 | 5 |
| arms (held at 0) | 100 | 0.5 |

One gain cannot serve a 6× kp spread (40→250); `d=20` is 4–10× too high and
`i=0.5` adds windup. Same failure documented in
`G1_sim/docs/decoupled_wbc_findings.md §5` (uniform gains → pelvis 0.79→0.16 m).

**Fix:** 29 per-joint `gz::sim::systems::JointPositionController` plugins with
the trained gains, driven by one `std_msgs/Float64` per joint.

### Bug 3 — WBC was flying blind (the actual cause of the face-plant)
Even after Bugs 1–2, measurement showed:

```
/joint_states : 0 msgs      /imu/data : 0 msgs
/g1/joint/left_knee_joint : 240 msgs/8s, value ALWAYS exactly 0.3 (= default)
```

The WBC hit its `self._qpos is None` early-return and published a **static
default pose** forever. A fixed pose cannot balance. Two separate causes:

- **IMU dropped entirely.** Sensor attached to `<gazebo reference="imu_in_pelvis_link">`
  but the real link is `imu_in_pelvis` (no `_link`). Worse, that link has no
  inertia, so URDF→SDF **collapses it into a `<frame>`**, and Gazebo silently
  drops sensors on frames. Converted SDF had **0 IMU sensors**.
  **Fix:** attach to the real `pelvis` link, bake the offset into `<pose>`.
  Verified: IMU sensor count 0 → 1.
- **Wrong joint_state topic.** `JointStatePublisher` publishes on the
  world-scoped `/world/warehouse/model/g1/joint_state`. The short
  `/model/g1/joint_state` is *advertised but silent* — bridging it yielded
  nothing. **Fix:** bridge the world-scoped topic.

**Result after fix (measured):** `/joint_states` 640 Hz, `/imu/data` 193 Hz, and
the knee command took **459 distinct values** instead of a constant 0.3 —
policy genuinely running. Pelvis measured at **z = 0.739 m** (target 0.71–0.74)
instead of 0.109 m collapsed.

### Bug 4 — split TF tree + Livox plugin never loaded
RViz showed no robot model, no TF, no LiDAR points.

- **Split tree.** `PosePublisher` had `publish_link_pose=true`, emitting a
  **model-prefixed** tree (`g1 → g1/pelvis`, `g1/left_knee_link`, …, 27 frames)
  parallel to the unprefixed tree robot_state_publisher builds from the URDF
  (43 frames). Two disjoint trees, roots `g1` and `base_link`, and **no `world`
  frame at all** — while the RViz config's Fixed Frame is `world`. Hence
  nothing rendered.
  **Fix:** `publish_link_pose=false` + `publish_model_pose=true` (Gazebo
  supplies only `world → g1`), plus a new static TF `g1 → base_link` to chain
  onto the existing `base_link → pelvis`.
- **Livox plugin path off by one.** `ws_lib` used `parents[4]` →
  `/home/thakk100/Projects/thesis/install/...` (above the workspace). Gazebo
  logged `Failed to load system plugin [libros2_livox.so] : Could not find
  shared library` → LiDAR published nothing.
  **Fix:** `parents[3]`. Verified: the load error is gone.

---

## 4. Files changed

| File | Change |
|---|---|
| `src/g1_description/urdf/g1_29dof.urdf` | Flat controller → 29 per-joint `JointPositionController` w/ trained gains; IMU re-attached to `pelvis` link w/ baked pose; PosePublisher → model pose only |
| `src/g1_wbc/g1_wbc/wbc_node.py` | Publish 29 `std_msgs/Float64` per-joint targets instead of one `JointTrajectory`; arms held at 0 |
| `src/g1_bringup/config/gz_bridge.yaml` | 1 trajectory entry → 29 `Float64 ↔ gz.msgs.Double`; joint_state → world-scoped topic |
| `src/g1_bringup/launch/sim.launch.py` | Default URDF → Gazebo one; fixed fallback; spawn z=0.75; livox `parents[4]`→`[3]`; new `g1 → base_link` static TF; modern ROS 2 CLI flags for static TFs; corrected `mid360_link → livox_frame` to identity (URDF already handles sensor pitch/roll) |
| `src/g1_description/config/g1_sim.rviz` | Updated default fixed frame to `world` and set PointCloud2 LiDAR topic to `/livox/mid360/points` |
| `src/g1_wbc/launch/wbc.launch.py` | Match per-joint contract; fixed policy path |

Related: `../GAZEBO_BALANCE_FIX.md` (Bugs 1–2 in more depth).

---

## 5. Environment

`.venv` is a **uv** venv with `include-system-site-packages=true`, so it sees
ROS/rclpy from `/opt/ros/jazzy` while adding its own packages.

- `onnxruntime` 1.28.0 (CPU) — **installed this session** via
  `uv pip install --python .venv/bin/python onnxruntime` (no `pip` binary in
  this venv; use `uv`).
- `torch` 2.3.1+cu121, `torchvision`, numpy 2.4.4 — already present.

WBC needs only onnxruntime (torch is for CenterPoint). Verified in-venv: both
policies load, 516→15 inference, finite targets. The
`CUDAExecutionProvider is not in available provider names` warning is
**expected and harmless** — no system cuDNN, so it uses CPU; the model is
1.8 MB, ample at 50 Hz. For GPU later: `uv pip install onnxruntime-gpu
nvidia-cudnn-cu12`.

**Note:** `detection:=true` launches CenterPoint, which needs torch in the
*running* interpreter. Under a plain ROS shell it logs
`Backend initialization failed: No module named 'torch'` and publishes empty
detections — harmless for balance testing, hence `detection:=false` above.

---

## 6. Verification Results & Status

1. **Bug-4 & TF Tree (VERIFIED)**:
   - Single unified root `world` with 47 TF frames.
   - 0 `g1/`-prefixed frames.
   - Static TFs `world -> warehouse -> g1 -> base_link -> pelvis` chain perfectly to all URDF links.
2. **Sustained Balance (VERIFIED)**:
   - Model height holds stably at `z ≈ 0.735 - 0.747 m` (average 0.737 m, target ~0.74 m).
   - Policy actively adjusts knee joint targets (hundreds of distinct command values published at 50 Hz).
   - `/joint_states` published at ~800 Hz, `/imu/data` published at ~190 Hz.
3. **Walking Locomotion (VERIFIED)**:
   - Tested `/g1/cmd_vel` forward command `linear.x = 0.3 m/s` for 8s followed by stop recovery.
   - Robot traversed +2.91 m forward while maintaining standing/walking height `z = 0.727 - 0.762 m` with zero falls.
4. **LiDAR Stream (VERIFIED)**:
   - `/livox/mid360/points` publishes at ~9.5 Hz with 28,800 points per frame on frame `livox_frame`.
   - Fields: `x, y, z, intensity, ring` (BEST_EFFORT QoS).
5. **RViz2 Visualizer (CONFIGURED & RUNNING)**:
   - Config updated in `src/g1_description/config/g1_sim.rviz` with fixed frame `world` and PointCloud2 topic `/livox/mid360/points`.
   - Launched on active X11 display with full robot model, TF tree, and LiDAR cloud.

---

## 7. Debugging playbook

**Always probe with BEST_EFFORT QoS.** Gazebo bridges publish best-effort;
plain `ros2 topic echo` (reliable) prints nothing and looks like a dead topic.

```python
qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                 history=QoSHistoryPolicy.KEEP_LAST, depth=1)
```

Order of checks:

1. **Controllers in the spawned model** —
   `grep -c JointPositionController install/g1_description/share/g1_description/g1_gazebo_sim.urdf`
   (want 29; regenerated from `src/` each launch).
2. **Is WBC driving, or holding a static pose?** Subscribe to
   `/g1/joint/left_knee_joint` and count **distinct** values. Constant `0.3`
   = static default = no sensor feedback (Bug 3). Many values = policy live.
3. **Feedback present?** `/joint_states` and `/imu/data` must be non-zero rate.
4. **Gazebo-side truth** — compare `gz topic -l` against the bridge config;
   a topic can be *advertised but silent* (Bug 3). Needs:
   `export LD_LIBRARY_PATH=$(find /opt/ros/jazzy/opt -maxdepth 2 -name lib -type d | tr '\n' ':')$LD_LIBRARY_PATH`
   and `GZ=/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz`.
5. **Does the sensor survive URDF→SDF?** This is where silent drops happen:
   ```bash
   gz sdf -p src/g1_description/urdf/g1_29dof.urdf > /tmp/g1.sdf
   grep -c "type='imu'" /tmp/g1.sdf     # must be 3 (pelvis, mid360, d435 -- see §8)
   ```
   A `<gazebo reference="X">` whose `X` is not a **link** (typo, or an
   inertia-less link collapsed to a `<frame>`) is dropped without error.
6. **TF sanity** — collect `/tf` + `/tf_static`, compute roots
   (`parents - children`). Expect the single root `world`; two roots or any
   `g1/`-prefixed frame means the tree is split again.
7. **Height** — read `world → g1` translation z from `/tf`. Do **not** read
   `base_link → pelvis`; it is a static identity (z=0) and will read as a
   false collapse.

   > **Note (2026-08-17, §8 below):** items 6-7 describe the single-tree
   > `world → g1 → base_link → pelvis` architecture from the original
   > verification run. That architecture was superseded by the dual-tree
   > design (`warehouse → g1 → gt_base_link → gt_pelvis` for ground truth,
   > `map → odom → pelvis → base_link` for proprioception/SLAM) — there is
   > no longer a single `world`-rooted tree, and `base_link` now hangs off
   > `pelvis` rather than the other way around. See §8 for the current
   > architecture and how to sanity-check it.

---

## 8. [2026-08-17] TF continuity, LIO crash fix, VoxelNeXt detector, moving actor

Follow-up session against the dual-tree architecture from
[TASKS.md](../TASKS.md) §3.4 / [UPDATES.md](../UPDATES.md). Four separate
fixes, landed as separate commits across `g1_perception_ws`,
`g1_description`, and `plain_slam_ros2` (the latter two are submodules,
forked to `thdhyan/*` since this session had no push access to their
upstreams — see `.gitmodules`).

### TF tree: multi-parent conflict + no startup fallback
`base_link → pelvis` (static, left over from the pre-dual-tree design)
collided with `lio_3d_node`'s dynamic `odom → pelvis` broadcast — two
publishers claiming different parents for the same frame. Flipped to
`pelvis → base_link` (pelvis's only parent is now `odom`) and added
identity fallback statics for `map → odom` / `odom → pelvis` so the tree is
connected from `t=0` instead of only after `lio_3d_node`'s first scan.
Also removed a dead `g1_29dof → gt_base_link` static publisher that
duplicated the ground-truth tree's real `g1 → gt_base_link` parent.

### lio_3d_node SIGABRT
Root-caused via `gdb` against live sim topics (not guesswork — the crash
wasn't reproducible from reading the code alone): `JointOptimizer::Estimate()`
can produce a non-finite Gauss-Newton step when scan-to-map correspondences
are too few (near-singular `H`/`P` inversion). The resulting NaN/Inf
reaches `Sophus::SO3::exp()`, which hard-`abort()`s — a raw signal, not a
C++ exception, so it skipped straight past the existing try/catch around
`SetScanCloud()`. Fixed with a finite-check guard before `exp()`.

Separately (real cause of the *non-convergence*, distinct from the crash):
`lio_3d_params.yaml`'s `imu_to_lidar` extrinsic was calibrated for an IMU at
`torso_link`'s origin, but the launch files feed `lio_3d_node` the **pelvis**
IMU — a different link, off by ~14cm/5cm in Z/X. Recomputed from the URDF
kinematic chain. (`waist_yaw_joint`/`waist_roll_joint`/`torso_joint` are all
`type="fixed"` in this URDF, so pelvis/torso/lidar are one rigid body in
sim — a constant extrinsic is valid here, no need to source LIO from the
lidar's own onboard IMU instead.)

Added onboard IMU sensors for Mid-360 (`/livox/imu`) and D435i
(`/camera/imu`) to the URDF, matching real-robot topic parity (`/livox/imu`
is documented in [PLAN.md](../PLAN.md) §4 but wasn't previously simulated).

### VoxelNeXt detector: wrong checkpoint, now verified working
`detection_algorithm:=voxelnext` was silently falling back to a PointPillar
clustering heuristic every launch — the launch files were passing the
CenterPoint/PointPillar checkpoint to it regardless of algorithm, which
made `load_params_from_file()` raise `KeyError('model_state')`. Fixed
(`VOXELNEXT_CHECKPOINT_PATH`, see [livox_detection/README.md](../src/livox_detection/README.md)).
Verified against a live headless sim run: real per-frame varying scores and
pedestrian counts, not a frozen fallback value.

Also added `class_filter` (default `"pedestrian"`, drops `car`/`cyclist`)
and aligned `accumulate_frames` to the `max_hz` cadence (2 frames = 200ms
at the Mid-360's 10Hz scan rate, matching a 5Hz inference cap 1:1 instead
of the old 4-frame/400ms mismatched window). Achieved rate is ~4.0-4.2Hz
against the 5Hz cap — real inference latency, not further tunable via
these params. See [livox_detection/README.md](../src/livox_detection/README.md)
for the full parameter reference.

### Moving pedestrian actor
`g1_warehouse.sdf`'s five `human_1`..`human_5` models are static primitive
geometry — no motion to test detection tracking against. Added a sixth,
`human_walking`, using Gazebo Harmonic's `<actor>`/`<trajectory>` system
(https://gazebosim.org/docs/harmonic/actors/): walks a straight 8m path
(X=-3, Y: -4↔4, clear of the shelves and the static humans) at ~1.2 m/s,
loops forever. Requires network access on first launch to fetch the actor
mesh from Fuel (cached locally afterward, per Gazebo's usual behavior).
