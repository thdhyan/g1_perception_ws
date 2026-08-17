# Gazebo: G1 balance / collapse — root cause & fix

**Symptom:** Gazebo Harmonic spawns the G1 fine, the WBC node runs, but the
robot can't balance — it falls / collapses.

Two independent bugs stacked, both on the Gazebo actuation side. The policy
math and ONNX contract were already correct (verified: 516-in → 15-out).

---

## Bug 1 (primary): the spawned URDF had no controller at all

`sim.launch.py` defaulted `urdf_path` to
`~/Projects/thesis/G1_sim/assets/robot/g1_29/g1_29dof.urdf`, which contains
**zero `<gazebo>` tags** — no joint controller, no IMU, no sensors. The
fallback path was also broken (`parents[3]` dropped the `src/` segment).

Result: the robot spawned **unactuated** (pure gravity → falls) and the WBC
was publishing to a topic nothing listened on.

**Fix** (`src/g1_bringup/launch/sim.launch.py`): default `urdf_path` to the
Gazebo-tailored URDF `src/g1_description/urdf/g1_29dof.urdf` (has sensors +
per-joint controllers). Fallback corrected. Spawn `z` lowered `0.98 → 0.75`
(standing pelvis height ≈ 0.74; 0.98 meant a ~24 cm drop on spawn).

## Bug 2: joint gains mismatch (the documented collapse cause)

The old Gazebo controller was a single
`gz-sim-joint-trajectory-controller-system` with one flat gain set for all 15
joints: `p=150, i=0.5, d=20`.

The GR00T `decoupled_wbc` Balance/Walk policies were trained against
**per-joint** PD gains (no integral):

| joint group      | kp (per joint)          | kd |
|------------------|-------------------------|----|
| hips             | 150                     | 2  |
| knees            | 200                     | 4  |
| ankles           | 40                      | 2  |
| waist (3)        | 250                     | 5  |
| arms (14)        | 100 (held at 0)         | 0.5|

A single flat gain can't serve a 6× kp spread (40→250); `d=20` is 4–10× too
high and `i=0.5` adds windup. This is exactly the failure documented in
`G1_sim/docs/decoupled_wbc_findings.md §5` (uniform gains → pelvis
0.79→0.16 m collapse; per-joint gains → stable 0.71–0.74 m).

**Fix:** replace the flat controller with **29 per-joint**
`gz::sim::systems::JointPositionController` plugins, one per DOF, each with
its trained `p_gain`/`d_gain` (i=0).

---

## Files changed

| File | Change |
|------|--------|
| `src/g1_description/urdf/g1_29dof.urdf` | Flat trajectory controller → 29 per-joint `JointPositionController` plugins with trained gains |
| `src/g1_wbc/g1_wbc/wbc_node.py` | Publish one `std_msgs/Float64` per joint (29) instead of a single `JointTrajectory`; arms held at 0 |
| `src/g1_bringup/config/gz_bridge.yaml` | 1 trajectory bridge entry → 29 `Float64 ↔ gz.msgs.Double` entries |
| `src/g1_bringup/launch/sim.launch.py` | Default URDF → the Gazebo one; fixed fallback path; spawn z=0.75; WBC param `joint_topic_prefix` |
| `src/g1_wbc/launch/wbc.launch.py` | Match new per-joint contract + policy path |

## Signal chain (now)

```
/g1/cmd_vel ─┐
/joint_states├─► wbc_node (50 Hz) ─► /g1/joint/<joint_name>  (Float64, ×29)
/imu/data ───┘        │                    │ ros_gz_bridge
                      │                    ▼
                policy out (15)   /model/g1/<joint_name> (gz.msgs.Double)
                                  │
                                  ▼
                        29× JointPositionController (per-joint kp/kd)
                                  │
                                  ▼
                              G1 joints
```

## Run

```bash
cd ~/Projects/thesis/g1_perception_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# no rebuild needed: install is --symlink-install (Python/YAML/URDF edits are live)
# the Gazebo URDF is regenerated from src/ at launch time

ros2 launch g1_bringup sim.launch.py rviz:=true
# headless:  ros2 launch g1_bringup sim.launch.py rviz:=false detection:=false
```

The robot should rise to and hold pelvis ≈ 0.71–0.74 m. To walk:
```bash
ros2 topic pub -r 1 /g1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.4}}"
# stop:
ros2 topic pub --once /g1/cmd_vel geometry_msgs/msg/Twist "{}"
```

## If it still doesn't balance — diagnose in this order

1. **Is the controller even in the model?**
   `grep -c JointPositionController install/g1_description/share/g1_description/g1_gazebo_sim.urdf`
   (must be 29; it's regenerated from src each launch)

2. **Is the WBC actually driving?** (should be 1450 msgs/s ≈ 29 topics × 50 Hz)
   ```bash
   ros2 topic hz /g1/joint/left_hip_pitch_joint
   ros2 topic list | grep '/g1/joint/'
   ```
   If zero: the WBC isn't publishing — check `ros2 run` log for
   `Failed to load policies` (onnxruntime missing in the runtime interpreter)
   or a bridge/QoS error.

3. **Is the policy loaded or falling back to a static hold?** A WBC that can't
   find the ONNX files holds the default pose, which is *not* a balance policy
   → the robot topples. Look for `WBC policies loaded` in the log.

4. **Gains sanity.** If standing but "mushy"/overshooting, nudge the ankle
   `p_gain` (currently 40) or knee (200) in the URDF per-joint blocks. Do not
   reintroduce a single flat gain or an integral term.

## Run with the venv (self-contained)

The `.venv` (uv, `include-system-site-packages=true`) now has everything the
WBC needs: `rclpy` + ROS msg packages (system-site, via `setup.bash`) and
`onnxruntime` 1.28.0 (CPU) installed into it. `torch` 2.3.1+cu121 is already
there for the CenterPoint detection node.

Run the launch with the venv's interpreter first on PATH:

```bash
cd ~/Projects/thesis/g1_perception_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
source .venv/bin/activate          # now `python3`/`ros2` resolve to the venv
ros2 launch g1_bringup sim.launch.py rviz:=true
```

Verified end-to-end in the venv: both ONNX policies load, 516→15 inference,
finite leg targets. `onnxruntime` is CPU here (no system cuDNN); the policy is
1.8 MB so CPU is ample at 50 Hz. The `CUDAExecutionProvider` warning is
expected and harmless — it falls back to CPU.

To get a GPU onnxruntime later (optional): install `onnxruntime-gpu` **and**
cuDNN (e.g. `uv pip install onnxruntime-gpu nvidia-cudnn-cu12`) into the venv,
since onnxruntime-gpu's CUDA EP requires cuDNN, which is not currently on the
system.
