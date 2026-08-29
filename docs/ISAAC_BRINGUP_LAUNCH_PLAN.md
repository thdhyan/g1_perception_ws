# Plan: `isaac_bringup.launch.py` — WBC + detection on top of the standalone Isaac Sim

Status: **plan, not implemented**. This file is the contract for the launch file
to be added at `src/g1_bringup/launch/isaac_bringup.launch.py`.

## Premise (different from the Gazebo bringup)

Isaac Sim **runs independently of ROS launch**. It is started manually and stays
up across launches:

```bash
cd ~/Projects/thesis/G1_sim && python scripts/g1_warehouse_sim.py --headless --wbc-mode external
```

The launch file therefore must NOT start any simulator. It only starts ROS-side
consumers/drivers, exactly like `real.launch.py` does for the real robot.
Teardown order matters less than with Gazebo (no gz process group to kill), but
the sim should be restarted if its state wedges (`pkill -9 -f g1_warehouse_sim`).

## What the sim already publishes (verified 2026-08-25)

| Topic | Type | Notes |
|---|---|---|
| `/livox/mid360/points/a` | PointCloud2 (~10 Hz) | frame `mid360_link`, ~190k pts/s |
| `/joint_states` | JointState | 29 DOF, sim ground truth |
| `/tf`, `/tf_static` | TFMessage | full robot tree, root frame `World` |
| `/clock` | Clock | sim time — every node here MUST set `use_sim_time:=true` |
| `/g1/imu` | Imu | pelvis IMU (200 Hz) |
| `/livox/imu`, `/camera/imu` | Imu | onboard IMUs (parity with real robot) |
| `/camera/color/image_raw`, `/camera/depth/image_rect_raw`, `/camera/color/camera_info`, `/camera/depth/points` | Image/CameraInfo/PointCloud2 | D435 |
| `/g1/cmd_vel` | Twist | OG-graph subscriber (readable by nodes) |

## What the sim subscribes

| Topic | Contract |
|---|---|
| `/g1/joint/<name>` ×29 | `std_msgs/Float64` position targets @50 Hz — consumed by the sim when started with `--wbc-mode external` (same contract as the Gazebo bridge) |

## Launch file contents

1. **wbc_node** (`g1_wbc`) — identical to the Gazebo usage:
   ```python
   Node(package="g1_wbc", executable="wbc_node",
        parameters=[{
            "balance_policy_path": "<G1_sim>/assets/policy/GR00T-WholeBodyControl-Balance.onnx",
            "walk_policy_path":    "<G1_sim>/assets/policy/GR00T-WholeBodyControl-Walk.onnx",
            "imu_topic": "/g1/imu",
            "joint_states_topic": "/joint_states",
            "cmd_vel_topic": "/g1/cmd_vel",
            "control_hz": 50.0,
            "use_sim_time": True,
        }])
   ```
2. **Detection** — `livox_detection_node` (`voxelnext` default), input
   `/livox/mid360/points/a` (**note the `/a` suffix**, single-prim publisher),
   `accumulate_frames`/`max_hz` unchanged from Gazebo values,
   `use_sim_time:=true`.
3. **Snapshot trigger devices** (parity with the Gazebo bringup):
   - reuse the operator-gated flow from `real_live_detection.launch.py`:
     detection markers on `/g1/detection_markers/livox`, selection on
     `/g1/selected_human`, arming via `/g1/approach_selected`;
   - expose the 2-pass cloud-snapshot capture as a **ROS service**
     (`std_srvs/Trigger` on `/g1/lidar_snapshot/take`) backed by a small node
     that latches N clouds off `/livox/mid360/points/a` and writes a bag/PCD —
     port of the Gazebo snapshot step, source topic renamed.
4. **Keyboard teleop** (opt-in `teleop:=true`):
   `teleop_twist_keyboard` remapped `/cmd_vel → /g1/cmd_vel`.
5. **RViz** (opt-in `rviz:=false` default): Fixed Frame `World`,
   Simulation Time ON, PointCloud2 `/livox/mid360/points/a`, RobotModel,
   MarkerArray `/g1/detection_markers/livox`.

## Arguments

`wbc:=true|false`, `detection:=true|false`,
`detection_algorithm:=voxelnext`,
`teleop:=false`, `rviz:=false`, `checkpoint_path:=…`.

## Gotchas carried over

- Source system ROS (`/opt/ros/jazzy`), NOT the isaac venv, for these nodes;
  export `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and `ROS_DOMAIN_ID=0` or DDS
  discovery with the sim silently fails.
- wbc_node needs `onnxruntime`: use `~/Projects/thesis/g1_perception_ws/.venv`
  (has 1.28 CPU) after `source install/setup.bash`.
- Do NOT `unset` the GTK vars requirement away — RViz still needs
  `unset GTK_PATH GIO_MODULE_DIR LOCPATH` when launched from VSCode terminals.
- Detection was tuned on Gazebo densities; Isaac publishes ~190k pts/s vs
  Gazebo's grid pattern — re-check `score_threshold` once before trusting runs.

## Implementation steps

1. Write `isaac_bringup.launch.py` per above (start: wbc + rviz only).
2. Add the snapshot-service node (`g1_perception` pkg or new small node in
   `g1_bringup`), unit-test against a bagged `/livox/mid360/points/a`.
3. Wire detection + teleop args; verify end-to-end walking + detections in RViz.
4. Mirror any param drift back into `real_live_detection.launch.py` so real and
   sim stay one mental model.
