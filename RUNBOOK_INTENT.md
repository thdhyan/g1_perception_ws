# Runbook: PhaseSpace labeller + intent-sim LLM planner + G1 goto

What runs:

| Package | Node | Role |
|---|---|---|
| `g1_mocap_labeler` | `labeler_gui` (Tk) | Name each PhaseSpace LED group, set it as human or robot, capture its heading. Publishes TF `phasespace -> <name>` and `/mocap/agents`. |
| `g1_mocap_labeler` | `fake_phasespace` | Laptop stand-in for the tracker: 3 hats, and the robot hat follows the dry-run velocity. |
| `g1_intent` | `intent_planner` | On `/intent/task`, snapshots `/mocap/agents` and runs intent-sim's `run_policy` (proxemics prompt, function calling). Publishes `/intent/plan`. |
| `g1_intent` | `intent_review_gui` (pygame) | intent-sim style view of the live scene with the LLM trace, pose goals, goto prediction and velocity preview. You approve, replan or E-stop here. Records `review.mp4`. |
| `g1_intent` | `goto_executor` | Runs an approved plan closed-loop on the PhaseSpace pose (never odometry). Sends `set_velocity` through `robot_bridge` to `LocoClient.SetVelocity`, then stops and gestures at the terminal call. |

Data flow: `PhaseSpace TF -> labeler -> /mocap/agents -> planner -> /intent/plan -> review GUI (approve) -> executor -> robot_bridge socket -> LocoClient`

---

## 0. One-time setup

```bash
cd ~/Projects/thesis/g1_perception_ws
source setup_g1_env.sh
colcon build --symlink-install --packages-select g1_mocap_labeler g1_intent
source install/setup.bash
```

LLM keys go in `~/Projects/thesis/intent-sim/.env` (the planner loads it). Status on 2026-10-07:

| Provider | Status |
|---|---|
| `openai` | Works. `gpt-6.1-sol` exists and planned in about 7 s. |
| `openrouter` | Works. `anthropic/claude-sonnet-4.5` planned in about 11 s. |
| `nim` | 403 |
| `gemini` | Prepaid credits depleted |
| `ollama` / `dl` | Lab box unreachable off the lab network |
| Local `gemma4:12b` | Makes no tool calls, so not usable |

Provider and model can also be switched live, without a restart:

```bash
ros2 param set /intent_planner provider openrouter
ros2 param set /intent_planner model anthropic/claude-sonnet-4.5
```

---

## Web console (default, `ui:=web`)

`intent.launch.py` now opens **one RViz + one browser page** (`web_console`, http://localhost:8090, `port:=`). `ui:=tk` brings back the Tk labeller + pygame review windows described below.

The page is intent-sim's run browser (scene canvas, System Prompt / Steps / SNGNN / Nav / VO tabs, run list over `intent_runs/`) plus a **live** run (the current `/mocap/agents` scene, refreshed in place) and three tabs:

- **Live**:
  - Task, provider/model, variant (a/r/ra/rs), style, scene format, temperature and anchor. Sent as JSON on `/intent/task`; empty fields fall back to the planner's launch params.
  - `ra` needs an anchor: pick it in the dropdown or click a human on the canvas (gold ring). Earlier `ra` failed because the planner never passed `scene_params={"anchor": ...}`; it does now.
  - Preview prompt renders the system prompt for the live scene.
  - The plan card shows the LLM calls, goals, terminal call, raw content, and "Open trace".
  - **Low-level planner**:
    - `direct`: the goto controller goes straight to each goal.
    - `A*` / `GD` / `Adam` / `AdamW`: path from robot to each goal on the live SNGNN + GMM + obstacle cost map, using the Nav tab weights. It is inserted as waypoint goals every 0.5 m (reached at 0.25 m, no turn), and the plan is republished as `<id>~<planner>` before approving.
  - Preview path draws the path; Approve computes it, publishes, and approves once the executor has loaded it.
  - Big E-STOP button, executor state, status log.
- **Labeler**: the same Labeler node, run headless in the web_console process.
  - Name + Apply, Role toggle, Capture, Save.
  - **Heading slider** (−180…180°): drag it to see the orange arrow on the canvas, release to apply. For a captured hat, apply rotates the layout so its current facing becomes that heading.
  - **Static humans** (one tracker, several people standing still): put the hat on/at person 1, pick the hat, type a name, optional heading (slider or box; empty = the tracker's yaw), press **Add** and hold still 1 s (the pose is averaged over 20 samples). Move the hat to person 2, repeat. Leave the tracker hat itself unnamed, else it is also a moving human.
  - They are saved in the labels file under `static_humans` (name: x, y, z, yaw_deg, hat), reloaded on start, and published in `/mocap/agents` (role human, `yaw_source: static_pose`) + TF `phasespace -> <name>`, so the planner, the scene and the bag see them. **Delete** in their table removes one.
- **Robot**: robot_bridge buttons: walk mode, stop, FSM, wave, shake hand, damp (asks for confirmation).
  - Refused in `dry_run`.
  - Only Stop is allowed while a plan runs.

**RViz G1 model:** `phasespace.launch.py` publishes a static TF `ps_robot_head -> ps/pelvis` (`hat_to_pelvis_x/y/z`, default z = −0.55 m: head top 0.525 m above the pelvis plus the LEDs). It also runs robot_state_publisher on `g1_29dof.urdf` with the `ps/` prefix (joints zero; `model_joints:=robot` uses the robot's `/joint_states`). Tune `hat_to_pelvis_z` by eye until the feet touch the floor. With `fake_mocap:=true`, `tracker:=false` keeps robot_head, the model and RViz running on the fake TFs.

## 1. Laptop-only dry run (no robot, no PhaseSpace)

```bash
ros2 launch g1_intent intent.launch.py fake_mocap:=true dry_run:=true \
    provider:=openai model:=gpt-6.1-sol
```

1. In the **labeller** window, label each group.
   - Layout: the left side has one table row per PhaseSpace group, then the **Name** and **Heading** boxes below it. The right side has a live 2D top view of every group (robot: red square; humans: blue dots; LEDs: grey) with the selected group ringed, and a heading **dial** and **slider** under it.
   - Name: select a row with Up/Down, type the name, Enter. Ctrl+R toggles the role, so set `dhyan-hat-3` to robot.
   - Heading: drag the dial (snaps to 5°; hold Shift for free), move the slider, type it, or press Ctrl+Left/Right (±5°; add Shift for ±1°). The dashed orange arrow previews it.
   - With the hat held still and facing the dial heading, press Ctrl+P to capture its LED layout. The yaw is then measured live (`src` = `fit`). Ctrl+S saves.
   - To correct a captured group's heading later, set the dial and press Enter. Its current facing becomes that heading; no re-capture needed.
   - Fake headings: `dhyan-hat` = 180, `dhyan-hat-2` = 150, `dhyan-hat-3` = 0.
2. In the **review** window, type a task (e.g. `Go greet alice with a wave`) and press Enter. Wait for the plan.
3. Check the plan:
   - the orange LLM trace;
   - the blue dashed goto prediction;
   - the goal list with max |vx|, |vy|, |wz| and time. A red goal row means the prediction did not reach it.
4. Approve by pressing Enter twice within 1.5 s on an empty box (the panel turns red after the first press). Reject or stop with Esc. Replan with Ctrl+R.

Outputs go to `intent_runs/<run_id>/`:
- `trace.json` (intent-sim trace plus a `g1_intent` block holding the mocap snapshot and goals)
- `system_prompt.md`
- `review.mp4`
- `execution_report.json` (per-goal reached pose, errors, 10 Hz timeline, gesture result)

---

## 2. Real robot + PhaseSpace

Run each step in its own terminal, in this order. Every terminal starts with:

```bash
cd ~/Projects/thesis/g1_perception_ws && source setup_g1_env.sh && source install/setup.bash
```

| # | Terminal | Command |
|---|---|---|
| 1 | Robot sensors (on the G1 via ssh/tmux) | `./scripts/start_robot_sensors.sh "src/g1_sensors.launch.py joint_states:=true"` |
| 2 | Robot bridge (unitree_sdk2py, **not** in the same process as rclpy) | `python3 src/g1_control/g1_control/robot_bridge.py enp2s0` |
| 3 | Intent stack (includes PhaseSpace) | `ros2 launch g1_intent intent.launch.py provider:=openai model:=gpt-6.1-sol` |
| 4 | Recording (optional) | `record:=true` on step 3 (bag only), or `./scripts/record_intent_session.sh --name take1` (bag + VoxelNeXt, section 3) |

Before approving anything:

1. **Hand on the remote E-stop.** The review GUI's Esc sends a zero-velocity stop through the bridge, but it is software, not a safety system.
2. **Labeller:**
   - Name the humans by hat.
   - Set the **robot** role on `dhyan-hat-3`.
   - Stand the robot facing PhaseSpace +x (or type its true heading), then press Ctrl+P to capture. Capture every human hat with its wearer facing a known direction. Press Ctrl+S.
   - The executor refuses to start unless the robot's `src` column shows `fit` or `tracker`.
3. **Canvas placement:**
   - The LLM sees a 10 × 7 m room. PhaseSpace (x, y) maps to canvas (x − ox, oy − y), with the default `canvas_origin_x:=-5.0 canvas_origin_y:=3.5`, so the canvas is centred on the PhaseSpace origin.
   - If the planner warns that an agent is "outside the canvas", relaunch with a different origin.
4. Approve. Watch for these stops:
   - **Pose lost** (no fresh PhaseSpace pose for 0.3 s): the robot holds and stops. The SDK watchdog also stops it after 0.3 s without commands.
   - **Human closer than 0.35 m** to the robot: the executor pauses.

Useful launch arguments:

| Argument | Values |
|---|---|
| `provider` | openai, openrouter, nim, gemini, ollama, dl, spark01..03 |
| `model` | Any model ID for that provider |
| `base_url` | Override the endpoint, e.g. `http://localhost:11434/v1` |
| `dry_run` | `true`: drive nothing; the review GUI still works |
| `phasespace` | `false`: PhaseSpace already running, or a bag is being replayed |
| `robot_group` | Default `dhyan-hat-3` |
| `canvas_origin_x`, `canvas_origin_y` | Canvas placement (see step 3) |
| `prompt_style` | Default `proxemics` |
| `temperature` | Default 0.7 |
| `runs_dir` | Where run outputs go |
| `record` | `true`: ros2 bag into `bag_dir` (section 2b) |
| `bag_dir` | Default `/Storage/g1_bags` |
| `slam` | `true`: LIO odometry + SLAM on the robot's Mid-360 (section 2b) |
| `obstacles` | `true`: LiDAR walls / tables as scene obstacles (section 2b) |

Provider examples:

```bash
ros2 launch g1_intent intent.launch.py provider:=openrouter model:=anthropic/claude-sonnet-4.5
ros2 launch g1_intent intent.launch.py provider:=dl model:=qwen3:32b          # lab Ollama
ros2 launch g1_intent intent.launch.py provider:=gemini model:=gemini-3.8-flash
```

Trigger a plan without the GUI:

```bash
ros2 topic pub --once /intent/task std_msgs/msg/String "{data: 'Go greet alice with a wave'}"
```

The terminal call maps to an arm action:

| LLM call | Robot action |
|---|---|
| `interact(action=talk / low_wave)` | stop + `wave` |
| `interact(action=shake_hand)` | stop + `shake_hand` |
| `listen` / `stand_still` | stop only |

The distance to each target is logged in `execution_report.json`.

---

## 2b. LiDAR on the laptop: recording, SLAM / odometry, obstacles

```bash
ros2 launch g1_intent intent.launch.py record:=true slam:=true obstacles:=true
```

The robot needs only step 1 of section 2. `livox_ros_driver2` on the Orin publishes `/livox/lidar` (10 Hz, `mid360_link`) and `/livox/imu` over ethernet, and Jazzy reads Foxy fine. Everything below runs on the laptop.

**`record:=true`**
- Writes a sqlite3 bag, split every 4 GB, to `/Storage/g1_bags/intent_<date>_<time>`.
- It records:
  - The `scripts/record_bag.sh --source real` set: Mid-360, D435i images and `camera_info`, `/camera/depth/color/points`, `/joint_states`, `/lowstate`, `/secondary_imu`, `/state_estimator/odom_pelvis`, `/tf`, `/tf_static`, `/robot_description`.
  - `/phasespace/*`, `/mocap/agents` and `/intent/*` (including the obstacles).
  - `/pslam/imu_odom`, `imu_pose`, `pose_graph_poses` and `filtered_map_cloud`.
- The topic list is a regex, so topics that start after the recorder (robot sensors, new hats) are still picked up.
- `/lowstate` and `/sportmodestate` need `unitree_hg` / `unitree_go` built on the laptop; otherwise they are skipped.
- **Size:** about 100 MB/s with the D435i point cloud. Start the sensors with `pointcloud:=false` to save about 55 MB/s.

**`slam:=true`** (`lidar_slam.launch.py`, also runs standalone)
- Runs `plain_slam_ros2` `lio_3d_node` (odometry: `/pslam/imu_odom`, `/pslam/imu_pose`) plus `slam_3d_node` (loop-closed map `/pslam/filtered_map_cloud`, saved to `/Storage/pslam_maps/`).
- Params come from `g1_intent/config/pslam_real/`: the Mid-360 internal-IMU extrinsic (Livox spec) and `acc_scale` 9.80665, because the driver publishes acceleration in g. The sim pelvis-IMU calibration does not apply here.
- TF is `lio_map -> lio_odom -> lio_imu`, separate from the robot's own tree and from `phasespace`. Set the RViz fixed frame to `lio_map` to see the map.
- **Not yet run on real data:** no recorded bag with `/livox/imu` existed on 2026-10-21. Record one take with `record:=true`, then replay it with `ros2 bag play` against `lidar_slam.launch.py` to check drift against PhaseSpace (`/phasespace/robot_head`).

**`obstacles:=true`** (`lidar_obstacles`)
- Each sweep is placed in `phasespace` through the G1 model under the head hat (`phasespace -> ps_robot_head -> ps/pelvis -> ps/mid360_link`), so **tune `hat_to_pelvis_*` and capture the robot heading first**. A heading error rotates the walls with every turn.
- Points between z 0.15 and 1.6 m vote into a 0.1 m grid over the canvas. Points within 0.5 m of any `/mocap/agents` agent (people, the robot) are ignored. The score decays with a half-life of about 3 s.
- Occupied cells merge into up to 25 boxes on `/intent/obstacles`. RViz shows the grid as "LiDAR obstacles".
- The boxes become the scene's `obstacles` for:
  - the LLM prompt ("non-navigable rectangular obstacle ...")
  - the web console's scene, Nav and VO tabs
  - the **A\* / GD / Adam waypoints**: pick a low-level planner other than `direct` to go around them. `direct` drives straight.
- Variant `r` leaves the boxes out (intent-sim's relative encoding has no obstacles). Boxes older than 3 s are dropped.
- Without PhaseSpace: `ros2 run g1_intent lidar_obstacles --ros-args -p fixed_frame:=lio_map -p "sensor_frame:=''"` maps in the SLAM frame. The canvas is not aligned to `lio_map`, though, so the planner cannot use those boxes yet.
- **Not done:** the goto executor has no live LiDAR stop. Obstacle avoidance is only in the waypoint path, so keep the hand on the E-stop.

**Foxy/Jazzy:** the laptop already publishes `/tf` (robot_head, labeler, the `ps/` model), and now also `lio_*`. That is fine as long as no robot node subscribes to `/tf`. If the robot floods with `serdata.cpp:308`, check `ros2 topic info /tf -v` on the robot.

---

## 3. Data collection (for off-robot / off-PhaseSpace testing)

```bash
./scripts/record_intent_session.sh --name take1 --out /Storage/bags   # VoxelNeXt + bag
./scripts/record_intent_session.sh --no-detect                        # bag only
./scripts/record_intent_session.sh --no-images                        # without raw camera
```

- It starts `g1_bringup real_live_detection.launch.py algorithm:=voxelnext follow:=false`, so perception runs without driving the robot.
- It records a sqlite3 bag (readable by `offline_pipeline/bag_processor.py`) with these topics:
  - LiDAR: `/livox/lidar`, `/utlidar/cloud`, `/livox/mid360/points`
  - Cameras: color and depth images plus `camera_info`
  - Robot state: `/joint_states`, `/lowstate`, `/tf`, `/tf_static`
  - PhaseSpace: `/phasespace*`
  - Labeller, LLM and detections: `/mocap/agents`, `/intent/*`, `/g1/*`
- The bag splits every 4 GB.
- **Disk:** `/` has about 18 GB free and `/Storage` was not mounted on 2026-10-07. Raw images run at about 45 MB/s, which fills 18 GB in about 6 minutes. Use `--out` on a big disk, or `--no-images`.

Replay with everything off:

```bash
ros2 bag play bags/take1 --clock
ros2 launch g1_intent intent.launch.py phasespace:=false dry_run:=true   # labeller works on the replayed LED TFs
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Labeler shows `microdriver-*` groups with 0/8 LEDs | Server default session profile changed. `intent.launch.py` now passes `profile:="Dhyan Hat"` (default); check the profile in the Configuration Manager. |
| Robot stops ~0.3 m short and each `move_to_x_y` waits 60 s (2026-10-21) | The G1 steps in place below ~0.15 m/s. `goto_controller.py`: `POS_TOL` 0.20 m, and a goal within `NEAR` with < 2 cm progress in 1 s counts as arrived (`stalled_m` in the report). Bridge `TURN_IMU_SETTLE_S` 0.8 -> 0.1 s. |
| `no fresh /mocap/agents` | Labeller not running, or no group labelled. |
| `need exactly one agent with role 'robot'` | Press Ctrl+R on the robot hat in the labeller. |
| Executor `aborted: no fresh oriented robot pose` | Capture the robot hat (Ctrl+P). Its `src` must show `fit`. |
| `LLM made no usable calls` | The model doesn't do tool calling well. Replan or switch model. |
| `robot_bridge: [Errno 2]` | Bridge not running (step 2), or the socket is not at `/tmp/g1_robot_bridge.sock`. |
| Robot walks the wrong way | The captured heading is wrong. Recapture with the robot facing a known direction. |
| `ros2 topic hz/echo` shows nothing, but nodes run | Stale ros2 daemon: `ros2 daemon stop`, or add `--no-daemon`. |
| RViz shows cameras, but no LEDs or TF | The tracker only publishes LEDs that cameras see (cond > 0). Power the caps and bring them into view. |
