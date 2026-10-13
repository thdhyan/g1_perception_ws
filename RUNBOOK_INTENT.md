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
| 4 | Recording (optional, see section 3) | `./scripts/record_intent_session.sh --name take1` |

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
| `no fresh /mocap/agents` | Labeller not running, or no group labelled. |
| `need exactly one agent with role 'robot'` | Press Ctrl+R on the robot hat in the labeller. |
| Executor `aborted: no fresh oriented robot pose` | Capture the robot hat (Ctrl+P). Its `src` must show `fit`. |
| `LLM made no usable calls` | The model doesn't do tool calling well. Replan or switch model. |
| `robot_bridge: [Errno 2]` | Bridge not running (step 2), or the socket is not at `/tmp/g1_robot_bridge.sock`. |
| Robot walks the wrong way | The captured heading is wrong. Recapture with the robot facing a known direction. |
| `ros2 topic hz/echo` shows nothing, but nodes run | Stale ros2 daemon: `ros2 daemon stop`, or add `--no-daemon`. |
| RViz shows cameras, but no LEDs or TF | The tracker only publishes LEDs that cameras see (cond > 0). Power the caps and bring them into view. |
