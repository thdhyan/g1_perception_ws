# Runbook: Real G1 — sensors, live detection, teleop

Copy-paste commands for a live session on the real robot. Architecture and
rationale live in [REAL_ROBOT_WORKFLOW.md](REAL_ROBOT_WORKFLOW.md); this file is
just the sequence.

Every laptop terminal starts with:

```bash
cd /home/thakk100/Projects/thesis/g1_perception_ws && source setup_g1_env.sh
```

Use plain terminals, not the VSCode integrated one — its snap environment breaks
RViz.

## 0. Preflight

| | |
|---|---|
| Robot | `192.168.123.164`, user `unitree`, passwordless SSH, `sudo` needs a password |
| Laptop | `enp2s0` at `192.168.123.222/24` |
| DDS | `ROS_DOMAIN_ID=0`, `rmw_cyclonedds_cpp`, both set by `setup_g1_env.sh` |

```bash
ping -c 3 192.168.123.164
```

## 1. Robot sensors (from the laptop)

```bash
./scripts/start_robot_sensors.sh
```

Starts, in a tmux session `sensors` on the robot: `robot_state_publisher`,
`livox_ros_driver2`, `lowstate_to_jointstate`, `realsense2_camera`, and the
`d435_link → camera_link` static TF.

**All TF is computed on the robot.** The laptop publishes none — the robot is
Foxy and the laptop Jazzy, and Foxy cannot deserialise Jazzy's XCDR2, so a
laptop-side `robot_state_publisher` floods the robot with
`invalid data size ... serdata.cpp:308` and fights it for TF authority.

Options:

```bash
./scripts/start_robot_sensors.sh "src/g1_sensors.launch.py camera:=false"
./scripts/start_robot_sensors.sh "src/g1_sensors.launch.py pointcloud:=false"
./scripts/start_robot_sensors.sh "src/g1_sensors.launch.py camera_imu:=true"   # see Troubleshooting
```

Verify:

```bash
ros2 topic hz /livox/lidar          # 10 Hz
ros2 topic hz /joint_states         # ~750 Hz, real encoder angles
ros2 topic hz /camera/color/image_raw
ros2 run tf2_ros tf2_echo pelvis mid360_link
```

Attach to the robot's console:

```bash
ssh unitree@192.168.123.164 -t 'tmux attach -t sensors'
```

## 2. Locomotion bridge (the Unix socket)

`rclpy` and `unitree_sdk2py` segfault in one process, so the LocoClient lives in
a non-ROS bridge listening on `/tmp/g1_robot_bridge.sock`.

```bash
python3 src/g1_control/g1_control/robot_bridge.py enp2s0
```

Leave it running. Check it by hand — note the trailing newline, the protocol is
line-delimited JSON:

```bash
python3 -c "
import socket, json
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(3)
s.connect('/tmp/g1_robot_bridge.sock')
s.sendall((json.dumps({'cmd': 'get_fsm'}) + '\n').encode())
print(s.recv(4096).decode())"
```

`{"ok": true, "fsm_id": 501}` means it is connected to the robot.
`{"cmd": "get_arm_actions"}` lists every arm gesture the SDK accepts.

If a velocity command comes back `{"ok": false, "error": "unknown cmd:
set_velocity"}`, the running bridge predates that command — restart it.

The bridge serves one thread per connection. An older single-threaded build
blocks every command for the full duration of a gesture, then executes the
backlog of queued velocity commands against a stale world — which looks like
"the robot stops responding after an arm action". If you see
`BrokenPipeError: [Errno 32] Broken pipe` in the bridge, it is that build:
restart it.

## 3. Live 3D detection + approach

```bash
ros2 launch g1_bringup real_live_detection.launch.py \
  algorithm:=voxelnext \
  score_threshold:=0.2 \
  class_filter:=pedestrian \
  offset_ground:=-0.3 \
  accumulate_frames:=3 \
  max_distance:=15.0 \
  max_hz:=10.0 \
  rviz:=true
```

This runs `human_follow_and_greet_node` alongside the detector (`follow:=true` is
the default), which is what provides `/g1/approach_selected` — the service the
console's `Y` key calls. Without it the console reports
`no /g1/approach_selected (is human_follow_and_greet_node running?)`.

**The robot still does not move on its own.** `auto_execute` defaults to false,
so selecting a human only ARMS the approach and the node waits for `Y`.

`follow:=false` drops the approach node entirely, leaving a perception-only stack
that cannot move the robot.

`offset_ground` is **not** the sensor height: it is whatever puts the ground
where the nuScenes-trained model expects it. Measured on the real robot, with
the ground at `z = -1.27` in `mid360_link`, a person at 4.3 m scores 0.28 at
`-0.3`, 0.24 at `-0.45`, 0.22 at `-0.6`, and 0.14 at the old sim default of
`1.33`.

Detections publish on `/g1/detections/livox`; RViz uses
`src/livox_detection/config/livox_human_viz.rviz`.

To run RViz separately (`rviz:=false` above):

```bash
rviz2 -d src/livox_detection/config/livox_human_viz.rviz
```

Other backends:

```bash
ros2 launch g1_bringup real_live_detection.launch.py algorithm:=pointpillar \
  checkpoint_path:=/home/thakk100/Projects/Thesis/livox_detection/pt/livox_model_1.pt
```

## 4. Human sorting

```bash
ros2 run livox_detection human_distance_sorter_node --ros-args \
  -p input_topic:=/g1/detections/livox -p output_topic:=/g1/sorted_humans -p min_score:=0.1
```

Sorts detections by distance onto `/g1/sorted_humans`, which is what the console
below selects from.

## 5. Keyboard console — walking, selection, gestures

One terminal for all three. Needs the bridge from step 2. **The robot walks.**

```bash
ros2 run livox_detection human_keyboard_selector_node --ros-args \
  -p input_topic:=/g1/sorted_humans \
  -p linear_speed:=0.3 -p yaw_rate:=0.5
```

| Keys | |
|---|---|
| `W` `A` `S` `D` | walk forward / strafe left / back / strafe right |
| `Q` `E` | turn left / right |
| `SPACE` | stop |
| `Z` `X` | linear speed −/+ 0.05 m/s |
| `-` `+` | yaw rate −/+ 0.10 rad/s |
| `1`–`9` | lock onto human #1..#9 |
| `0` / `C` | clear selection |
| `R` | rescan: fresh cloud + re-run detection (halts the robot first) |
| `T` | same as `R`, via `/g1/trigger_snapshot` |
| `Y` | **YES — confirm the approach** and walk to the selected human |
| `H` | greet where the robot stands, without approaching |
| `V` `B` `N` `M` `,` `.` | shake hand, high wave, low wave, high five, heart, hands up |
| `ESC` | quit (halts the robot on the way out) |

Motion is hold-to-move: velocity decays to zero `key_hold_timeout` (0.5 s) after
the last keypress, so letting go stops the robot rather than leaving it walking.

**Walking to a detection needs two keys.** With `auto_execute:=false`, selecting
`1`-`9` only ARMS an approach; the robot moves when you press `Y`
(`/g1/approach_selected`). A detection that jumps between frames therefore cannot
send the robot walking on its own. Both scan keys also halt the robot and wait
`settle_time` before collecting, so no cloud is ever accumulated while walking.

The console talks to `robot_bridge`'s socket directly — `cmd_vel_bridge` is only
needed when something else (Nav2, a follow node) drives via `/cmd_vel`:

```bash
ros2 run g1_control cmd_vel_bridge
```

`g1_nav keyboard_teleop` still exists and publishes `/g1/cmd_vel`; it needs
`cmd_vel_bridge` and does not do selection or gestures.

## 5b. Snapshot pipeline and walk-up-and-greet

The 2-pass snapshot pipeline (accumulate a dense cloud, then detect once) with
the follow-and-greet controller:

```bash
ros2 launch g1_bringup real_human_follow.launch.py \
  algorithm:=voxelnext \
  offset_ground:=-0.3 \
  score_threshold:=0.2 \
  standoff_distance:=0.80 \
  greeting_action:=shake_hand \
  auto_execute:=false
```

`auto_execute:=false` keeps the robot still until you press `Y` — start there.

Approach behaviour is set on this launch, not in the console:
`standoff_distance` (default **1.50 m**) is where the robot stops in front of the
human, and `linear_speed` (default **0.30 m/s**) is the walk-up speed. Both are
forwarded to `human_follow_and_greet_node`, whose own bare defaults (0.60 m,
0.50 m/s) only apply if you run that node with `ros2 run`.

`greeting_action` (default `low_wave`) accepts `shake_hand`, `low_wave`,
`high_wave`, `wave_and_shake`, `none`, or any bridge action name.

## 6. Recording

```bash
./scripts/record_bag.sh --source real --name lab_walk_01
```

Records `/livox/lidar`, `/livox/imu`, the camera streams, `/joint_states`,
Unitree `/lowstate`, and TF. The RGB depth cloud dominates the file size — drop
it with `pointcloud:=false` in step 1 for long runs.

## Troubleshooting

**Clock skew.** The robot's chrony has no reachable upstream and drifts minutes
to hours away from the laptop, which surfaces as `TF_OLD_DATA ignoring data from
the past`. The robot is `Asia/Shanghai` and the laptop is US-local, so set UTC
explicitly or the fix lands 13 hours off:

```bash
ssh -t unitree@192.168.123.164 "sudo date -u -s \"$(date -u '+%F %T')\""
```

**`/camera/imu` never publishes**, and the robot logs
`HID set_power 1 failed` plus `iio_hid_sensor: Frames didn't arrived` every 5 s.
The D435i's IMU goes through the kernel HID/iio path here and fails, which is
why `camera_imu` defaults to false. Root fix, needs sudo:

```bash
sudo tee /etc/modprobe.d/blacklist-realsense-hid.conf <<'EOF'
blacklist hid_sensor_accel_3d
blacklist hid_sensor_gyro_3d
blacklist hid_sensor_trigger
blacklist hid_sensor_iio_common
EOF
sudo rmmod hid_sensor_accel_3d hid_sensor_gyro_3d hid_sensor_trigger
```

Then relaunch with `camera_imu:=true`. The Mid-360's IMU (`/livox/imu`, 200 Hz)
is unaffected.

**`Backend init warning: Cannot import pcdet ... Using PointPillar clustering
fallback`.** VoxelNeXt was not found and the node silently degraded — a
detection run that looks alive but is not running the model you asked for. Check
the resolved path:

```bash
ros2 launch g1_bringup real_live_detection.launch.py --show-args | grep -A2 voxelnext_dir
```

It must be `<workspace>/VoxelNeXt`, not `~/Projects/thesis/VoxelNeXt`.

**Detections stay at 0 with plausible point counts.** Check in this order:
`class_filter` (`pedestrian` discards every other nuScenes class — and a literal
`all` is treated as a class name, matching nothing), then `offset_ground`, then
whether anyone is actually inside `max_distance`.

**Topics visible but no data.** `ROS_DOMAIN_ID` or `RMW_IMPLEMENTATION`
mismatch. Both come from `setup_g1_env.sh`; DDS discovery fails silently, not
with an error.
