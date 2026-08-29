# HANDOFF: Real G1 — sensors, live detection, teleop, and voice next

**Date:** 2026-08-20. **Repo:** `g1_perception_ws` on `main`, HEAD `1ed7967`.
**Status:** the real-robot perception + teleop loop works end to end. VoxelNeXt
detects people on live Mid-360 data, the operator selects one from a keyboard
console, confirms the approach, and the robot walks to a standoff and greets.
The next workstream is **voice: onboard ASR/TTS through ROS**, planned at the
end of this document.

Operating commands live in [RUNBOOK_REAL_ROBOT.md](RUNBOOK_REAL_ROBOT.md).
This document is what changed, what was wrong, and what to do next.

Other active workstreams, unchanged by this session:
[HANDOFF.md](HANDOFF.md) (sim / laptop Isaac ROS) and
[HANDOFF_ISAAC_ROS_ROBOT.md](HANDOFF_ISAAC_ROS_ROBOT.md) (Isaac ROS on the
robot's Orin NX).

## What the session started from

A laptop-side stack that ran detection against the real robot but produced
nothing useful: zero detections for hundreds of frames, TF that did not follow
the robot, and a robot-side log flooded with rmw errors.

## Root causes found (all fixed)

**TF was fake.** The robot's `lowstate_to_jointstate_node` was never started —
the robot's copy of `g1_sensors.launch.py` defaulted `joint_states` to false,
while the repo copy said true. The only `/joint_states` on the wire came from a
**laptop** `joint_state_publisher` publishing the URDF **zero pose** at 30 Hz.
Every detection was transformed into a pelvis unrelated to the robot's posture.

**The Foxy/Jazzy gap is one-directional, and the repo assumed the opposite.**
The robot is Foxy, the laptop Jazzy. Verified empirically: Jazzy reads Foxy fine
(`/livox/lidar` at 10 Hz, `/robot_description` echoes cleanly, `/tf` resolves),
but Foxy cannot deserialise Jazzy's XCDR2. So every `serdata.cpp:308`
(`invalid data size`, `string data is not null-terminated`) flood on the robot
came from **laptop publishers**, not the robot. Comments in the launch files
claiming TF could not cross the gap were wrong in the direction that mattered.

**VoxelNeXt was mis-tuned, not broken.** `offset_ground` is not the sensor
height — it is whatever puts the ground where the nuScenes-trained model expects
it. Measured offline against a captured 10-sweep cloud, with the ground at
`z = -1.27` in `mid360_link` and a real person at 4.3 m:

| `offset_ground` | pedestrian score |
|---|---|
| 1.33 (inherited from sim) | 0.141 |
| 0.00 | not in top 8 |
| **-0.30** | **0.282** |
| -0.45 | 0.243 |
| -0.60 | 0.221 |

**The workspace root was computed as `parents[4]`,** which under
`--symlink-install` resolves through `src/` and lands on `~/Projects/thesis`.
`pcdet` then failed to import and the node fell back to a generic clustering
heuristic with only a warning — a run that looks alive but is not running the
model asked for. Both real-robot launches now walk up to the directory containing
`VoxelNeXt`.

**`cmd_vel_bridge` could never have worked.** It sent
`{"command": "move", vx, vy, wz}`; `robot_bridge` keys on `"cmd"` and reads
`dx/dy/speed`. Every message returned `unknown cmd: None`.

**The bridge had no streaming velocity command at all.** `move`/`rotate` are
blocking distance/angle goals that hold the dispatch lock for the entire
traversal.

**The bridge was single-threaded.** A 3.5 s gesture blocked every command for
its duration; a teleop client opening a connection per tick piled up behind it
and then executed that backlog against a stale world — which presented as "after
arm control, cannot control the robot". The reply write was also unguarded, so a
client that timed out and hung up took the handler down with a `BrokenPipeError`
*after* the command had already run.

**Stop was not a stop.** It queued behind the `move` it was meant to interrupt
(dispatch lock), could not be typed during a gesture (the console slept on the
key loop), and did not cancel an approach — the robot halted mid-walk and then
carried on into arrival and greeting as if it had arrived.

**The snapshot pipeline could not start.** `install/`'s `entry_points.txt`
predated the `setup.py` entries for `livox_snapshot_pipeline_node`,
`human_loco_approach_node` and `livox_front_filter_node`. The console script
raised `StopIteration` and the launch reported only `process has died`.

## What the stack looks like now

```
ROBOT (Foxy, Orin NX)                          LAPTOP (Jazzy)
  robot_state_publisher  ──/tf, /tf_static──►  livox_detection_node (VoxelNeXt)
  lowstate_to_jointstate ──/joint_states────►        │ /g1/detections/livox
  livox_ros_driver2      ──/livox/lidar,imu─►        ▼
  realsense2_camera      ──/camera/*────────►  human_distance_sorter_node
  tf d435_link→camera_link                           │ /g1/sorted_humans
                                                     ▼
                                            human_keyboard_selector_node
                                                     │ socket        │ /g1/select_human_id
                                                     ▼               ▼
                                            robot_bridge.py ◄── human_follow_and_greet_node
```

**All TF is computed on the robot.** The laptop publishes none. Everything
laptop-side that used to publish TF is now behind `publish_tf`, default false.

Robot-side `g1_sensors.launch.py` now also brings up the D435i: colour, depth,
aligned depth, an **RGB-textured** point cloud (`pointcloud.stream_filter: 2` —
without it the cloud has only x/y/z and renders grey), and the
`d435_link → camera_link` static TF that the RealSense tree was missing. IR
streams are off. The camera IMU is off: its kernel HID path fails on this Jetson
(`HID set_power 1 failed`, `iio_hid_sensor: Frames didn't arrived`), and the
blacklist workaround is recorded in the launch file. `/livox/imu` at 200 Hz is
unaffected.

## Safety model (deliberate, do not weaken casually)

1. **Selection never moves the robot.** With `auto_execute:=false`, selecting a
   human only ARMS the plan; the node holds it and waits for
   `/g1/approach_selected` — `[Y]` in the console. This exists because
   detections still jump between frames.
2. **No cloud is collected while walking.** `[R]`/`[T]` halt the robot and wait
   `settle_time` first: sweeps accumulated in motion smear across poses and
   every box built from them is misplaced.
3. **`[SPACE]` preempts everything** — zeroes teleop velocity, sends the
   bridge's `stop` (which bypasses the dispatch lock and raises an abort flag
   the traversal loops poll), calls `/g1/abort_approach`, and cuts a gesture's
   hold short. Gestures run on their own thread precisely so the key loop can
   still read it.
4. **Teleop is hold-to-move.** Velocity decays 0.5 s after the last keypress.

## Commits

`2e8ec3f` robot-side TF and sensors, VoxelNeXt tuning · `2a430de` non-blocking
velocity + unified keyboard console · `a8aa31b` stop before every cloud
collection · `22ff1c1` gate laptop TF behind `publish_tf` · `72ca07f`
operator-confirmed approach · `6a5a623` approach node on by default ·
`a5ed58c` threaded bridge, broken-pipe guard · `96e2213` standoff 1.5 m, speed
0.3 m/s, low-wave greeting · `1ed7967` real emergency stop.

## Open items

- **Clock skew.** The robot's chrony has no reachable upstream. It drifted
  156 s during the session, and a naive `date -s` left it **13 hours** off
  because the robot is `Asia/Shanghai` and the laptop is US-local. Fix with
  `sudo date -u -s "$(date -u '+%F %T')"`. Symptom is `TF_OLD_DATA`. A durable
  fix (laptop as the subnet's NTP server) is not done.
- **The robot's `g1_sensors.launch.py` is deployed by `scp`, not tracked from
  the robot side.** Redeploy after editing the repo copy, or the robot silently
  runs the old file — that is how `joint_states:=false` survived.
- **VoxelNeXt's ceiling is the domain gap.** nuScenes-trained on a 32-beam
  spinning lidar at 1.84 m; the Mid-360 has a different scan pattern, height and
  intensity scale. Best pedestrian score on a real person is ~0.28, with
  "barrier" false positives scoring higher. Tuning is exhausted; fine-tuning on
  Mid-360 data is the real fix.
- **`livox_detection` needs a plain `colcon build`.** With `--symlink-install`
  the egg-link cannot be resolved by `importlib.metadata`
  (`PackageNotFoundError`).
- **`src/g1_voice/` is still uncommitted**, as are the Isaac ROS robot docs.
- **`real_human_follow.launch.py` still runs the greeting on arrival by
  default** (`auto_greet:=true`); only the walk is gated.

---

# Next step: voice client and TTS over ROS

`src/g1_voice/` exists, builds, and its dialog loop was verified against the lab
Ollama server in an earlier session — but it has **not been run against this
stack**, and never together with the perception/teleop nodes. Read
[`src/g1_voice/README.md`](src/g1_voice/README.md) first; it has the hardware
facts (16 kHz mono s16le PCM, `speaker_id` 0 = Chinese / 1 = English, no mixed
Chinese/English in one call, LED calls spaced > 200 ms).

Architecture mirrors `robot_bridge.py` for the same reason — `rclpy` and
`unitree_sdk2py` segfault in one process — so `audio_backend.py` is a plain
process on `/tmp/g1_audio_backend.sock` and the ROS nodes talk to it.

## Preconditions

- **The robot must be in wake-up mode** (from the app or the remote), or the
  microphone produces no ASR at all. This is the single most likely reason for
  a silent `/g1/asr/text`.
- `Vui_Service >= 2.0.3.8`, `Vui Module >= 2.0.0.3`.
- Robot and laptop clocks in sync — same skew problem as above.

## Phase 0 — build and reach the service

```bash
colcon build --packages-select g1_voice && source install/setup.bash
python3 src/g1_voice/g1_voice/audio_backend.py enp2s0
```

Verify: the backend prints a DDS connection and creates the socket. **Check
whether it coexists with `robot_bridge.py`** — both open their own DDS
participant in separate processes, which should be fine, but this is the first
time they run together, so start the voice backend with the loco bridge already
up and watch for a segfault in either. If they clash, that is a finding worth
writing down immediately, not working around.

Success: `ls -l /tmp/g1_audio_backend.sock`, no traceback in either process.

## Phase 1 — TTS out, no ROS, no LLM

Speak through the backend socket directly, bypassing ROS, so a failure is
unambiguously SDK-side:

```bash
python3 -c "
import socket, json
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(10)
s.connect('/tmp/g1_audio_backend.sock')
s.sendall((json.dumps({'cmd': 'tts', 'text': 'Hello, I am G1.', 'speaker_id': 1}) + '\n').encode())
print(s.recv(4096))"
```

The command schema is verified against `audio_backend.py`'s dispatch: `tts`,
`get_volume`, `set_volume`, `led`, `play_pcm` (base64 PCM plus a `stream_id`),
all line-delimited JSON like the loco bridge. Success: audible speech from the
robot's speaker.

## Phase 2 — TTS out, through ROS

```bash
ros2 launch g1_voice voice.launch.py use_dialog:=false
ros2 topic pub --once /g1/audio/volume std_msgs/Int32 "{data: 80}"
ros2 topic pub --once /g1/tts/text std_msgs/String "{data: 'Hello, I am G1.'}"
ros2 topic echo /g1/audio/play_state
```

Success: speech, and `/g1/audio/play_state` goes true then false. Then test the
one reliable interrupt:

```bash
ros2 topic pub --once /g1/tts/text std_msgs/String "{data: '<a long paragraph>'}"
ros2 topic pub --once /g1/audio/stop std_msgs/Empty "{}"
```

Success: playback stops **and** the queued remainder is dropped, not resumed.

## Phase 3 — ASR in

Put the robot in wake-up mode, then:

```bash
ros2 topic echo /g1/asr/text
ros2 topic echo /g1/asr/event      # full JSON: angle, speaker_id, confidence
```

Speak. Success: the utterance appears within a second or two. Record the
`angle` field for a few known speaker positions — it is a free bearing estimate,
and worth checking against the LiDAR detection bearing for the same person,
which is the obvious fusion later.

If nothing arrives: wake-up mode first, then whether `rt/audio_msg` is on the
wire at all (`ros2 topic list` will not show it — it is a raw DDS topic the
backend subscribes to, so check the backend's own log).

## Phase 4 — full dialog loop

```bash
ros2 launch g1_voice voice.launch.py
```

The `llm_chain` tries onboard Ollama, then the lab server, then OpenAI. Two
things to measure rather than assume:

- **Latency per backend.** The README records `gemma4:latest` on the lab server
  at 20-33 s for three sentences, which is unusable conversationally. Time the
  onboard `gemma4:e2b` and a 3-4B model on the lab box, and pick on measured
  numbers.
- **Whether the onboard model is even installed.** `gemma4:e2b` may not exist as
  a tag; the README says to verify and fall back.

Success criterion for the phase: speak a question, get a spoken answer, with
end-to-end latency written down.

## Phase 5 — voice alongside perception

Run the voice stack and the detection/teleop stack together and check:

- neither DDS participant destabilises the other;
- laptop GPU headroom with VoxelNeXt loaded (RTX 4060, 8 GB) — voice is CPU/net
  bound on the laptop, so this should be fine, but measure with `nvidia-smi`;
- the robot's Jetson budget if the onboard LLM is used: the README's table puts
  Ubuntu + Unitree services + drivers at 3-4 GB and `gemma4:e2b` at ~3 GB, which
  fits in 16 GB — but Isaac ROS on the robot (the other workstream) does not fit
  alongside it. Decide which one owns the Jetson before combining.

## Phase 6 — the actual integration worth building

`dialog_node` calls no robot behaviour today. The obvious first tool call is
into `g1_control/robot_bridge.py`: "wave at me", "come here", "stop" mapped onto
the same socket commands the keyboard console already uses.

**Keep the safety model.** A spoken "come here" must arm an approach, not start
one — the same `[Y]` confirmation the console requires, or an explicit spoken
confirmation. Voice is a less reliable trigger than a keypress, not a more
reliable one, and `[SPACE]` must remain the override. `/g1/audio/stop` is the
voice equivalent for speech, and the two should be wired to the same abort.

## Custom TTS, later

Kokoro/Piper on the **laptop** (the robot is Python 3.8 / CUDA 11.4; the laptop
is 3.12 with an RTX 4060), resampled to 16 kHz s16le and pushed to
`/g1/audio/play_pcm`. It replaces consumption of `/g1/tts/text` only; every
other topic is unchanged. Worth doing only if the onboard voice proves to be the
limiting factor in the demo.
