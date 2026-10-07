# PhaseSpace in g1_perception_ws: how to operate

Motion capture ground truth for the G1 experiments: three LED hats tracked by the PhaseSpace system in the Drone Lab (Shepherd Labs). Two hats are people (shown as SMPL-X bodies), one hat sits on the G1 head.

| Hat (microdriver) | LED IDs | Role | ROS output |
|---|---|---|---|
| `dhyan-hat` | 0–7 | human | TF `dhyan-hat`, SMPL-X body, entry in `/phasespace/humans` |
| `dhyan-hat-2` | 8–15 | human | TF `dhyan-hat-2`, SMPL-X body, entry in `/phasespace/humans` |
| `dhyan-hat-3` | 16–23 | G1 head centre | TF `dhyan-hat-3` → `ps_robot_head`, `/phasespace/robot_head` |

The groups come from the PhaseSpace session profile `Dhyan Hat` (the server default). Each microdriver in the profile is one group, named after the device.

Packages:
- `src/Phasespace` (submodule, https://github.com/thdhyan/Phasespace) → `phasespace_ros2`: connects to the server and publishes the LEDs, TF and cameras
- `src/g1_phasespace` → `humans` and `robot_head` nodes, launch file, RViz config

## 1. Before the session

1. Charge the three hats and switch them on. Each hat's microdriver must be powered for its LEDs to show up.
2. Power on the PhaseSpace hub (the short black box near the lab entrance; the power button is on the front, on the left).
3. Put the laptop on the campus network (eduroam) and check the server answers:
   ```bash
   ping -c2 cs-phasespace.cs.umn.edu
   ```
4. Check the session profile in the Configuration Manager, http://cs-phasespace.cs.umn.edu (`admin` / `phasespace`), under Session Profiles. `Dhyan Hat` should be the default and should list `dhyan-hat`, `dhyan-hat-2` and `dhyan-hat-3` with LED IDs 0–7, 8–15 and 16–23. If you renumber or add a hat there, the ROS side picks it up on the next launch; nothing needs rebuilding.

## 2. Build (once, and after pulling changes)

```bash
cd ~/Projects/thesis/g1_perception_ws
git submodule update --init src/Phasespace
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to g1_phasespace
source install/setup.bash
```

## 3. Run

```bash
ros2 launch g1_phasespace phasespace.launch.py
```

This starts the tracker, the `humans` and `robot_head` nodes, and RViz. Launch arguments:

| Argument | Default | Use |
|---|---|---|
| `device` | `cs-phasespace.cs.umn.edu` | PhaseSpace server |
| `profile` | empty (server default) | another session profile, e.g. `profile:="Dhyan Hat"` |
| `robot_group` | `dhyan-hat-3` | which hat is on the robot |
| `smplx_model` | `~/Projects/HOVER/body_models/smplx/SMPLX_NEUTRAL.npz` | licensed SMPL-X model, never copied into the repo |
| `rviz` | `true` | `rviz:=false` on the robot or headless |

Tracker and RViz only, with no humans or robot nodes: `ros2 launch phasespace_ros2 phasespace.launch.py`.

In RViz (fixed frame `phasespace`, origin = alignment origin on the floor tape, Z up, metres) you should see the floor, the camera cones, one coloured sphere per visible LED, a SMPL-X body under each human hat, and a red box with an arrow at the robot head.

## 4. Check it is working

```bash
ros2 topic echo /phasespace/groups --once --qos-durability transient_local --field data   # the 3 hats and their LED IDs
ros2 topic echo /phasespace/humans --once --qos-durability transient_local --field data   # dhyan-hat, dhyan-hat-2
ros2 topic echo /phasespace/robot_head --once                                             # G1 head pose
ros2 run tf2_ros tf2_echo phasespace dhyan-hat-3/led_16                                   # a single LED
ros2 topic hz /phasespace_leds                                                            # ~60 Hz
```

A hat shows up only while at least one of its LEDs is visible. A hat that has been out of view for more than 0.5 s is dropped from `/phasespace/humans` and `/phasespace/robot_head`.

## 5. Topics and frames

| Name | Type | Content |
|---|---|---|
| TF `phasespace → <hat>` | tf | hat pose: LED centroid, rotated once a reference is captured (step 6) |
| TF `phasespace → <hat>/led_<id>` | tf | every visible LED of a hat (position only) |
| TF `phasespace → ps_cam_<id>` | tf_static | calibrated cameras |
| TF `dhyan-hat-3 → ps_robot_head` | tf_static | hat to head centre (`offset` parameter) |
| `/phasespace/groups` | String JSON, latched | `{"frame", "groups": {name: {"leds", "oriented"}}}` |
| `/phasespace/humans` | String JSON, latched, 10 Hz | G1_sim `/sim/humans` schema: `{"frame", "humans": [{"name", "x", "y", "z", "heading_deg"}]}` |
| `/phasespace/robot_head` | PoseStamped, 30 Hz | `ps_robot_head` in `phasespace` |
| `/phasespace_leds`, `/phasespace_world`, `/phasespace/bodies`, `/phasespace/robot` | MarkerArray | RViz |

## 6. Hat orientation (heading)

Without a reference, a hat's TF is the LED centroid with no rotation: the SMPL-X body always faces +x and `heading_deg` is `null`. To get heading:

1. Put the hat on (or on the robot), facing the PhaseSpace +x axis, upright, and stand still.
2. Capture its LED layout into the source tree, then rebuild:
   ```bash
   ros2 run phasespace_ros2 tracker --capture dhyan-hat \
       --ref-dir ~/Projects/thesis/g1_perception_ws/src/Phasespace/phasespace_ros2/trackers
   colcon build --symlink-install --packages-select phasespace_ros2
   ```
3. Repeat for `dhyan-hat-2` and `dhyan-hat-3`. Commit the `trackers/*.json` files in the Phasespace repo and bump the submodule.

At runtime each hat is fitted to its reference (Kabsch, needs at least 3 visible LEDs). If the hat is re-glued or LEDs move, capture again.

## 7. Robot head offset

`ps_robot_head` equals the hat's LED centroid by default. If the hat sits above the head centre, set the offset in the hat frame (metres):

```bash
ros2 run g1_phasespace robot_head --ros-args -p offset:="[0.0, 0.0, -0.08]"
```

Or add it to the `robot_head` node's parameters in `src/g1_phasespace/launch/phasespace.launch.py`. The offset follows the hat's rotation only after `dhyan-hat-3` has a reference (step 6).

## 8. Use with the G1 planner

The planner reads humans from `/sim/humans`; on the real robot remap it:

```bash
ros2 run ... --ros-args -r /sim/humans:=/phasespace/humans
```

Positions are in `phasespace`, not the robot's `map`/`World`. `ps_robot_head` gives the robot in the same frame. Linking it to the G1's `head_link` (or `map`) still needs a transform; that is not done yet.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OpenError: connection timed out` | server unreachable: hub off, or not on the campus network. `ping cs-phasespace.cs.umn.edu` |
| "1 uncalibrated cameras detected: 12" in the log | expected; camera 12 needs recalibration. Tracking continues with the other cameras |
| `/phasespace/humans` is empty | hats off or out of view; check LEDs in RViz and `/phasespace/groups` |
| LED spheres are grey | that LED is in no group of the active profile; check the profile (step 1.4) |
| no SMPL-X bodies but humans are listed | SMPL-X model missing at `smplx_model`; the log says "No SMPL-X model" |
| nothing on another machine | same `ROS_DOMAIN_ID` on both, and the same network |

## 10. Docker (optional)

The Phasespace repo has a Jazzy image for the tracker and RViz only, without SMPL-X or the robot head:

```bash
cd ~/Projects/thesis/g1_perception_ws/src/Phasespace
xhost +local:docker
docker compose -f docker/docker-compose.yml up --build
```

Use the native build above for the full G1 setup.
