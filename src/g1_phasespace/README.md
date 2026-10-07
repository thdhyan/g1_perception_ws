# g1_phasespace

PhaseSpace hats for the G1 experiments. Uses `phasespace_ros2` from the `src/Phasespace` submodule (https://github.com/thdhyan/Phasespace), which publishes LEDs, per-LED and per-group TF under `phasespace` (origin = floor-tape alignment origin, Z up, m).

| Node | Does |
|---|---|
| `humans` | every LED group except `robot_group` is a person (group TF = head centre): SMPL-X neutral body on `/phasespace/bodies` (latched, frame-locked), `/phasespace/humans` JSON in G1_sim's `/sim/humans` schema `{"frame", "humans": [{"name", "x", "y", "z", "heading_deg"}]}` (`heading_deg` null until the group has an orientation reference) |
| `robot_head` | `robot_group` (default `dhyan-hat-3`) is on the G1 head: static TF `<robot_group> → ps_robot_head` (param `offset`, m, hat frame), `/phasespace/robot_head` PoseStamped, head box + facing arrow on `/phasespace/robot` |

Current `Dhyan Hat` profile: `dhyan-hat` (LEDs 0–7) and `dhyan-hat-2` (8–15) are humans, `dhyan-hat-3` (16–23) is the robot head.

## Run

```bash
git submodule update --init src/Phasespace
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to g1_phasespace
source install/setup.bash
ros2 launch g1_phasespace phasespace.launch.py                 # tracker + humans + robot head + RViz
ros2 launch g1_phasespace phasespace.launch.py robot_group:=dhyan-hat-3 rviz:=false
```

The SMPL-X model is licensed and read from `smplx_model` (default `~/Projects/HOVER/body_models/smplx/SMPLX_NEUTRAL.npz`); it is never copied into the workspace.

G1 planner on the real robot: `-r /sim/humans:=/phasespace/humans`. Positions are in `phasespace`, not the robot's `World`/`map`; `ps_robot_head` gives the robot's head in the same frame.
