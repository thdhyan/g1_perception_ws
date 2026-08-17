# Data Collection — rosbag2

## Connect to the robot first — DDS unicast setup

The robot (`192.168.123.164`) runs ROS2 Foxy + `rmw_cyclonedds_cpp`,
`ROS_DOMAIN_ID=0`. Your laptop is on the same subnet (`192.168.123.222` via
`enp2s0`) but multicast DDS discovery may not cross the ethernet link.
A cyclonedds config at `~/.config/cyclonedds/config.xml` forces unicast to
the robot (already written — see `scripts/record_bag.sh` setup):

```bash
# One-time: already written by the setup step. Verify it exists:
cat ~/.config/cyclonedds/config.xml

# Export before every terminal where you want to see robot topics:
export CYCLONEDDS_URI=file://$HOME/.config/cyclonedds/config.xml
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/jazzy/setup.bash

ros2 topic list   # should now show /lowstate, /unitree/slam_mapping/points, etc.
```

Add those three exports to `~/.bashrc` or a `robot_env.sh` sourced per
session so you don't forget them.

Record LiDAR, camera (RGB/depth), IMU, and joint angles from either the real
Unitree G1 or the Isaac Sim scene, using `scripts/record_bag.sh`.

## Quick start

```bash
source /opt/ros/jazzy/setup.bash
cd g1_perception_ws
chmod +x scripts/record_bag.sh   # already executable if freshly checked out

# Real robot
./scripts/record_bag.sh --source real --out ~/g1_bags --name test_walk_01

# Isaac Sim
./scripts/record_bag.sh --source sim --out ~/g1_bags --name sim_test_01
```

Stop with `Ctrl+C`. The bag lands at `~/g1_bags/<name>/` (a directory: a
`metadata.yaml` + one or more `.mcap`/`.db3` files, per rosbag2's default
storage plugin).

## What gets recorded

### `--source real` (physical G1)

| Topic | Message type | Content |
|---|---|---|
| `/utlidar/cloud` | `sensor_msgs/PointCloud2` | Mid-360 LiDAR sweep |
| `/lowstate` | `unitree_hg/msg/LowState` | IMU (orientation, gyro, accel) **and** joint angles/velocities/torques, all in one message — this is Unitree's SDK convention, not `sensor_msgs/Imu` + `sensor_msgs/JointState` |
| `/camera/color/image_raw` | `sensor_msgs/Image` | D435 RGB, if the robot's camera driver is running |
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | D435 depth |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Intrinsics |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | Transform tree |

Source: [Unitree G1 LiDAR instructions](https://support.unitree.com/home/en/G1_developer/lidar_Instructions),
[ROS2 communication routine](https://support.unitree.com/home/en/G1_developer/ros2_communication_routine).
`/lowstate`'s exact field layout depends on `unitree_hg` (or `unitree_go` on
older firmware) message definitions — install the matching Unitree ROS2
package on any machine that needs to *decode* the bag, not just record it;
recording works with zero Unitree-specific packages installed since
`ros2 bag record` only needs topic names, not deserialized types.

If your G1's camera isn't wired through `realsense2_camera` (topic names
above), check `ros2 topic list` on the robot and pass the actual names via
`--extra /your/topic`.

If your G1 publishes raw Livox SDK output instead of `/utlidar/cloud`
(topic `/livox/lidar`, type `livox_ros_driver2/msg/CustomMsg`), record that
topic directly — `ros2 bag record` doesn't care that it isn't PointCloud2 as
long as the message type's `.msg` package is on the recording machine's
`AMENT_PREFIX_PATH` so it can look up the type to serialize.

### `--source sim` (Isaac Sim, `g1_rtx_sim.py`)

| Topic | Message type | Content |
|---|---|---|
| `/livox/mid360/points` | `sensor_msgs/PointCloud2` | RTX LiDAR Mid-360 sweep |
| `/g1/camera/rgb` | `sensor_msgs/Image` | D435 RGB |
| `/g1/camera/depth` | `sensor_msgs/Image` | D435 depth |
| `/g1/camera/semantic` | `sensor_msgs/Image` | Semantic segmentation |
| `/g1/camera/camera_info` | `sensor_msgs/CameraInfo` | Intrinsics |
| `/g1/joint_states` | `sensor_msgs/JointState` | Standard ROS2 joint state (no separate sim IMU topic exists yet — see Gaps below) |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | Transform tree, World-rooted |
| `/clock` | `rosgraph_msgs/Clock` | Sim time |

`--source sim` passes `--use-sim-time` to `ros2 bag record` so recorded
timestamps follow `/clock` rather than wall time — required for the bag to
replay consistently with `use_sim_time:=true` nodes later.

## Options

```
--source {real|sim}     required
--out DIR               output directory for the bag folder (default: ./bags)
--name NAME             bag folder name (default: g1_<source>_<timestamp>)
--duration SEC          split into new bag files every SEC seconds
--extra TOPIC           record an additional topic; repeatable
```

Example — add the robot's odometry and battery state on top of the defaults:

```bash
./scripts/record_bag.sh --source real --extra /odom --extra /sportmodestate
```

## Inspecting a bag afterward

```bash
ros2 bag info ~/g1_bags/test_walk_01
ros2 bag play ~/g1_bags/test_walk_01        # republishes on the original topics
```

To replay into RViz with the existing `g1_rtx.rviz` config, play the bag and
launch `robot_state_publisher` + RViz per `launch/g1_bringup.launch.py`
(sim-recorded bags only — real-robot bags need the real robot's URDF, not
the sim one, unless the joint names line up).

## Gaps / follow-ups

- **No standalone `/imu` topic in sim.** The real robot's IMU rides inside
  `/lowstate`; the sim currently has no equivalent published separately.
  If you need a directly comparable IMU stream from sim, add an
  `IsaacReadIMU` → `ROS2PublishImu` pair to `g1_sim/rtx_camera.py`'s
  action graph on the pelvis/torso IMU frame — not done yet.
- **`/lowstate` decode** requires the matching `unitree_hg`/`unitree_go`
  ROS2 message package on whatever machine parses the bag later. Recording
  doesn't need it; analysis does.
- **Real-robot topic names are per Unitree's docs, not yet confirmed against
  an actual running G1** — run `ros2 topic list` on the robot once available
  and adjust `scripts/record_bag.sh`'s `TOPICS` array (real branch) if names
  differ, e.g. across firmware versions.
