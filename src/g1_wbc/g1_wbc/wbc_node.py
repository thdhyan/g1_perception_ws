#!/usr/bin/env python3
"""Sim-agnostic GR00T Whole-Body Control node for Unitree G1.

Subscribes
----------
/joint_states          sensor_msgs/JointState   — from Gazebo bridge, Isaac Sim, or real robot
/imu/data              sensor_msgs/Imu          — pelvis IMU (body-frame ang vel + orientation)
/g1/cmd_vel, /cmd_vel  geometry_msgs/Twist      — continuous velocity command (vx, vy, wz) with watchdog timeout
/g1/cmd_pose           geometry_msgs/Twist      — relative pose delta move (dx, dy, dyaw) like LocoClient.move()
/g1/emergency_stop     std_msgs/Bool            — emergency stop / damping mode
/g1/base_height        std_msgs/Float64         — stand height command (default 0.74m)
/g1/stop               std_msgs/Empty           — instant stop & balance

Publishes
---------
/g1/joint/<joint_name>  std_msgs/Float64   — per-joint position target (29 joints:
                                             15 leg+waist from policy, 14 arms held at 0)
/g1/wbc_status          std_msgs/String    — policy mode: "balance", "walk", "pose_move", "damp"
/g1/fsm_id              std_msgs/Int32     — LocoClient-compatible FSM state (4=Stand, 501=Walk, 1=Damp)

Deadman Watchdog & Wandering Prevention:
----------------------------------------
To ensure the robot ONLY moves when actively commanded and never wanders or drifts:
1. Every velocity command (/cmd_vel or /g1/cmd_vel) has a deadman timeout (default 0.5s).
   If no new command is received within cmd_vel_timeout, velocity is zeroed immediately.
2. When velocity is within deadband (<= 0.05 m/s), the node executes stationary balance.
"""

from __future__ import annotations

import collections
import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Empty, Float64, Int32, String

# ─────────────────────────────────────────────────────────────────────────────
# Policy constants (from G1_sim/g1_sim/wbc_bridge.py)
# ─────────────────────────────────────────────────────────────────────────────

LEG_WAIST_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

ALL_JOINTS = LEG_WAIST_JOINTS + ARM_JOINTS  # 29

NUM_ACTIONS = len(LEG_WAIST_JOINTS)   # 15
NUM_JOINTS = len(ALL_JOINTS)           # 29
SINGLE_OBS_DIM = 86
OBS_HISTORY_LEN = 6
NUM_OBS = SINGLE_OBS_DIM * OBS_HISTORY_LEN  # 516

DEFAULT_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     0.0, 0.0, 0.0],
    dtype=np.float32,
)
_PADDED_DEFAULT = np.concatenate(
    [DEFAULT_ANGLES, np.zeros(NUM_JOINTS - NUM_ACTIONS, dtype=np.float32)]
)

CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
ACTION_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ANG_VEL_SCALE = 0.5
DEFAULT_HEIGHT_CMD = 0.74
WALK_CMD_DEADBAND = 0.05

FSM_DAMP = 1
FSM_READY_STAND = 4
FSM_WALK = 501


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_rotate_inverse(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector v into the body frame."""
    w, x, y, z = quat_wxyz
    qc = np.array([w, -x, -y, -z], dtype=np.float64)
    return np.array([
        v[0] * (qc[0]**2 + qc[1]**2 - qc[2]**2 - qc[3]**2)
        + v[1] * 2*(qc[1]*qc[2] - qc[0]*qc[3])
        + v[2] * 2*(qc[1]*qc[3] + qc[0]*qc[2]),
        v[0] * 2*(qc[1]*qc[2] + qc[0]*qc[3])
        + v[1] * (qc[0]**2 - qc[1]**2 + qc[2]**2 - qc[3]**2)
        + v[2] * 2*(qc[2]*qc[3] - qc[0]*qc[1]),
        v[0] * 2*(qc[1]*qc[3] - qc[0]*qc[2])
        + v[1] * 2*(qc[2]*qc[3] + qc[0]*qc[1])
        + v[2] * (qc[0]**2 - qc[1]**2 - qc[2]**2 + qc[3]**2),
    ])


def gravity_orientation(quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0]))


# ─────────────────────────────────────────────────────────────────────────────
# ONNX policy runner
# ─────────────────────────────────────────────────────────────────────────────

def _load_policy(path: str):
    """Load an ONNX policy; returns callable(obs_flat) -> action."""
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(path, providers=providers)
    in_name = sess.get_inputs()[0].name

    def run(obs: np.ndarray) -> np.ndarray:
        return sess.run(None, {in_name: obs[None, :].astype(np.float32)})[0].squeeze(0)

    return run


# ─────────────────────────────────────────────────────────────────────────────
# WBC Node
# ─────────────────────────────────────────────────────────────────────────────

class WbcNode(Node):
    def __init__(self):
        super().__init__("g1_wbc_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("balance_policy_path", "")
        self.declare_parameter("walk_policy_path", "")
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("joint_topic_prefix", "/g1/joint")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("cmd_vel_topic", "/g1/cmd_vel")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("cmd_vel_timeout", 0.5)  # Deadman watchdog timeout
        self.declare_parameter("default_move_speed", 0.25)
        self.declare_parameter("default_yaw_rate", 0.5)

        balance_path = self.get_parameter("balance_policy_path").value
        walk_path = self.get_parameter("walk_policy_path").value
        control_hz = self.get_parameter("control_hz").value
        prefix = self.get_parameter("joint_topic_prefix").value.rstrip("/")
        imu_topic = self.get_parameter("imu_topic").value
        cmdvel_topic = self.get_parameter("cmd_vel_topic").value
        js_topic = self.get_parameter("joint_states_topic").value
        self.cmd_vel_timeout = float(self.get_parameter("cmd_vel_timeout").value)
        self.default_move_speed = float(self.get_parameter("default_move_speed").value)
        self.default_yaw_rate = float(self.get_parameter("default_yaw_rate").value)

        # Resolve policy paths
        balance_path = self._resolve_policy(balance_path, "GR00T-WholeBodyControl-Balance.onnx")
        walk_path = self._resolve_policy(walk_path, "GR00T-WholeBodyControl-Walk.onnx")

        # ── Load ONNX policies ───────────────────────────────────────────────
        self._balance_fn = None
        self._walk_fn = None
        if balance_path and walk_path:
            try:
                self._balance_fn = _load_policy(balance_path)
                self._walk_fn = _load_policy(walk_path)
                self.get_logger().info(f"WBC policies loaded (balance={balance_path})")
            except Exception as e:
                self.get_logger().error(f"Failed to load policies: {e}")
        else:
            self.get_logger().warning("Policy paths not found. Publishing default standing pose.")

        # ── State ────────────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self._history: collections.deque = collections.deque(
            [np.zeros(SINGLE_OBS_DIM, dtype=np.float32)] * OBS_HISTORY_LEN,
            maxlen=OBS_HISTORY_LEN,
        )

        # Sensor data
        self._qpos: Optional[np.ndarray] = None
        self._qvel: Optional[np.ndarray] = None
        self._quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._ang_vel_body = np.zeros(3, dtype=np.float32)

        # Locomotion command state
        self._cmd_vx = 0.0
        self._cmd_vy = 0.0
        self._cmd_wz = 0.0
        self._last_cmd_time = 0.0
        self._target_height = DEFAULT_HEIGHT_CMD
        self._emergency_stopped = False
        self._mode = "balance"
        self._stationary_time = 0.0
        self._dt = 1.0 / control_hz
        self._initialized_history = False


        # Relative pose move trajectory state (LocoClient.move / cmd_pose parity)
        self._pose_move_active = False
        self._pose_move_end_time = 0.0
        self._pose_move_vx = 0.0
        self._pose_move_vy = 0.0
        self._pose_move_wz = 0.0


        # ── QoS ─────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(JointState, js_topic, self._on_joint_states, sensor_qos)
        self.create_subscription(Imu, imu_topic, self._on_imu, sensor_qos)

        # Dual velocity topic subscription (/g1/cmd_vel and /cmd_vel for Nav2/teleop)
        self.create_subscription(Twist, cmdvel_topic, self._on_cmd_vel, reliable_qos)
        if cmdvel_topic != "/cmd_vel":
            self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, reliable_qos)

        # LocoClient-compatible command endpoints
        self.create_subscription(Twist, "/g1/cmd_pose", self._on_cmd_pose, reliable_qos)
        self.create_subscription(Bool, "/g1/emergency_stop", self._on_emergency_stop, reliable_qos)
        self.create_subscription(Bool, "/g1pilot/emergency_stop", self._on_emergency_stop, reliable_qos)
        self.create_subscription(Bool, "/g1/stand", self._on_stand, reliable_qos)
        self.create_subscription(Bool, "/g1pilot/start", self._on_stand, reliable_qos)
        self.create_subscription(Float64, "/g1/base_height", self._on_base_height, reliable_qos)
        self.create_subscription(Float64, "/base_height", self._on_base_height, reliable_qos)
        self.create_subscription(Empty, "/g1/stop", self._on_stop, reliable_qos)

        # ── Publishers ───────────────────────────────────────────────────────
        self._joint_pubs = {
            name: self.create_publisher(Float64, f"{prefix}/{name}", 10)
            for name in ALL_JOINTS
        }
        self._status_pub = self.create_publisher(String, "/g1/wbc_status", 10)
        self._fsm_pub = self.create_publisher(Int32, "/g1/fsm_id", 10)

        # ── Control timer ────────────────────────────────────────────────────
        period = 1.0 / control_hz
        self.create_timer(period, self._control_step)

        self.get_logger().info(
            f"WbcNode ready | js={js_topic} imu={imu_topic} cmd_vel={cmdvel_topic},/cmd_vel "
            f"cmd_pose=/g1/cmd_pose | watchdog={self.cmd_vel_timeout:.1f}s -> {prefix}/ @ {control_hz:.0f} Hz"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_joint_states(self, msg: JointState) -> None:
        with self._lock:
            if self._qpos is None:
                self._qpos = np.zeros(NUM_JOINTS, dtype=np.float32)
                self._qvel = np.zeros(NUM_JOINTS, dtype=np.float32)

            name_to_idx = {n: i for i, n in enumerate(msg.name)}
            for j_idx, j_name in enumerate(ALL_JOINTS):
                src = name_to_idx.get(j_name)
                if src is not None:
                    if src < len(msg.position):
                        self._qpos[j_idx] = msg.position[src]
                    if src < len(msg.velocity):
                        self._qvel[j_idx] = msg.velocity[src]

    def _on_imu(self, msg: Imu) -> None:
        q = msg.orientation
        self._quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
        av = msg.angular_velocity
        self._ang_vel_body = np.array([av.x, av.y, av.z], dtype=np.float32)

    def _on_cmd_vel(self, msg: Twist) -> None:
        with self._lock:
            # New direct velocity command interrupts any active relative pose move
            self._pose_move_active = False
            self._cmd_vx = float(msg.linear.x)
            self._cmd_vy = float(msg.linear.y)
            self._cmd_wz = float(msg.angular.z)
            self._last_cmd_time = time.time()
            self._emergency_stopped = False

    def _on_cmd_pose(self, msg: Twist) -> None:
        """Relative pose delta move (dx, dy in meters, dyaw in deg or rad)."""
        dx = float(msg.linear.x)
        dy = float(msg.linear.y)
        dyaw = float(msg.angular.z)

        # Auto-detect degrees vs radians (if > 2*pi, assume degrees)
        if abs(dyaw) > math.pi * 2.0:
            dyaw = math.radians(dyaw)

        dist = math.hypot(dx, dy)
        speed = self.default_move_speed
        yaw_rate = self.default_yaw_rate

        with self._lock:
            self._emergency_stopped = False
            if dist > 0.01:
                duration = dist / speed
                self._pose_move_active = True
                self._pose_move_end_time = time.time() + duration
                self._pose_move_vx = (dx / dist) * speed
                self._pose_move_vy = (dy / dist) * speed
                self._pose_move_wz = 0.0
                self.get_logger().info(f"Executing cmd_pose move: dx={dx:.2f}m, dy={dy:.2f}m, dist={dist:.2f}m ({duration:.2f}s)")
            elif abs(dyaw) > 0.01:
                duration = abs(dyaw) / yaw_rate
                self._pose_move_active = True
                self._pose_move_end_time = time.time() + duration
                self._pose_move_vx = 0.0
                self._pose_move_vy = 0.0
                self._pose_move_wz = math.copysign(yaw_rate, dyaw)
                self.get_logger().info(f"Executing cmd_pose rotate: dyaw={math.degrees(dyaw):.1f}° ({duration:.2f}s)")
            else:
                self._pose_move_active = False
                self._cmd_vx = 0.0
                self._cmd_vy = 0.0
                self._cmd_wz = 0.0

    def _on_emergency_stop(self, msg: Bool) -> None:
        with self._lock:
            self._emergency_stopped = bool(msg.data)
            self._pose_move_active = False
            self._cmd_vx = 0.0
            self._cmd_vy = 0.0
            self._cmd_wz = 0.0
            if self._emergency_stopped:
                self.get_logger().warn("EMERGENCY STOP activated - holding default stance.")

    def _on_stand(self, msg: Bool) -> None:
        with self._lock:
            if msg.data:
                self._emergency_stopped = False
                self._pose_move_active = False
                self._cmd_vx = 0.0
                self._cmd_vy = 0.0
                self._cmd_wz = 0.0
                self.get_logger().info("Reset to stationary balance / standing mode.")

    def _on_base_height(self, msg: Float64) -> None:
        with self._lock:
            target = max(0.50, min(0.85, float(msg.data)))
            self._target_height = target
            self.get_logger().info(f"Target standing height set to {target:.2f}m")

    def _on_stop(self, msg: Empty) -> None:
        with self._lock:
            self._pose_move_active = False
            self._cmd_vx = 0.0
            self._cmd_vy = 0.0
            self._cmd_wz = 0.0
            self.get_logger().info("Stop command received.")

    # ── Control step ──────────────────────────────────────────────────────────

    def _control_step(self) -> None:
        now = time.time()

        with self._lock:
            if self._qpos is None:
                self._publish_targets(DEFAULT_ANGLES)
                return

            qpos = self._qpos.copy()
            qvel = self._qvel.copy() if self._qvel is not None else np.zeros(NUM_JOINTS, dtype=np.float32)
            emergency = self._emergency_stopped

            # Resolve active velocity command
            if self._pose_move_active:
                if now < self._pose_move_end_time:
                    vx = self._pose_move_vx
                    vy = self._pose_move_vy
                    wz = self._pose_move_wz
                    self._mode = "pose_move"
                else:
                    # Move completed -> active zero-velocity stop
                    self._pose_move_active = False
                    self._cmd_vx = 0.0
                    self._cmd_vy = 0.0
                    self._cmd_wz = 0.0
                    vx, vy, wz = 0.0, 0.0, 0.0
                    self._mode = "balance"
                    self.get_logger().info("Relative pose move complete. Holding place.")
            else:
                # Check deadman watchdog timeout
                if (now - self._last_cmd_time) < self.cmd_vel_timeout:
                    vx, vy, wz = self._cmd_vx, self._cmd_vy, self._cmd_wz
                else:
                    # Timed out -> firmly hold stationary position
                    vx, vy, wz = 0.0, 0.0, 0.0

            target_height = self._target_height

        quat = self._quat_wxyz.copy()
        # IMU angular velocity is body-frame; rotate or scale accordingly
        ang_vel = self._ang_vel_body.copy()

        if emergency or self._balance_fn is None:
            self._publish_targets(DEFAULT_ANGLES)
            self._publish_status("damp" if emergency else "standby", FSM_DAMP if emergency else FSM_READY_STAND)
            return

        loco_cmd = np.array([vx, vy, wz], dtype=np.float32)
        cmd_magnitude = np.linalg.norm(loco_cmd)

        # Stationary Deadband & Stance Stabilization:
        # When stationary (no active teleop cmd), strictly hold standing pose
        if cmd_magnitude <= WALK_CMD_DEADBAND:
            loco_cmd = np.zeros(3, dtype=np.float32)
            self._stationary_time += self._dt
            self._mode = "balance"
            fsm_id = FSM_READY_STAND

        else:
            loco_cmd = np.array([vx, vy, wz], dtype=np.float32)
            self._stationary_time = 0.0
            self._mode = "walk" if not self._pose_move_active else "pose_move"
            fsm_id = FSM_WALK

        # Build observation & step policy
        single_obs = self._build_obs(qpos, qvel, quat, ang_vel, loco_cmd, target_height)
        if not self._initialized_history:
            self._history = collections.deque([single_obs.copy()] * OBS_HISTORY_LEN, maxlen=OBS_HISTORY_LEN)
            self._initialized_history = True
        else:
            self._history.append(single_obs)
        obs = np.concatenate(list(self._history))


        if cmd_magnitude <= WALK_CMD_DEADBAND:
            self._action = self._balance_fn(obs)
        else:
            self._action = self._walk_fn(obs)



        targets = self._action * ACTION_SCALE + DEFAULT_ANGLES
        self._publish_targets(targets)
        self._publish_status(self._mode, fsm_id)


    def _build_obs(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        quat: np.ndarray,
        ang_vel: np.ndarray,
        loco_cmd: np.ndarray,
        target_height: float,
    ) -> np.ndarray:
        command = np.zeros(7, dtype=np.float32)
        command[:3] = loco_cmd * CMD_SCALE
        command[3] = float(target_height)

        qj_scaled = (qpos - _PADDED_DEFAULT) * DOF_POS_SCALE
        dqj_scaled = qvel * DOF_VEL_SCALE
        grav = gravity_orientation(quat).astype(np.float32)
        omega = ang_vel * ANG_VEL_SCALE

        obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
        obs[0:7] = command
        obs[7:10] = omega
        obs[10:13] = grav
        obs[13: 13 + NUM_JOINTS] = qj_scaled
        obs[13 + NUM_JOINTS: 13 + 2*NUM_JOINTS] = dqj_scaled
        obs[13 + 2*NUM_JOINTS: 13 + 2*NUM_JOINTS + NUM_ACTIONS] = self._action
        return obs

    def _publish_targets(self, leg_targets: np.ndarray) -> None:
        for i, name in enumerate(LEG_WAIST_JOINTS):
            m = Float64()
            m.data = float(leg_targets[i])
            self._joint_pubs[name].publish(m)
        for name in ARM_JOINTS:
            m = Float64()
            m.data = 0.0
            self._joint_pubs[name].publish(m)

    def _publish_status(self, mode_str: str, fsm_id: int) -> None:
        status_msg = String()
        status_msg.data = mode_str
        self._status_pub.publish(status_msg)

        fsm_msg = Int32()
        fsm_msg.data = int(fsm_id)
        self._fsm_pub.publish(fsm_msg)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _resolve_policy(self, param_path: str, filename: str) -> Optional[str]:
        if param_path and Path(param_path).exists():
            return param_path

        here = Path(__file__).resolve()
        for ancestor in here.parents:
            candidate = ancestor.parent / "G1_sim" / "assets" / "policy" / filename
            if candidate.exists():
                return str(candidate)

        self.get_logger().warning(f"Policy not found: {filename}")
        return None


# ─────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = WbcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
