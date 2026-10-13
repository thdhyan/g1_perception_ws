"""Slow, planned arm motion on the real G1 through ``rt/arm_sdk``, next to the loco controller.

The loco controller keeps the legs (and waist) while ``rt/arm_sdk`` overrides the 14 arm
joints with a blend weight (motor slot 29: 0 = loco controller, 1 = these commands), as in
unitree_sdk2_python's ``example/g1/high_level/g1_arm7_sdk_dds_example.py``. Only arm joints
are commanded; the waist stays with the loco controller.

Used by robot_bridge.py (same unitree_sdk2py DDS participant, so no rclpy in this process).
A 50 Hz thread plays a joint trajectory sent by the teleop relay:

* weight ramps 0 -> 1 over ``RAMP_IN_S`` while holding the measured arm pose, then the
  trajectory plays;
* every tick the commanded q moves at most ``MAX_QD`` rad/s towards the trajectory sample
  (second guard on top of the planner's own slow timing; a jump in a plan cannot pass);
* ``hold`` freezes at the current command, ``release`` ramps the weight back to 0 over
  ``RAMP_OUT_S`` (the loco controller then takes the arms back to its own pose).
"""
import threading
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
ARM_NAMES = [f"{s}_{j}_joint" for s in ("left", "right") for j in ARM]
ARM_MOTORS = list(range(15, 29))     # G1 29-DoF motor index of ARM_NAMES, in order
WEIGHT_SLOT = 29                     # kNotUsedJoint: arm_sdk blend weight
DT = 0.02
KP, KD = 60.0, 1.5                   # the SDK example's gains
MAX_QD = 0.6                         # rad/s, per-tick cap on the commanded q
RAMP_IN_S, RAMP_OUT_S = 1.0, 2.0


def step_toward(cur: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    return cur + np.clip(target - cur, -max_step, max_step)


def sample(points: np.ndarray, dt: float, t: float) -> np.ndarray:
    """Linear interpolation of ``points`` (N, 14) spaced ``dt`` s apart, clamped to the ends."""
    i = t / dt
    if i >= len(points) - 1:
        return points[-1]
    k = int(i)
    a = i - k
    return (1.0 - a) * points[k] + a * points[k + 1]


class ArmSdkPlayer:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = None                 # latest LowState_
        self.state_t = 0.0                 # monotonic time of _state (robot_bridge reads the IMU from it)
        self._crc = CRC()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self.weight = 0.0
        self._weight_goal = 0.0
        self.q_cmd = None                  # (14,) commanded arm q, None until weight leaves 0
        self._traj = None                  # (points, dt, t0)
        self.mode = "idle"                 # idle | ramp_in | playing | holding | releasing
        self._ramp_out = RAMP_OUT_S
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        threading.Thread(target=self._loop, daemon=True).start()

    def _on_state(self, msg):
        self._state = msg
        self.state_t = time.monotonic()

    def measured(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("no rt/lowstate yet")
        return np.array([self._state.motor_state[m].q for m in ARM_MOTORS], dtype=np.float64)

    # ── commands (robot_bridge dispatch) ──
    def play(self, names, points, dt: float):
        if list(names) != ARM_NAMES:
            raise ValueError(f"names must be the 14 arm joints in order {ARM_NAMES}")
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 14 or len(pts) < 1 or not np.isfinite(pts).all():
            raise ValueError("points must be (N, 14) finite")
        if not 0.005 <= dt <= 1.0:
            raise ValueError("dt out of range")
        with self._lock:
            if self.q_cmd is None:
                self.q_cmd = self.measured()
            self._weight_goal = 1.0
            self._traj = (pts, float(dt), None)       # starts once the weight is at 1
            self.mode = "ramp_in" if self.weight < 1.0 else "playing"

    def hold(self):
        with self._lock:
            self._traj = None
            if self.mode in ("ramp_in", "playing"):
                self.mode = "holding"

    def release(self, ramp_s: float = RAMP_OUT_S):
        with self._lock:
            self._traj = None
            self._weight_goal = 0.0
            self._ramp_out = ramp_s
            if self.weight > 0.0:
                self.mode = "releasing"

    def status(self) -> dict:
        with self._lock:
            traj = self._traj
            left = None
            if traj is not None and traj[2] is not None:
                left = max(0.0, (len(traj[0]) - 1) * traj[1] - (time.monotonic() - traj[2]))
            return {"mode": self.mode, "weight": round(self.weight, 3), "seconds_left": left,
                    "q_cmd": None if self.q_cmd is None else [round(float(v), 4) for v in self.q_cmd]}

    # ── 50 Hz writer ──
    def _loop(self):
        while True:
            t_next = time.monotonic() + DT
            try:
                self._tick()
            except Exception as e:                    # keep the writer alive; log and hold
                print(f"[ArmSdkPlayer] tick failed: {e}")
            time.sleep(max(0.0, t_next - time.monotonic()))

    def _tick(self):
        with self._lock:
            if self.weight == 0.0 and self._weight_goal == 0.0:
                self.q_cmd, self.mode = None, "idle"
                return                                 # nothing to send: loco controller owns the arms
            rate = DT / (RAMP_IN_S if self._weight_goal > self.weight else self._ramp_out)
            self.weight = float(np.clip(self._weight_goal, self.weight - rate, self.weight + rate))
            if self._traj is not None and self.weight >= 1.0:
                pts, dt, t0 = self._traj
                if t0 is None:
                    t0 = time.monotonic()
                    self._traj = (pts, dt, t0)
                    self.mode = "playing"
                t = time.monotonic() - t0
                self.q_cmd = step_toward(self.q_cmd, sample(pts, dt, t), MAX_QD * DT)
                if t >= (len(pts) - 1) * dt and np.allclose(self.q_cmd, pts[-1], atol=1e-3):
                    self._traj, self.mode = None, "holding"
            q, w = self.q_cmd.copy(), self.weight
        c = self._cmd
        c.motor_cmd[WEIGHT_SLOT].q = w
        for m, qi in zip(ARM_MOTORS, q):
            mc = c.motor_cmd[m]
            mc.q, mc.dq, mc.tau, mc.kp, mc.kd = float(qi), 0.0, 0.0, KP, KD
        c.crc = self._crc.Crc(c)
        self._pub.Write(c)
