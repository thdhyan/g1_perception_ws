#!/usr/bin/env python3
"""
G1 robot bridge -- standalone process, no rclpy.

Owns the single unitree_sdk2py LocoClient connection and exposes it over a
local Unix domain socket as a tiny JSON-line RPC service. Runs OUTSIDE any
ROS process on purpose: rclpy's Node() and unitree_sdk2py's
ChannelFactoryInitialize() each load their own copy of libddsc.so, and
creating both DDS participants in one process segfaults (confirmed while
debugging loco_client.py / move_to_xy.py -- see
../../unitree_sdk2_python/docs/SDK_TROUBLESHOOTING.md).

Run this once, standalone (not via ros2 run):

    python3 robot_bridge.py enp2s0 [socket_path]

Then any process (ROS node or plain script) can send it commands:

    echo '{"cmd":"move","dx":0.3,"dy":0.0}' | nc -U /tmp/g1_robot_bridge.sock

Protocol: one JSON object per line in, one JSON object per line out.
Commands:
    {"cmd": "get_fsm"}                          -> {"ok": true, "fsm_id": 501}
    {"cmd": "ensure_walk_mode"}                  -> {"ok": true, "fsm_id": 501}
    {"cmd": "move", "dx": 1.0, "dy": 0.0}        -> {"ok": true}
    {"cmd": "rotate", "degrees": 180}            -> {"ok": true}
    {"cmd": "turn_imu", "delta_deg": 90, "yaw_rate": 0.25, "tol_deg": 1.0}
                                                 -> {"ok": true, "turned_deg", "residual_deg", "passes"}
      turn in place by delta_deg (+ = counter-clockwise), closed loop on the robot's IMU yaw
    {"cmd": "damp"}                              -> {"ok": true}
    {"cmd": "stop"}                              -> {"ok": true}   (also holds the arms)
    {"cmd": "kill"}                              -> {"ok": true}   VR kill switch: zero velocity,
      arms back to the loco controller (0.5 s), lock stand (FSM 4), KILL_LOCK_S later Damp()
    {"cmd": "set_velocity", "vx":.., "vy":.., "wz":.., "duration": 0.3, "auto_walk": false}
    {"cmd": "arm_traj", "names": [14 arm joints], "points": [[14], ...], "dt": 0.1}
    {"cmd": "arm_hold"} / {"cmd": "arm_release"} / {"cmd": "arm_state"}
      rt/arm_sdk planned arm motion, see arm_sdk_player.py
Errors:
    {"ok": false, "error": "..."}
"""
import json
import math
import os
import socket
import socketserver
import sys
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map

try:
    from arm_sdk_player import ArmSdkPlayer          # run as a script (the usual way)
except ImportError:
    from g1_control.arm_sdk_player import ArmSdkPlayer

FSM_ZERO_TORQUE = 0
FSM_DAMP = 1
FSM_READY_STAND = 4
FSM_WALK = 501

DEFAULT_LINEAR_SPEED = 0.20
DEFAULT_YAW_RATE = 0.5
ROTATE_COMPENSATION = 9.0 / 6.0
SETTLE_TIME = 2.0
KILL_LOCK_S = 2.0       # kill: locked standing this long, then damp
# turn_imu: the loco controller turns only ~0.6-0.8x the commanded wz, and by a different amount
# each run (foot slip), so a timed turn misses by up to 45 deg; this stops on the IMU instead.
TURN_IMU_RATE = 0.25    # rad/s
TURN_IMU_COAST_S = 0.1  # stop this early (x rate): the robot keeps turning after zero velocity
TURN_IMU_SETTLE_S = 0.1  # zero velocity this long before measuring a pass (was 0.8)
TURN_IMU_PASSES = 3     # first pass + corrections
TURN_IMU_TIMEOUT = 40.0  # s, whole turn

DEFAULT_SOCKET_PATH = "/tmp/g1_robot_bridge.sock"


class G1Robot:
    """Same logic as scripts/g1_motion_skill.py's G1Motion, kept local here
    so this bridge has no dependency on the unitree_sdk2_python repo's
    scripts/ directory being on PYTHONPATH."""

    def __init__(self, network_interface: str, domain_id: int = 0, timeout: float = 8.0):
        ChannelFactoryInitialize(domain_id, network_interface)
        self.client = LocoClient()
        self.client.SetTimeout(timeout)
        self.client.Init()

        ret = self.client.SwitchToUserCtrl()
        if ret != 0:
            raise RuntimeError(
                f"SwitchToUserCtrl failed (code={ret}). Power-cycle the robot "
                "if the service name / SDK version are already correct."
            )

        # Initialize Arm Action Client
        try:
            self.arm_client = G1ArmActionClient()
            self.arm_client.SetTimeout(timeout)
            self.arm_client.Init()
            print("[G1Robot] G1ArmActionClient initialized successfully.")
        except Exception as e:
            print(f"[G1Robot] Warning: Could not initialize G1ArmActionClient ({e}). LocoClient arm fallback will be used.")
            self.arm_client = None

        self._lock = threading.Lock()
        # Tracks whether set_velocity() has already put the robot in walk mode, so
        # a streaming caller does not re-poll the FSM on every command.
        self._walking = False
        # Set by stop() to cut short an in-flight move()/rotate(). Those stream
        # velocity for the whole traversal, so without a flag they can only be
        # interrupted by killing the bridge -- which is not an emergency stop.
        # Checked from a different thread than the one running the traversal,
        # which is why the server is threaded.
        self._abort = threading.Event()
        # rt/arm_sdk: slow planned arm trajectories from the VR teleop relay
        # (G1_sim/teleop/real_robot_relay.py). Idle (sends nothing) until the first arm_traj.
        self.arm_sdk = ArmSdkPlayer()

    def get_fsm(self) -> int:
        code, fsm_id = self.client.GetFsmId()
        if code != 0:
            raise RuntimeError(f"GetFsmId failed, code={code}")
        return fsm_id

    def _switch_to_walk_with_retry(self, max_attempts: int = 3) -> int:
        """SetFsmId(FSM_WALK) sometimes doesn't take on the first try even
        from a stable standing state (observed live: FSM stayed at 4 after
        one SETTLE_TIME wait, worked on a plain retry). Retry a few times
        before giving up."""
        last_fsm_id = None
        for attempt in range(1, max_attempts + 1):
            try:
                self.client.SetFsmId(FSM_WALK)
                time.sleep(SETTLE_TIME)
                last_fsm_id = self.get_fsm()
                if last_fsm_id == FSM_WALK:
                    return last_fsm_id
            except Exception as e:
                last_fsm_id = None
                print(f"[G1Robot] walk-mode attempt {attempt}/{max_attempts} raised: {e}")
        raise RuntimeError(
            f"Failed to reach walk mode after {max_attempts} attempts, "
            f"current FSM id={last_fsm_id}"
        )

    def ensure_walk_mode(self):
        fsm_id = self.get_fsm()
        if fsm_id == FSM_WALK:
            return fsm_id
        if fsm_id == FSM_READY_STAND:
            return self._switch_to_walk_with_retry()

        self.client.Damp()
        time.sleep(SETTLE_TIME)
        self.client.SetFsmId(FSM_READY_STAND)
        time.sleep(SETTLE_TIME)
        return self._switch_to_walk_with_retry()

    def set_velocity(self, vx: float, vy: float, wz: float, duration: float = 0.3,
                     auto_walk: bool = True):
        """One non-blocking velocity command, for teleop and any other continuous
        controller streaming at a fixed rate.

        move()/rotate() below are distance/angle goals: they block the dispatch
        lock for the whole traversal, so a caller sending 10 commands a second
        would queue minutes of motion. Here the caller owns the loop, and
        `duration` is the SDK's own watchdog -- if the stream stops, the robot
        halts on its own after that long without needing an explicit stop.

        ensure_walk_mode() is only called when leaving zero velocity, since it
        polls the FSM and would otherwise run on every streamed command.

        auto_walk=False (VR teleop) never switches FSM -- ensure_walk_mode() may Damp()
        the robot -- and refuses to move unless the robot is already in walk mode.
        """
        if not auto_walk and not self._walking:
            fsm_id = self.get_fsm()
            if fsm_id != FSM_WALK:
                raise RuntimeError(f"robot not in walk mode (fsm {fsm_id}, need {FSM_WALK}); "
                                   "switch it with the remote first")
            self._walking = True
        if abs(vx) > 1e-3 or abs(vy) > 1e-3 or abs(wz) > 1e-3:
            if not self._walking:
                self.ensure_walk_mode()
                self._walking = True
        self.client.SetVelocity(vx, vy, wz, duration=duration)

    def move(self, dx: float, dy: float, speed: float = DEFAULT_LINEAR_SPEED):
        self.ensure_walk_mode()
        distance = math.hypot(dx, dy)
        if distance <= 0.01:
            return
        v_speed = max(0.05, abs(speed))
        duration = distance / v_speed
        vx = (dx / distance) * v_speed
        vy = (dy / distance) * v_speed
        print(f"[G1Robot] Walking dx={dx:+.2f}m, dy={dy:+.2f}m (dist={distance:.2f}m) at {v_speed:.2f}m/s for duration={duration:.2f}s...")

        # 20 Hz continuous velocity streaming
        self._abort.clear()
        start_time = time.time()
        while time.time() - start_time < duration:
            if self._abort.is_set():
                print("[G1Robot] Walk aborted by stop request.")
                break
            self.client.SetVelocity(vx, vy, 0.0, duration=0.2)
            time.sleep(0.05)

        # Active graceful zero-velocity brake: Stream (0,0,0) for 0.6s to firmly plant feet
        brake_start = time.time()
        while time.time() - brake_start < 0.6:
            self.client.SetVelocity(0.0, 0.0, 0.0, duration=0.2)
            time.sleep(0.05)

        self.client.StopMove()
        time.sleep(0.2)
        print("[G1Robot] Walk complete. Fully halted at target standoff.")

    def rotate(self, degrees: float, yaw_rate: float = DEFAULT_YAW_RATE,
               compensation: float = ROTATE_COMPENSATION):
        self.ensure_walk_mode()
        target_rad = math.radians(degrees) * compensation
        if target_rad == 0:
            return
        rate = math.copysign(yaw_rate, target_rad)
        duration = abs(target_rad) / yaw_rate
        print(f"[G1Robot] Rotating {degrees:+.1f}° at {rate:.2f}rad/s for duration={duration:.2f}s...")

        # 20 Hz streaming
        self._abort.clear()
        start_time = time.time()
        while time.time() - start_time < duration:
            if self._abort.is_set():
                print("[G1Robot] Rotation aborted by stop request.")
                break
            self.client.SetVelocity(0.0, 0.0, rate, duration=0.2)
            time.sleep(0.05)

        # Active brake
        brake_start = time.time()
        while time.time() - brake_start < 0.5:
            self.client.SetVelocity(0.0, 0.0, 0.0, duration=0.2)
            time.sleep(0.05)

        self.client.StopMove()
        time.sleep(0.2)
        print("[G1Robot] Rotation complete. Fully halted.")

    def imu_yaw(self) -> float:
        """Torso IMU yaw (rad) from rt/lowstate, through the arm_sdk player's subscription."""
        st = self.arm_sdk._state
        if st is None or time.monotonic() - self.arm_sdk.state_t > 0.2:
            raise RuntimeError("no fresh rt/lowstate (IMU)")
        return st.imu_state.rpy[2]

    def turn_imu(self, delta_deg: float, yaw_rate: float = TURN_IMU_RATE, tol_deg: float = 1.0) -> dict:
        """Turn in place by delta_deg, closed loop on IMU yaw: stream wz until the IMU has turned
        the remaining angle (less the coast), settle, re-measure, correct. Aborted by stop()."""
        if not self._walking:
            self.ensure_walk_mode()
            self._walking = True
        self._abort.clear()
        delta, tol, coast = math.radians(delta_deg), math.radians(tol_deg), yaw_rate * TURN_IMU_COAST_S
        prev, turned, passes = self.imu_yaw(), 0.0, 0
        t_end = time.monotonic() + TURN_IMU_TIMEOUT

        def track():
            nonlocal prev, turned
            y = self.imu_yaw()
            d = (y - prev + math.pi) % (2 * math.pi) - math.pi
            prev, turned = y, turned + d
            return d

        def hold_zero(seconds):
            t = time.monotonic() + seconds
            while time.monotonic() < t:
                self.client.SetVelocity(0.0, 0.0, 0.0, duration=0.2)
                time.sleep(0.05)
                track()

        while passes < TURN_IMU_PASSES:
            remaining = delta - turned
            if abs(remaining) <= tol:
                break
            passes += 1
            sign = math.copysign(1.0, remaining)
            stop_at = max(abs(remaining) - coast, abs(remaining) / 2)
            progress = 0.0
            while progress < stop_at:
                if self._abort.is_set() or time.monotonic() > t_end:
                    hold_zero(0.3)
                    raise RuntimeError(f"turn_imu {'aborted by stop' if self._abort.is_set() else 'timed out'} "
                                       f"after {math.degrees(turned):+.1f} of {delta_deg:+.1f} deg")
                self.client.SetVelocity(0.0, 0.0, sign * yaw_rate, duration=0.2)
                time.sleep(0.05)
                progress += sign * track()
            hold_zero(TURN_IMU_SETTLE_S)
            print(f"[G1Robot] turn_imu pass {passes}: turned {math.degrees(turned):+.1f} of {delta_deg:+.1f} deg")
        return {"turned_deg": round(math.degrees(turned), 2),
                "residual_deg": round(math.degrees(delta - turned), 2), "passes": passes}

    def damp(self):
        self.client.Damp()

    def stop(self):
        # Signal first, brake second: a move()/rotate() already streaming velocity
        # on another thread must see the abort before this braking stream ends, or
        # it resumes its own commands straight afterwards.
        self._abort.set()
        # Stream 0 velocity to brake
        for _ in range(5):
            self.client.SetVelocity(0.0, 0.0, 0.0, duration=0.2)
            time.sleep(0.05)
        self.client.StopMove()
        self._walking = False

    def kill(self):
        """VR kill switch: brake, give the arms back, lock stand, then damp. Runs on its own thread."""
        print("[G1Robot] KILL: zero velocity, arms released, lock stand, then damp")
        self.arm_sdk.release(ramp_s=0.5)
        self.stop()
        try:
            self.client.SetFsmId(FSM_READY_STAND)
        except Exception as e:                      # damp regardless
            print(f"[G1Robot] KILL: lock stand failed ({e}), damping now")
        else:
            time.sleep(KILL_LOCK_S)
        self.client.Damp()
        print(f"[G1Robot] KILL done: FSM {self.get_fsm()}")

    def execute_arm_action(self, action_name_or_id):
        """Execute action via G1ArmActionClient if available, with action_map lookups."""
        action_id = None
        if isinstance(action_name_or_id, int):
            action_id = action_name_or_id
        elif isinstance(action_name_or_id, str):
            # Normalize string (e.g. "shake_hand" -> "shake hand", "face_wave" -> "face wave")
            normalized = action_name_or_id.replace("_", " ").lower().strip()
            action_id = action_map.get(normalized)
            if action_id is None:
                # Direct string search match
                for k, v in action_map.items():
                    if normalized in k or k in normalized:
                        action_id = v
                        break
        
        if action_id is None:
            raise ValueError(f"Unknown arm action: {action_name_or_id}")

        if self.arm_client is not None:
            print(f"[G1Robot] Executing arm action ID {action_id} via G1ArmActionClient...")
            ret = self.arm_client.ExecuteAction(action_id)
            if ret != 0:
                print(f"[G1Robot] G1ArmActionClient returned non-zero code: {ret}")
            return ret
        else:
            # Fallback to LocoClient if available
            print(f"[G1Robot] G1ArmActionClient not ready; attempting LocoClient fallback for action {action_name_or_id}")
            if action_id == 27:  # shake hand
                return self.client.ShakeHand()
            elif action_id in (25, 26):  # wave
                return self.client.WaveHand(False)
            return 0

    def shake_hand(self, hold_seconds: float = 3.0, auto_release: bool = True):
        """Execute handshake gesture."""
        print(f"[G1Robot] Initiating Handshake (hold={hold_seconds}s, auto_release={auto_release})...")
        if self.arm_client is not None:
            self.arm_client.ExecuteAction(action_map.get("shake hand", 27))
            time.sleep(hold_seconds)
            if auto_release:
                self.arm_client.ExecuteAction(action_map.get("release arm", 99))
        else:
            self.client.ShakeHand()
            time.sleep(hold_seconds)
            self.client.ShakeHand()

    def wave(self, wave_type: str = "face", hold_seconds: float = 3.0, auto_release: bool = True):
        """Execute waving gesture (face wave / low wave or high wave)."""
        print(f"[G1Robot] Initiating Wave (type={wave_type}, hold={hold_seconds}s, auto_release={auto_release})...")
        if self.arm_client is not None:
            if wave_type in ("high", "high_wave"):
                self.arm_client.ExecuteAction(action_map.get("high wave", 26))
            else:
                # face wave is low wave
                self.arm_client.ExecuteAction(action_map.get("face wave", 25))
            time.sleep(hold_seconds)
            if auto_release:
                self.arm_client.ExecuteAction(action_map.get("release arm", 99))
        else:
            self.client.WaveHand(False)
            time.sleep(hold_seconds)

    def release_arm(self):
        """Release arm torques / neutral position."""
        if self.arm_client is not None:
            self.arm_client.ExecuteAction(action_map.get("release arm", 99))


def make_handler(robot: G1Robot):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    resp = dispatch(robot, req)
                except Exception as e:
                    resp = {"ok": False, "error": str(e)}
                try:
                    self.wfile.write((json.dumps(resp) + "\n").encode())
                except (BrokenPipeError, ConnectionResetError):
                    # The client gave up waiting and closed the socket -- normal
                    # when a gesture outruns a caller's timeout. The command still
                    # ran; only the reply had nowhere to go. Without this the
                    # exception propagates out of handle() and prints a traceback
                    # per gesture.
                    return

    return Handler


class ThreadedUnixStreamServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """One thread per connection.

    The plain UnixStreamServer serves one client at a time, so a gesture that
    holds for 3.5s blocked every other command for that whole time -- a teleop
    client opening a connection per tick piled up a backlog that then executed
    against a stale world once the gesture finished. SDK calls are still
    serialised, by robot._lock inside dispatch().
    """

    daemon_threads = True


def dispatch(robot: G1Robot, req: dict) -> dict:
    cmd = req.get("cmd")

    # STOP BYPASSES THE LOCK, DELIBERATELY.
    # Every other command runs under robot._lock, and a blocking move()/rotate()
    # holds it for the entire traversal -- so a stop that waited for the lock
    # would queue behind the exact motion it is meant to interrupt and only
    # arrive after the robot had already walked the full distance. That is not an
    # emergency stop. Here it sets the abort flag the traversal loops poll and
    # streams zero velocity immediately.
    if cmd == "stop":
        robot.arm_sdk.hold()
        robot.stop()
        return {"ok": True}
    if cmd == "kill":                       # same reason as stop: never wait for the lock
        threading.Thread(target=robot.kill, daemon=True).start()
        return {"ok": True}
    # rt/arm_sdk commands have their own lock and never block: no robot._lock either.
    if cmd == "arm_traj":
        robot.arm_sdk.play(req["names"], req["points"], float(req["dt"]))
        return {"ok": True}
    if cmd == "arm_hold":
        robot.arm_sdk.hold()
        return {"ok": True}
    if cmd == "arm_release":
        robot.arm_sdk.release()
        return {"ok": True}
    if cmd == "arm_state":
        return {"ok": True, **robot.arm_sdk.status()}

    with robot._lock:
        if cmd == "get_fsm":
            return {"ok": True, "fsm_id": robot.get_fsm()}
        elif cmd == "ensure_walk_mode":
            return {"ok": True, "fsm_id": robot.ensure_walk_mode()}
        elif cmd == "move":
            robot.move(float(req.get("dx", 0.0)), float(req.get("dy", 0.0)),
                       float(req.get("speed", DEFAULT_LINEAR_SPEED)))
            return {"ok": True}
        elif cmd == "set_velocity":
            robot.set_velocity(float(req.get("vx", 0.0)), float(req.get("vy", 0.0)),
                               float(req.get("wz", 0.0)), float(req.get("duration", 0.3)),
                               bool(req.get("auto_walk", True)))
            return {"ok": True}
        elif cmd == "rotate":
            robot.rotate(float(req.get("degrees", 0.0)),
                         float(req.get("yaw_rate", DEFAULT_YAW_RATE)),
                         float(req.get("compensation", ROTATE_COMPENSATION)))
            return {"ok": True}
        elif cmd == "turn_imu":
            return {"ok": True, **robot.turn_imu(float(req["delta_deg"]),
                                                 float(req.get("yaw_rate", TURN_IMU_RATE)),
                                                 float(req.get("tol_deg", 1.0)))}
        elif cmd == "damp":
            robot.damp()
            return {"ok": True}
        elif cmd == "shake_hand":
            robot.shake_hand(
                float(req.get("hold_seconds", 3.0)),
                bool(req.get("auto_release", True))
            )
            return {"ok": True, "action": "shake_hand"}
        elif cmd == "wave":
            robot.wave(
                str(req.get("wave_type", "face")),
                float(req.get("hold_seconds", 3.0)),
                bool(req.get("auto_release", True))
            )
            return {"ok": True, "action": "wave"}
        elif cmd == "release_arm":
            robot.release_arm()
            return {"ok": True, "action": "release_arm"}
        elif cmd == "arm_action":
            ret = robot.execute_arm_action(req.get("action"))
            return {"ok": True, "ret": ret}
        elif cmd == "get_arm_actions":
            return {"ok": True, "actions": list(action_map.keys())}
        else:
            return {"ok": False, "error": f"unknown cmd: {cmd}"}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface [socket_path]")
        sys.exit(-1)

    network_interface = sys.argv[1]
    socket_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SOCKET_PATH

    print(f"Connecting to robot on {network_interface}...")
    robot = G1Robot(network_interface)
    print(f"Connected. FSM id: {robot.get_fsm()}")

    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = ThreadedUnixStreamServer(socket_path, make_handler(robot))
    print(f"Listening on {socket_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if os.path.exists(socket_path):
            os.remove(socket_path)


if __name__ == "__main__":
    main()
