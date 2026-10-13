"""Execute an approved intent plan on the G1, closed-loop on PhaseSpace pose.

Never uses the robot's odometry: the pose is the robot agent in /mocap/agents
(labeller output, PhaseSpace frame), and its heading must come from a captured
LED layout (yaw_source fit/tracker). Each pose goal's position is followed with the
sim-verified goto controller (goto_controller.py); on arrival the facing is turned
as one delta (goal yaw - PhaseSpace yaw) by robot_bridge turn_imu, closed loop on
the robot's IMU (the loco controller under-turns open loop). Commands go to
g1_control/robot_bridge.py's socket as set_velocity -> LocoClient.SetVelocity,
with the SDK watchdog `cmd_duration` as deadman. At the terminal call the robot
stops and does the arm gesture (talk/low_wave -> wave, shake_hand -> shake_hand).

Stops (zero velocity + bridge stop) on: /intent/estop, pose older than
`pose_timeout`, a human closer than `min_human_clearance` (pauses until clear).

dry_run:=true sends nothing to the bridge; the command goes to
/intent/dry_run_cmd_vel (Twist) where fake_phasespace integrates it.

In:  /intent/plan (latched JSON), /intent/approve (String: plan id), /intent/estop (Empty), /mocap/agents
Out: /intent/exec_state (JSON, 10 Hz), <run_dir>/execution_report.json
"""

import json
import math
import socket
import threading
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Empty, String

from g1_intent.goto_controller import YAW_TOL, GoalRunner, wrap

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
GESTURES = {"talk": {"cmd": "wave", "wave_type": "face"}, "low_wave": {"cmd": "wave", "wave_type": "face"},
            "shake_hand": {"cmd": "shake_hand"}}
ORIENTED = ("fit", "tracker")


def bridge_call(path, req, timeout=5.0):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(path)
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode() or "{}")


class GotoExecutor(Node):
    def __init__(self):
        super().__init__("goto_executor")
        p = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.dry_run = p("dry_run", False)
        self.socket = p("bridge_socket", "/tmp/g1_robot_bridge.sock")
        self.rate = p("rate", 10.0)
        self.pose_timeout = p("pose_timeout", 0.3)
        self.cmd_duration = p("cmd_duration", 0.3)
        self.clearance = p("min_human_clearance", 0.35)
        self.gesture = p("gesture", True)
        self.turn_rate = p("turn_rate", 0.25)  # rad/s, turn_imu

        self.lock = threading.Lock()
        self.plan = None
        self.runner = None
        self.turning = False  # a turn_imu call is in flight
        self.state, self.msg = "idle", ""
        self.pose, self.pose_t, self.humans = None, 0.0, []
        self.cmd = (0.0, 0.0, 0.0)
        self.timeline, self.t0, self.terminal_result = [], None, None

        self.dry_pub = self.create_publisher(Twist, "/intent/dry_run_cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "/intent/exec_state", 10)
        self.create_subscription(String, "/intent/plan", self.on_plan, LATCHED)
        self.create_subscription(String, "/intent/approve", self.on_approve, 10)
        self.create_subscription(Empty, "/intent/estop", self.on_estop, 10)
        self.create_subscription(String, "/mocap/agents", self.on_agents, 10)
        self.create_timer(1.0 / self.rate, self.tick)
        self.get_logger().info(f"goto_executor: {'DRY RUN' if self.dry_run else 'robot_bridge ' + self.socket}")

    # ── inputs ───────────────────────────────────────────────────────────
    def on_plan(self, msg):
        with self.lock:
            if self.state == "running":
                self.get_logger().warn("new plan while running: ignored until this one ends")
                return
            self.plan = json.loads(msg.data)
            self.set_state("idle", f"plan {self.plan['id']} waiting for approval")

    def on_agents(self, msg):
        d = json.loads(msg.data)
        robot = [a for a in d["agents"] if a["role"] == "robot"]
        with self.lock:
            self.humans = [a for a in d["agents"] if a["role"] == "human"]
            if robot and robot[0]["yaw_deg"] is not None and robot[0]["yaw_source"] in ORIENTED:
                r = robot[0]
                self.pose, self.pose_t = (r["x"], r["y"], math.radians(r["yaw_deg"])), time.monotonic()

    def on_approve(self, msg):
        with self.lock:
            if self.plan is None or msg.data.strip() != self.plan["id"]:
                self.get_logger().warn(f"approve {msg.data!r}: no such pending plan")
                return
            if self.state == "running":
                return
            if self.pose is None or time.monotonic() - self.pose_t > self.pose_timeout:
                self.set_state("aborted", "no fresh oriented robot pose (capture the robot hat in the labeler)")
                return
            if not self.dry_run:
                try:
                    r = bridge_call(self.socket, {"cmd": "ensure_walk_mode"}, timeout=15.0)
                    if not r.get("ok"):
                        raise RuntimeError(r.get("error"))
                except Exception as e:
                    self.set_state("aborted", f"robot_bridge: {e}")
                    return
            # dry run: no IMU, so the facing stays in the mocap controller
            self.runner = GoalRunner(self.plan["goals"], final_yaw=self.dry_run)
            self.turning = False
            self.timeline, self.t0, self.terminal_result = [], time.monotonic(), None
            self.set_state("running", f"executing {len(self.plan['goals'])} goals")

    def on_estop(self, _msg):
        # bridge stop first: it bypasses the bridge lock, which a turn_imu in flight holds
        if not self.dry_run:
            try:
                bridge_call(self.socket, {"cmd": "stop"})
            except Exception as e:
                self.get_logger().error(f"bridge stop failed: {e}")
        self.send(0.0, 0.0, 0.0)
        with self.lock:
            if self.state in ("running", "gesture"):
                self.set_state("estopped", "E-STOP")
                self.write_report()
            else:
                self.set_state("estopped", "E-STOP (was idle)")

    # ── loop ─────────────────────────────────────────────────────────────
    def set_state(self, state, msg):
        self.state, self.msg = state, msg
        self.get_logger().info(f"[{state}] {msg}")

    def send(self, vx, vy, wz):
        self.cmd = (vx, vy, wz)
        if self.dry_run:
            t = Twist()
            t.linear.x, t.linear.y, t.angular.z = vx, vy, wz
            self.dry_pub.publish(t)
            return
        try:
            bridge_call(self.socket, {"cmd": "set_velocity", "vx": vx, "vy": vy, "wz": wz,
                                      "duration": self.cmd_duration}, timeout=1.0)
        except Exception as e:  # SDK watchdog stops the robot after cmd_duration anyway
            self.get_logger().error(f"set_velocity failed: {e}", throttle_duration_sec=1.0)

    def tick(self):
        with self.lock:
            if self.state == "running":
                self.step()
            pose = self.pose
            state = {"plan_id": self.plan["id"] if self.plan else None, "state": self.state, "msg": self.msg,
                     "goal": self.runner.i if self.runner else None,
                     "n_goals": len(self.runner.goals) if self.runner else 0,
                     "phase": self.runner.phase if self.runner else None,
                     "pose": None if pose is None else [round(pose[0], 3), round(pose[1], 3),
                                                        round(math.degrees(pose[2]), 1)],
                     "cmd": [round(v, 3) for v in self.cmd], "dry_run": self.dry_run}
        self.state_pub.publish(String(data=json.dumps(state)))

    def step(self):
        t = time.monotonic() - self.t0
        if self.turning:
            return
        if self.pose is None or time.monotonic() - self.pose_t > self.pose_timeout:
            self.send(0.0, 0.0, 0.0)
            self.msg = "robot pose lost: holding"
            return
        x, y, yaw = self.pose
        near = [h["name"] for h in self.humans if math.hypot(h["x"] - x, h["y"] - y) < self.clearance]
        if near:
            self.send(0.0, 0.0, 0.0)
            self.msg = f"paused: {near} closer than {self.clearance} m"
            self.runner.t_goal = None if self.runner.t_goal is None else self.runner.t_goal + 1.0 / self.rate
            return
        cmd = self.runner.step(t, x, y, yaw)
        if self.runner.phase == "turn":
            g = self.runner.goals[self.runner.i]
            delta = math.degrees(wrap(math.radians(g["yaw_deg"]) - yaw))
            self.turning, self.cmd = True, (0.0, 0.0, 0.0)
            self.msg = f"goal {self.runner.i + 1}/{len(self.runner.goals)} turn {delta:+.1f} deg (IMU)"
            self.timeline.append({"t": round(t, 2), "goal": self.runner.i, "phase": "turn",
                                  "pose": [round(x, 3), round(y, 3), round(math.degrees(yaw), 1)],
                                  "turn_delta_deg": round(delta, 1)})
            threading.Thread(target=self.do_turn, args=(self.runner.i, delta), daemon=True).start()
            return
        self.send(*cmd)
        g = self.runner.i
        self.msg = f"goal {min(g + 1, len(self.runner.goals))}/{len(self.runner.goals)} {self.runner.phase}"
        self.timeline.append({"t": round(t, 2), "goal": g, "phase": self.runner.phase,
                              "pose": [round(x, 3), round(y, 3), round(math.degrees(yaw), 1)],
                              "cmd": [round(v, 3) for v in cmd]})
        if self.runner.done:
            self.send(0.0, 0.0, 0.0)
            self.terminal()

    def do_turn(self, i, delta):
        try:
            r = bridge_call(self.socket, {"cmd": "turn_imu", "delta_deg": delta, "yaw_rate": self.turn_rate,
                                          "tol_deg": math.degrees(YAW_TOL)}, timeout=45.0)
        except Exception as e:
            r = {"ok": False, "error": str(e)}
        with self.lock:
            self.turning = False
            if self.state != "running" or self.runner.i != i:
                return  # e-stopped meanwhile
            g = self.runner.goals[i]
            g["imu_turn"] = dict(r, delta_deg=round(delta, 1))
            ok = r.get("ok") and abs(r.get("residual_deg", 1e9)) <= math.degrees(YAW_TOL)
            x, y, yaw = self.pose if self.pose else (float("nan"),) * 3
            self.runner.finish(time.monotonic() - self.t0, "reached" if ok else "turn_failed", x, y, yaw)
            self.get_logger().info(f"goal {i + 1} turn {delta:+.1f} deg: {r}")
            if self.runner.done:
                self.send(0.0, 0.0, 0.0)
                self.terminal()

    def terminal(self):
        term = self.plan.get("terminal")
        if term is None:
            self.set_state("done", "all goals done, no terminal call")
            self.write_report()
            return
        x, y, _ = self.pose
        dists = {h["name"]: round(math.hypot(h["x"] - x, h["y"] - y), 2) for h in self.humans}
        targets = term["args"].get("target_names", [])
        self.terminal_result = {"call": term, "distances_m": {n: dists.get(n) for n in targets}}
        action = term["args"].get("action") if term["function"] == "interact" else None
        gesture = GESTURES.get(action) if self.gesture else None
        if gesture is None:
            self.set_state("done", f"{term['function']}({targets}) logged; distances {self.terminal_result['distances_m']}")
            self.write_report()
            return
        self.set_state("gesture", f"{action} -> {gesture['cmd']}")
        threading.Thread(target=self.do_gesture, args=(gesture,), daemon=True).start()

    def do_gesture(self, gesture):
        ok, err = True, None
        if not self.dry_run:
            try:
                r = bridge_call(self.socket, dict(gesture, hold_seconds=3.0), timeout=20.0)
                ok, err = bool(r.get("ok")), r.get("error")
            except Exception as e:
                ok, err = False, str(e)
        with self.lock:
            self.terminal_result["gesture"] = {**gesture, "ok": ok, "error": err, "dry_run": self.dry_run}
            if self.state == "gesture":
                self.set_state("done", f"gesture {gesture['cmd']} {'ok' if ok else 'FAILED: ' + str(err)}")
            self.write_report()

    def write_report(self):
        if not self.plan or not self.runner:
            return
        path = Path(self.plan["run_dir"]) / "execution_report.json"
        path.write_text(json.dumps({
            "plan_id": self.plan["id"], "state": self.state, "msg": self.msg, "dry_run": self.dry_run,
            "pose_source": "PhaseSpace /mocap/agents (no odometry)",
            "controller": "g1_intent.goto_controller (port of G1_sim goto_trace) -> LocoClient.SetVelocity; "
                          "facing: robot_bridge turn_imu (IMU closed loop)",
            "all_reached": all(g["status"] == "reached" for g in self.runner.goals),
            "goals": self.runner.goals, "terminal": self.terminal_result, "timeline": self.timeline,
        }, indent=1))
        self.get_logger().info(f"report -> {path}")


def main():
    rclpy.init()
    node = GotoExecutor()
    executor = MultiThreadedExecutor()  # estop must not wait behind a slow bridge call
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.send(0.0, 0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
